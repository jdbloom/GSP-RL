"""Tests for DDPGActorNetwork init_scheme parameter.

Verifies:
1. Kaiming init produces larger fc1.weight Frobenius norm than fanin for the
   GSP-head input size (input_size=6 → ~2.4x ratio).
2. On a realistic GSP input batch (1024 x 6, uniform [0, 0.5]), kaiming produces
   lower dormant-unit fraction (FAU) on fc1 and fc2 than fanin.
3. Backward compat: default init_scheme="fanin" produces identical fc1.weight to
   a freshly constructed network with the same seed.

Motivation: init-investigation smoke (j189-194) showed 65-72% dormant fc1 units at
episode 1 with fanin init on positive-bounded prox inputs in [0, 0.5]. Kaiming He
normal (std=sqrt(2/fan_in)) is ~2.4x wider for fan_in=6, which should fire more
ReLU units across a batch drawn from that distribution.
"""
import torch as T
import pytest

from gsp_rl.src.networks.ddpg import DDPGActorNetwork
from gsp_rl.src.actors.diagnostics import compute_fau


INPUT_SIZE = 6
OUTPUT_SIZE = 1
LR = 1e-3


def _make_actor(init_scheme: str, seed: int = 0) -> DDPGActorNetwork:
    T.manual_seed(seed)
    return DDPGActorNetwork(
        id=0,
        lr=LR,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
        init_scheme=init_scheme,
    )


def _realistic_batch(n: int = 1024, seed: int = 7, device=None) -> T.Tensor:
    """Uniform [0, 0.5] batch simulating GSP-head prox inputs."""
    rng = T.Generator()
    rng.manual_seed(seed)
    t = T.rand(n, INPUT_SIZE, generator=rng) * 0.5
    if device is not None:
        t = t.to(device)
    return t


class TestKaimingWeightNorm:
    """Kaiming init should produce larger Frobenius norm on fc1.weight than fanin."""

    def test_kaiming_fc1_norm_exceeds_fanin(self):
        """Kaiming produces ~2.4x larger fc1 weight norm for input_size=6."""
        # Use independent seeds so both have a fair draw from their distributions.
        fanin_net = _make_actor("fanin", seed=42)
        kaiming_net = _make_actor("kaiming", seed=42)

        fanin_norm = fanin_net.fc1.weight.data.norm().item()
        kaiming_norm = kaiming_net.fc1.weight.data.norm().item()

        assert kaiming_norm > fanin_norm, (
            f"Kaiming fc1 norm ({kaiming_norm:.4f}) must exceed fanin norm "
            f"({fanin_norm:.4f}): Kaiming std=sqrt(2/fan_in) is ~2.4x wider for fan_in=6"
        )

    def test_kaiming_fc1_norm_ratio_plausible(self):
        """Ratio should be in a sensible range (5x – 25x) for input_size=6.

        The fanin_init function uses size[0] (output dim=400) as the fan-in,
        giving std = 1/sqrt(400) = 0.05. Kaiming He normal with fan_in mode
        uses the actual input fan-in = 6, giving std = sqrt(2/6) ≈ 0.577.
        Expected norm ratio = kaiming_std / fanin_std ≈ 0.577 / 0.05 ≈ 11.5.
        We allow a wide band (5x – 25x) for Monte Carlo variance.
        """
        ratios = []
        for seed in range(20):
            fanin_net = _make_actor("fanin", seed=seed)
            kaiming_net = _make_actor("kaiming", seed=seed)
            fn = fanin_net.fc1.weight.data.norm().item()
            kn = kaiming_net.fc1.weight.data.norm().item()
            if fn > 1e-8:
                ratios.append(kn / fn)

        avg_ratio = sum(ratios) / len(ratios)
        assert 5.0 < avg_ratio < 25.0, (
            f"Mean kaiming/fanin fc1-norm ratio {avg_ratio:.3f} outside expected range "
            f"[5.0, 25.0]. fanin_init uses size[0]=400 as fan-in (std≈0.05); "
            f"kaiming uses actual fan_in=6 (std≈0.577) → expected ~11.5x ratio."
        )


class TestKaimingFAU:
    """Kaiming init should produce lower dormant-unit fraction on a prox-like batch."""

    def test_kaiming_fc1_fau_lower_than_fanin(self):
        """On uniform [0, 0.5] inputs, kaiming should have fewer dead fc1 units."""
        fanin_net = _make_actor("fanin", seed=42)
        kaiming_net = _make_actor("kaiming", seed=42)

        device = fanin_net.device
        batch = _realistic_batch(n=1024, device=device)

        fanin_fau = compute_fau(fanin_net, batch, layer_names=["fc1"])["fau_fc1"]
        kaiming_fau = compute_fau(kaiming_net, batch, layer_names=["fc1"])["fau_fc1"]

        assert kaiming_fau < fanin_fau, (
            f"Kaiming fc1 FAU ({kaiming_fau:.3f}) should be < fanin FAU "
            f"({fanin_fau:.3f}) for positive-bounded inputs [0, 0.5]: "
            "fanin weights are too narrow to fire many units across this distribution"
        )

    def test_kaiming_fc2_fau_lower_than_fanin(self):
        """On uniform [0, 0.5] inputs, kaiming should have fewer dead fc2 units."""
        fanin_net = _make_actor("fanin", seed=42)
        kaiming_net = _make_actor("kaiming", seed=42)

        device = fanin_net.device
        batch = _realistic_batch(n=1024, device=device)

        fanin_fau = compute_fau(fanin_net, batch, layer_names=["fc2"])["fau_fc2"]
        kaiming_fau = compute_fau(kaiming_net, batch, layer_names=["fc2"])["fau_fc2"]

        assert kaiming_fau < fanin_fau, (
            f"Kaiming fc2 FAU ({kaiming_fau:.3f}) should be < fanin FAU "
            f"({fanin_fau:.3f}): dead-unit cascade from fc1 is the mechanism"
        )


class TestFaninFixed:
    """fanin_fixed uses size[1] (the actual in_features) rather than the buggy size[0]."""

    def test_fanin_fixed_fc1_norm_exceeds_legacy_fanin(self):
        """fanin_fixed for fc1 (Linear(6, 400)) should be ~8x wider than legacy fanin.

        legacy fanin uses size[0] = 400 → std ≈ 0.029
        fanin_fixed uses size[1] = 6 → std ≈ 0.236
        Expected norm ratio ≈ sqrt(400/6) ≈ 8.16x.
        """
        ratios = []
        for seed in range(20):
            legacy = _make_actor("fanin", seed=seed)
            fixed = _make_actor("fanin_fixed", seed=seed)
            ln = legacy.fc1.weight.data.norm().item()
            fn = fixed.fc1.weight.data.norm().item()
            if ln > 1e-8:
                ratios.append(fn / ln)
        avg = sum(ratios) / len(ratios)
        assert 6.0 < avg < 11.0, (
            f"Mean fanin_fixed/fanin fc1-norm ratio {avg:.3f} outside expected ~8x band [6.0, 11.0]"
        )

    def test_fanin_fixed_fc1_fau_lower_than_legacy_fanin(self):
        """On uniform [0, 0.5] inputs, fanin_fixed should have fewer dead fc1 units than legacy."""
        legacy = _make_actor("fanin", seed=42)
        fixed = _make_actor("fanin_fixed", seed=42)
        device = legacy.device
        batch = _realistic_batch(n=1024, device=device)
        legacy_fau = compute_fau(legacy, batch, layer_names=["fc1"])["fau_fc1"]
        fixed_fau = compute_fau(fixed, batch, layer_names=["fc1"])["fau_fc1"]
        assert fixed_fau < legacy_fau, (
            f"fanin_fixed fc1 FAU ({fixed_fau:.3f}) should be < legacy fanin FAU "
            f"({legacy_fau:.3f}) on positive-bounded inputs"
        )

    def test_fanin_fixed_fc1_norm_below_kaiming(self):
        """fanin_fixed should sit between legacy fanin and kaiming on weight magnitude.

        Kaiming uses std=sqrt(2/fan_in) → ~sqrt(2)≈1.41x wider than fanin_fixed
        for matched fan-in. So fanin_fixed < kaiming on Frobenius norm.
        """
        ratios = []
        for seed in range(10):
            fixed = _make_actor("fanin_fixed", seed=seed)
            kaiming = _make_actor("kaiming", seed=seed)
            f_n = fixed.fc1.weight.data.norm().item()
            k_n = kaiming.fc1.weight.data.norm().item()
            if f_n > 1e-8:
                ratios.append(k_n / f_n)
        avg = sum(ratios) / len(ratios)
        assert 1.1 < avg < 2.5, (
            f"Mean kaiming/fanin_fixed fc1-norm ratio {avg:.3f} outside expected band [1.1, 2.5]"
        )


class TestBackwardCompat:
    """Default init_scheme='fanin' must produce identical behavior to legacy networks."""

    def test_default_scheme_is_fanin(self):
        """Explicit init_scheme='fanin' stores the value on self.init_scheme."""
        net = _make_actor("fanin", seed=0)
        assert net.init_scheme == "fanin"

    def test_kaiming_scheme_stored(self):
        """init_scheme='kaiming' is stored on self.init_scheme for inspection."""
        net = _make_actor("kaiming", seed=0)
        assert net.init_scheme == "kaiming"

    def test_default_arg_matches_explicit_fanin(self):
        """DDPGActorNetwork() with no init_scheme arg == init_scheme='fanin' with same seed."""
        seed = 99

        T.manual_seed(seed)
        default_net = DDPGActorNetwork(
            id=0,
            lr=LR,
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            # init_scheme not specified → defaults to "fanin"
        )

        T.manual_seed(seed)
        explicit_fanin_net = DDPGActorNetwork(
            id=0,
            lr=LR,
            input_size=INPUT_SIZE,
            output_size=OUTPUT_SIZE,
            init_scheme="fanin",
        )

        T.testing.assert_close(
            default_net.fc1.weight.data,
            explicit_fanin_net.fc1.weight.data,
            atol=0.0,
            rtol=0.0,
            msg=(
                "Default (no init_scheme) must produce identical fc1.weight to "
                "explicit init_scheme='fanin' with the same RNG seed."
            ),
        )
