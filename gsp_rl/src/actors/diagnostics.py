"""Per-episode plasticity and representation diagnostics.

Pure functions that take a network + a fixed eval batch of states, return dicts of
Python floats. Designed to be called once per diagnostic episode and the result
dict dropped into HDF5Logger as per-episode attrs.

See docs/specs/2026-04-17-diagnostics-instrumentation.md for the spec.

Metrics
-------
- ``compute_fau``           dormant-neuron fraction per ReLU layer (Sokar 2023)
- ``compute_overactive_fau``over-active fraction per ReLU layer (Qin 2024 MARL signal)
- ``compute_weight_norms``  L2 (Frobenius) norm per layer — Lyle 2024 regenerative criterion
- ``compute_effective_rank``99% singular-value threshold on penultimate activations
- ``compute_q_action_gap``  Q(a*) − Q(a_next) over eval batch — Weng-Lee 2026 signal
- ``compute_gsp_pred_diversity`` Shannon entropy of binned predictions — catches collapse-to-constant

Design notes
------------
- All forward passes run under ``torch.no_grad()`` — diagnostics never perturb training.
- Functions tolerate any ``nn.Module`` that has named submodule attributes matching
  the ``layer_names`` list (e.g., ``net.fc1``, ``net.fc2``). They do NOT assume any
  specific architecture class.
- For ``compute_effective_rank``, callers specify a method name (``penultimate_fn``)
  that returns the penultimate-layer activations. Production networks should expose
  such a method or the compute call should use a forward-hook alternative.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------------------
# Dormant-neuron fraction (FAU = Fraction of Active Units — but we report the DORMANT
# fraction, i.e., 1 - FAU_active, to match the Sokar 2023 convention)
# --------------------------------------------------------------------------------------

def compute_fau(
    net: nn.Module,
    eval_batch: torch.Tensor,
    layer_names: list[str],
    tau_dead: float = 0.1,
) -> dict[str, float]:
    """Per-layer fraction of units whose post-ReLU activation is below ``tau_dead``
    for EVERY sample in the eval batch (Sokar 2023 definition).

    A unit is "dormant" iff its activation ≤ tau_dead across all samples. Since ReLU
    clamps the post-ReLU activation to [0, ∞), tau_dead=0.1 catches units that are
    never meaningfully firing.

    Args:
        net: network with named ``nn.Linear`` submodules matching ``layer_names``.
            The forward pass must apply ReLU immediately after each named layer;
            the canonical DQN/DDQN/DDPG trunks we use do exactly this.
        eval_batch: ``(N, input_dim)`` tensor of states to measure against.
        layer_names: ordered list of linear layer attribute names to probe.
        tau_dead: activation threshold below which a unit counts as dead. 0.1 per
            Sokar 2023's τ.

    Returns:
        ``{"fau_<layer_name>": float in [0, 1]}`` for each layer.
    """
    net.eval()
    device = eval_batch.device
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(_module, _inp, out):
            # Capture post-Linear output; the network's own forward applies ReLU after.
            # We apply ReLU here ourselves to inspect the post-activation values.
            activations[name] = torch.relu(out).detach()
        return hook

    for name in layer_names:
        layer = getattr(net, name)
        hooks.append(layer.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            _ = net(eval_batch.to(device))
    finally:
        for h in hooks:
            h.remove()

    result: dict[str, float] = {}
    for name in layer_names:
        if name not in activations:
            result[f"fau_{name}"] = float("nan")
            continue
        act = activations[name]  # shape (N, layer_dim)
        # A unit is dormant if its activation ≤ tau_dead for ALL samples.
        per_sample_dead = (act <= tau_dead)  # (N, D)
        unit_always_dead = per_sample_dead.all(dim=0)  # (D,)
        result[f"fau_{name}"] = float(unit_always_dead.float().mean().item())
    return result


def compute_overactive_fau(
    net: nn.Module,
    eval_batch: torch.Tensor,
    layer_names: list[str],
    tau_over: float = 0.9,
) -> dict[str, float]:
    """Per-layer fraction of units whose post-ReLU activation is above
    ``tau_over * max_activation_in_layer`` for EVERY sample in the batch.

    This is the Qin 2024 MARL-specific signal: shared-policy networks often develop
    a small number of "dominant" units that always saturate, suppressing learning
    in the rest of the layer. Naive ReDo resets only dormant units and misses this.

    Args:
        tau_over: fraction of per-layer max activation above which a unit is
            considered over-active.
    """
    net.eval()
    device = eval_batch.device
    activations: dict[str, torch.Tensor] = {}
    hooks = []

    def _make_hook(name: str):
        def hook(_module, _inp, out):
            activations[name] = torch.relu(out).detach()
        return hook

    for name in layer_names:
        layer = getattr(net, name)
        hooks.append(layer.register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            _ = net(eval_batch.to(device))
    finally:
        for h in hooks:
            h.remove()

    result: dict[str, float] = {}
    for name in layer_names:
        if name not in activations:
            result[f"overactive_{name}"] = float("nan")
            continue
        act = activations[name]  # (N, D)
        # Per-layer max across all samples and units
        layer_max = act.max().item()
        if layer_max <= 1e-8:
            # Layer completely dead — no over-active units by definition
            result[f"overactive_{name}"] = 0.0
            continue
        threshold = tau_over * layer_max
        per_sample_over = (act >= threshold)  # (N, D)
        unit_always_over = per_sample_over.all(dim=0)  # (D,)
        result[f"overactive_{name}"] = float(unit_always_over.float().mean().item())
    return result


# --------------------------------------------------------------------------------------
# Weight norms (L2 / Frobenius per layer)
# --------------------------------------------------------------------------------------

def compute_weight_norms(
    net: nn.Module,
    layer_names: list[str],
) -> dict[str, float]:
    """Frobenius norm ‖W‖ for each named layer's weight matrix.

    Lyle 2024's "regenerative weight-norm criterion": plasticity interventions that
    produce sub-linear weight-norm growth across training transfer across tasks;
    those that don't, don't. Logging ‖W‖ per layer per episode is the data needed
    to evaluate that criterion on any training run.
    """
    result: dict[str, float] = {}
    for name in layer_names:
        try:
            layer = getattr(net, name)
            w = layer.weight.detach()
            result[f"wnorm_{name}"] = float(torch.linalg.norm(w).item())
        except AttributeError:
            result[f"wnorm_{name}"] = float("nan")
    return result


# --------------------------------------------------------------------------------------
# Effective rank of penultimate activations (99% SVD threshold)
# --------------------------------------------------------------------------------------

def compute_effective_rank(
    net: nn.Module,
    eval_batch: torch.Tensor,
    penultimate_fn: str = "penultimate",
    threshold: float = 0.99,
) -> float:
    """Number of singular values of the penultimate-activation matrix required to
    cover ``threshold`` fraction of the total variance.

    Representation rank is the empirical handle on Lyle 2023's "feature rank
    collapse is the predominant failure mode of RL trunks" finding. Kumar 2020
    uses this exact metric.

    Args:
        net: must expose a method named ``penultimate_fn`` (default ``"penultimate"``)
            that returns the post-ReLU activations of the layer immediately before
            the output projection.
        eval_batch: ``(N, input_dim)`` states.
        threshold: cumulative-variance cutoff (0.99 per Kumar 2020).

    Returns:
        Effective rank as an integer in ``[1, D]`` where D is penultimate dim.
        Returned as float for h5 attr compatibility.
    """
    net.eval()
    fn = getattr(net, penultimate_fn)
    with torch.no_grad():
        acts = fn(eval_batch)  # (N, D)
    acts_np = acts.detach().cpu().numpy().astype(np.float64)

    # Center the activations per-feature — standard pre-SVD step for rank estimation
    acts_centered = acts_np - acts_np.mean(axis=0, keepdims=True)

    # If the batch is entirely constant or zero, rank is 0 — cover it explicitly
    if np.allclose(acts_centered, 0.0, atol=1e-10):
        return 0.0

    # Singular-value decomposition. np.linalg.svd returns singular values in descending order.
    try:
        _, s, _ = np.linalg.svd(acts_centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")

    # Cumulative variance explained = cumsum(s^2) / sum(s^2)
    s_squared = s ** 2
    total = s_squared.sum()
    if total <= 1e-20:
        return 0.0
    cumvar = np.cumsum(s_squared) / total
    # Number of singular values needed to reach threshold
    rank = int(np.searchsorted(cumvar, threshold)) + 1
    return float(rank)


# --------------------------------------------------------------------------------------
# Q-value action gap (Weng-Lee 2026 cooperation-collapse signal)
# --------------------------------------------------------------------------------------

def compute_q_action_gap(
    q_network: nn.Module,
    eval_batch: torch.Tensor,
) -> dict[str, float]:
    """Q-value gap between the best and second-best action, averaged over the batch.

    Weng & Lee 2026 "Cooperation Collapse in Shared-Policy Multi-Agent DQN" identifies
    ``Q(a*) − Q(a_next_best) → 0`` as the representational-entanglement signature that
    is ORTHOGONAL to plasticity loss (dormant-neuron fraction). Dropping to 0 while
    FAU stays healthy means the network hasn't lost capacity — it's lost
    decidability between behaviors.

    Returns:
        ``q_action_gap_mean``, ``q_action_gap_std``, ``q_max_mean``
    """
    q_network.eval()
    with torch.no_grad():
        q = q_network(eval_batch)  # (N, A)
    # Sort per-sample descending, take top two
    sorted_q, _ = torch.sort(q, dim=1, descending=True)
    best = sorted_q[:, 0]
    second = sorted_q[:, 1] if sorted_q.shape[1] >= 2 else torch.zeros_like(best)
    gaps = best - second

    return {
        "q_action_gap_mean": float(gaps.mean().item()),
        "q_action_gap_std": float(gaps.std(unbiased=False).item()),
        "q_max_mean": float(best.mean().item()),
    }


# --------------------------------------------------------------------------------------
# GSP prediction diversity (Shannon entropy of binned predictions)
# --------------------------------------------------------------------------------------

def compute_gsp_pred_diversity(
    predictions: np.ndarray,
    n_bins: int = 10,
    low: float = -1.0,
    high: float = 1.0,
) -> float:
    """Shannon entropy (natural log) of per-timestep GSP predictions binned into
    ``n_bins`` equal-width bins over ``[low, high]``.

    A constant prediction → entropy = 0; uniformly varied predictions → log(n_bins).
    Gives a scalar per-episode summary that distinguishes ``out_std=0.07 but all
    the same value`` (collapse) from ``out_std=0.07 but actually varying``.
    """
    preds = np.asarray(predictions, dtype=np.float64).ravel()
    if preds.size == 0:
        return 0.0
    # Bin into [low, high]; out-of-range values get clipped to the edges.
    preds_clipped = np.clip(preds, low, high)
    counts, _ = np.histogram(preds_clipped, bins=n_bins, range=(low, high))
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    # Shannon entropy with natural log; 0 * log(0) = 0 by convention.
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(probs > 0, np.log(probs), 0.0)
    entropy = -float((probs * logs).sum())
    return entropy
