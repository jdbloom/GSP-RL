"""JEPA (Joint Embedding Predictive Architecture) modules for the GSP head.

Provides two nn.Module classes used by the JEPA training path in actor.py:

- JEPAEncoder: maps raw GSP head input (state_t or state_t+k) to a latent
  vector of shape (batch, latent_dim). Two-layer MLP, raw linear output (no
  tanh), with LayerNorm on the hidden representation.

- JEPAPredictor: maps online encoder latent z_t → predicted future latent
  z_{t+k}. One-layer MLP. Output is in the same latent space as the target
  encoder (used for MSE prediction loss).

Design notes:
- No tanh on the encoder output — the latent lives in an unbounded linear
  space so that VICReg variance/covariance losses operate without saturation.
- LayerNorm on the hidden (encoder fc1 output) stabilizes training but is
  intentionally omitted from the final linear projection (matching the
  VICReg expander design in Bardes et al. ICLR 2022).
- The target encoder is maintained as an EMA copy of the online encoder in
  actor.py; this file only defines the shared architecture.

See: docs/superpowers/specs/2026-04-16-jepa-mvp.md for the design rationale.
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F


def simnorm(x: T.Tensor, group_size: int = 8) -> T.Tensor:
    """Simplicial normalization (DreamerV3 / TD-MPC2).

    Reshapes the trailing latent dimension into contiguous groups of
    ``group_size`` and applies a softmax within each group, projecting every
    group onto the probability simplex (each group sums to 1, all entries in
    [0, 1]). This bounds and regularizes an otherwise unbounded latent so it can
    be fed as a stable input to the actor's Q-net (and compared consistently in
    the JEPA self-prediction loss).

    Reference: Hafner et al. DreamerV3 (2301.04104) — "simlog"/simplical latent;
    Hansen et al. TD-MPC2 (2310.16828) §3 SimNorm.

    Args:
        x: Tensor of shape ``(..., latent_dim)``. ``latent_dim`` MUST be an exact
            multiple of ``group_size``.
        group_size: Number of elements per simplex group. Default 8 (TD-MPC2).

    Returns:
        Tensor of the same shape as ``x``, with each contiguous group of
        ``group_size`` entries lying on the simplex (summing to 1).
    """
    if group_size <= 0:
        raise ValueError(f"simnorm group_size must be positive, got {group_size}")
    latent_dim = x.shape[-1]
    if latent_dim % group_size != 0:
        raise ValueError(
            f"simnorm: latent_dim ({latent_dim}) must be divisible by "
            f"group_size ({group_size})."
        )
    lead_shape = x.shape[:-1]
    n_groups = latent_dim // group_size
    x = x.view(*lead_shape, n_groups, group_size)
    x = F.softmax(x, dim=-1)
    return x.reshape(*lead_shape, latent_dim)


class JEPAEncoder(nn.Module):
    """Two-layer MLP encoder: state → latent.

    Architecture:
        fc1 (input_dim → hidden) → LayerNorm → ReLU → fc2 (hidden → latent_dim)

    The final projection is a raw linear layer — no activation, no LayerNorm.
    This preserves gradient flow into the latent space and lets VICReg
    variance/covariance regularization operate on unbounded representations.

    SimNorm (flag ``GSP_JEPA_SIMNORM``, DreamerV3 / TD-MPC2): when ``simnorm=True``
    the final latent is projected onto a per-group simplex before it leaves the
    encoder. Because the target encoder is an EMA *copy of this class* and the
    host env's ``choose_agent_gsp`` calls this same ``forward``, applying SimNorm
    here keeps the online latent, the EMA-target latent, and the runtime
    augmented-obs latent byte-consistent with a single code path. ``simnorm=False``
    (default) is bit-identical to the legacy encoder.

    Args:
        input_dim:  Dimensionality of the GSP head input (gsp_network_input).
        latent_dim: Output latent dimensionality. Default 32 (GSP_ENCODER_DIM).
        hidden:     Hidden layer width. Default 128.
        simnorm:    Apply SimNorm to the latent. Default False (legacy).
        simnorm_group_size: Group size for SimNorm. Default 8.
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int = 32,
        hidden: int = 128,
        simnorm: bool = False,
        simnorm_group_size: int = 8,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, latent_dim)
        self.simnorm = bool(simnorm)
        self.simnorm_group_size = int(simnorm_group_size)
        if self.simnorm and latent_dim % self.simnorm_group_size != 0:
            raise ValueError(
                f"GSP_JEPA_SIMNORM: GSP_ENCODER_DIM ({latent_dim}) must be "
                f"divisible by simnorm group_size ({self.simnorm_group_size})."
            )

        # Determine device from environment (mirrors existing network convention)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Latent tensor of shape (batch, latent_dim). When ``simnorm`` is set,
            each contiguous group of ``simnorm_group_size`` entries lies on the
            simplex.
        """
        h = self.relu(self.ln1(self.fc1(x)))
        z = self.fc2(h)
        if self.simnorm:
            z = simnorm(z, self.simnorm_group_size)
        return z


class JEPAPredictor(nn.Module):
    """One-layer MLP predictor: (latent_t[, action_t]) → predicted latent_{t+k}.

    Maps the online encoder's latent representation z_t to a prediction of
    the target encoder's latent representation z_{t+k}. The latent loss between
    predictor output and the (detached) target latent drives supervised
    learning in the latent space.

    Action-conditioning (flag GSP_JEPA_ACTION_COND, SPR 2007.05929):
        When ``action_dim > 0`` the predictor is action-conditioned:
        ``fc1`` takes ``latent_dim + action_dim`` inputs and ``forward`` REQUIRES
        an action tensor ``a`` of shape (batch, action_dim), which is concatenated
        to z before the first layer. This lets the predictor distinguish
        action-caused futures — the SPR fix for coordination targets that look
        "unlearnable" when the predictor cannot see which action produced them.

        When ``action_dim == 0`` (default) the predictor is byte-identical to the
        legacy z-only architecture: same layer shapes, same parameter order,
        ``forward(z)`` takes a single argument. Existing checkpoints load
        unchanged.

    Args:
        latent_dim: Both input and output dimensionality (latent space).
        hidden:     Hidden layer width. Default 64.
        action_dim: Action vector width to condition on. 0 (default) = legacy
            z-only predictor (back-compat preserved).
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 64, action_dim: int = 0):
        super().__init__()
        self.action_dim = int(action_dim)
        # Input width grows by action_dim only when action-conditioned; at
        # action_dim=0 this reduces to the legacy (latent_dim -> hidden) layer,
        # keeping the parameter shapes byte-identical to pre-flag checkpoints.
        self.fc1 = nn.Linear(latent_dim + self.action_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, latent_dim)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, z: T.Tensor, a: T.Tensor | None = None) -> T.Tensor:
        """Forward pass.

        Args:
            z: Latent tensor of shape (batch, latent_dim).
            a: Action tensor of shape (batch, action_dim). REQUIRED when the
                predictor was built with action_dim > 0; must be None/omitted
                otherwise.

        Returns:
            Predicted future latent of shape (batch, latent_dim).
        """
        if self.action_dim > 0:
            if a is None:
                raise ValueError(
                    "JEPAPredictor built with action_dim > 0 requires an action "
                    "tensor 'a' of shape (batch, action_dim)."
                )
            x = T.cat([z, a], dim=-1)
        else:
            x = z
        h = self.relu(self.fc1(x))
        return self.fc2(h)
