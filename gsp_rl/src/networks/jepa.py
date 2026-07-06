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


class JEPAEncoder(nn.Module):
    """Two-layer MLP encoder: state → latent.

    Architecture:
        fc1 (input_dim → hidden) → LayerNorm → ReLU → fc2 (hidden → latent_dim)

    The final projection is a raw linear layer — no activation, no LayerNorm.
    This preserves gradient flow into the latent space and lets VICReg
    variance/covariance regularization operate on unbounded representations.

    Args:
        input_dim:  Dimensionality of the GSP head input (gsp_network_input).
        latent_dim: Output latent dimensionality. Default 32 (GSP_ENCODER_DIM).
        hidden:     Hidden layer width. Default 128.
    """

    def __init__(self, input_dim: int, latent_dim: int = 32, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, latent_dim)

        # Determine device from environment (mirrors existing network convention)
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, input_dim).

        Returns:
            Latent tensor of shape (batch, latent_dim).
        """
        h = self.relu(self.ln1(self.fc1(x)))
        return self.fc2(h)


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
