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
    """One-layer MLP predictor: latent_t → predicted latent_{t+k}.

    Maps the online encoder's latent representation z_t to a prediction of
    the target encoder's latent representation z_{t+k}. The MSE loss between
    predictor output and the (detached) target latent drives supervised
    learning in the latent space.

    Args:
        latent_dim: Both input and output dimensionality (latent space).
        hidden:     Hidden layer width. Default 64.
    """

    def __init__(self, latent_dim: int = 32, hidden: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, latent_dim)

        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, z: T.Tensor) -> T.Tensor:
        """Forward pass.

        Args:
            z: Latent tensor of shape (batch, latent_dim).

        Returns:
            Predicted future latent of shape (batch, latent_dim).
        """
        h = self.relu(self.fc1(z))
        return self.fc2(h)
