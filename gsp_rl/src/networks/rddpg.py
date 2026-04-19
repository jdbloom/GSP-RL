"""Recurrent DDPG (RDDPG) network wrappers.

Composes EnvironmentEncoder (LSTM) with standard DDPG actor/critic networks.
The encoder transforms observation sequences into fixed-size encodings that
replace raw state as input to the DDPG networks.

Architecture:
    State -> EnvironmentEncoder -> encoding -> DDPGActorNetwork -> Action
    State -> EnvironmentEncoder -> encoding -> DDPGCriticNetwork(encoding, action) -> Q-value

In make_RDDPG_networks: actor and critic share one EnvironmentEncoder
instance (shared_ee), while target networks get separate encoder instances
for proper gradient flow isolation.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RDDPGActorNetwork(nn.Module):
    """RDDPG actor: EnvironmentEncoder + DDPGActorNetwork composition.

    Owns an Adam optimizer over all parameters (encoder + DDPG actor).
    Device is inherited from the encoder.

    Attributes:
        ee: EnvironmentEncoder (LSTM-based).
        actor: DDPGActorNetwork.
        DIAGNOSTIC_PROFILE: Declarative profile consumed by Actor._diagnose_network.
            fau_layers and wnorm_layers refer to the inner DDPGActorNetwork layers.
            LSTM cell weights are excluded from FAU (not ReLU units) — tracked by
            separate hidden-norm diagnostic via EnvironmentEncoder in a future followup.
    """

    DIAGNOSTIC_PROFILE = {
        'fau_layers':      ['actor.fc1', 'actor.fc2'],
        'wnorm_layers':    ['actor.fc1', 'actor.fc2', 'actor.mu'],
        'has_penultimate': True,
        'output_kind':     'continuous_action',
    }
    def __init__(self, environmental_encoder, ddpg_actor):
        super().__init__()
        self.ee = environmental_encoder
        self.actor = ddpg_actor
        # Use encoder's device — ensures all components on same device
        self.device = self.ee.device
        self.actor.to(self.device)
        self.actor.device = self.device
        self.optimizer = optim.Adam(self.parameters(), lr = ddpg_actor.lr, weight_decay = 1e-4)

    def forward(self, x, hidden=None):
        """Encode observation through LSTM, then compute action via DDPG actor.

        Args:
            x: Observation tensor of shape (seq_len, input_size) or
               (batch, seq_len, input_size).
            hidden: Optional (h_0, c_0) tuple for the LSTM encoder.

        Returns:
            Tuple of (mu, (h_n, c_n)):
                mu: Action tensor.
                (h_n, c_n): Final LSTM hidden state.
        """
        encoding, hidden_out = self.ee(x, hidden=hidden)
        mu = self.actor(encoding)
        return mu, hidden_out

    def penultimate(self, x):
        """Return post-ReLU activations of the inner DDPGActorNetwork's fc2.

        For diagnostics: each row in x is treated as a single-step sequence
        (seq_len=1) so the LSTM produces one encoding per sample. The DDPG
        actor's penultimate layer is then applied to those encodings.

        Args:
            x: State tensor of shape (N, input_size). Each row is wrapped as a
               sequence of length 1 before passing through the LSTM encoder.

        Returns:
            Tensor of shape (N, fc2_dims) — penultimate features of inner actor.
        """
        # Unsqueeze to (N, 1, input_size) — batch of single-step sequences.
        if x.dim() == 2:
            x_seq = x.unsqueeze(1)
        else:
            x_seq = x
        encoding, _ = self.ee(x_seq, hidden=None)
        # encoding shape: (N, 1, output_size) — take last timestep
        if encoding.dim() == 3:
            encoding = encoding[:, -1, :]  # (N, output_size)
        return self.actor.penultimate(encoding)

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.save_checkpoint(path, intention)
        self.actor.save_checkpoint(path, intention)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.load_checkpoint(path, intention)
        self.actor.load_checkpoint(path, intention)


class RDDPGCriticNetwork(nn.Module):
    """RDDPG critic: EnvironmentEncoder + DDPGCriticNetwork composition.

    Owns an Adam optimizer over all parameters (encoder + DDPG critic).
    Note: save_checkpoint does NOT save the encoder (assumed saved by actor).
    load_checkpoint DOES load the encoder.

    Attributes:
        ee: EnvironmentEncoder (LSTM-based).
        critic: DDPGCriticNetwork.
        DIAGNOSTIC_PROFILE: Declarative profile consumed by Actor._diagnose_network.
            fau_layers and wnorm_layers refer to the inner DDPGCriticNetwork layers.
    """

    DIAGNOSTIC_PROFILE = {
        'fau_layers':      ['critic.fc1', 'critic.fc2'],
        'wnorm_layers':    ['critic.fc1', 'critic.fc2', 'critic.q'],
        'has_penultimate': False,
        'output_kind':     'q_scalar',
    }
    def __init__(self, environmental_encoder, ddpg_critic):
        super().__init__()
        self.ee = environmental_encoder
        self.critic = ddpg_critic
        # Use encoder's device — ensures all components on same device
        self.device = self.ee.device
        self.critic.to(self.device)
        self.critic.device = self.device
        self.optimizer = optim.Adam(self.parameters(), lr = ddpg_critic.lr, weight_decay = 1e-4)
    
    def forward(self, state, action, hidden=None):
        """Encode state through LSTM, then compute Q-value via DDPG critic.

        Args:
            state: Observation tensor of shape (seq_len, input_size) or
                   (batch, seq_len, input_size).
            action: Action tensor of shape (seq_len, action_dim).
            hidden: Optional (h_0, c_0) tuple for the LSTM encoder.

        Returns:
            Tuple of (action_value, (h_n, c_n)):
                action_value: Q-value tensor.
                (h_n, c_n): Final LSTM hidden state.
        """
        encoding, hidden_out = self.ee(state, hidden=hidden)
        action_value = self.critic(encoding, action)
        return action_value, hidden_out
    
    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        # NOTE The ee will be saved in the actor network
        # self.ee.save_checkpoint(path, intention)
        self.critic.save_checkpoint(path, intention)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        path = path+'_recurrent'
        self.ee.load_checkpoint(path, intention)
        self.critic.load_checkpoint(path, intention)

