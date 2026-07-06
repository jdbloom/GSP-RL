"""Successor-Features DDQN head (GSP_SF_ENABLED).

Barreto et al. 2017 (arXiv:1606.05312). Instead of a scalar Q per action, the
network predicts the *successor features* psi(s, a) in R^(n_actions x d_phi) — the
expected discounted sum of a low-dim cumulant phi. The action value is a linear
readout:

    Q(s, a) = psi(s, a) . w        w in R^d_phi  (learned reward weights)

This makes the prediction psi *be* the value computation: zeroing psi zeroes Q by
construction, so the causal-ablation gate is definitional (the two prior GSP
attempts — concat coupled-JEPA and latent-primary — failed precisely because the
predicted quantity was decision-irrelevant / could not carry control info). The
raw observation is KEPT as the network input (latent-primary's fatal mistake was
dropping it), so the policy can still learn.

The trunk (fc1/fc2) is architecturally identical to DDQN; only the output layer
changes from (n_actions) to (n_actions * d_phi). w is a small nn.Parameter trained
by regressing Q = psi . w to the scalar reward-to-go / reward (see
learn_DDQN_sf in learning_aids.py).

See the pre-registration: docs/research/2026-07-06-successor-features-escalation-prereg.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gsp_rl.src.networks import get_device


class DDQN_SF(nn.Module):
    """Successor-Features Q-network for Double DQN.

    Same fc1/fc2 trunk as :class:`DDQN`; the output head produces psi of shape
    (*, n_actions, d_phi). A learned reward-weight vector ``w`` (nn.Parameter of
    shape (d_phi,)) maps psi to Q via a dot product over the last axis.

    Attributes:
        n_actions: number of discrete actions.
        d_phi: dimensionality of the cumulant phi (successor-feature width).
        fc1, fc2: hidden trunk (identical to DDQN).
        psi_head: output layer (fc2_dims -> n_actions * d_phi).
        w: learned reward-weight nn.Parameter of shape (d_phi,).
        psi_optimizer: Adam over the trunk + psi_head (the successor-feature TD path).
        w_optimizer: separate Adam over ``w`` only (the reward-regression path),
            so the reward regression cannot perturb the psi trunk and vice-versa.
        loss: SmoothL1 (huber) or MSE, matching the DDQN critic_loss flag. Used for
            the psi TD loss (applied per-component).
    """

    DIAGNOSTIC_PROFILE = {
        'fau_layers':      ['fc1', 'fc2'],
        'wnorm_layers':    ['fc1', 'fc2', 'psi_head'],
        'grad_layers':     ['fc1', 'fc2', 'psi_head'],
        'has_penultimate': True,
        'output_kind':     'q_values',
    }

    def __init__(
            self,
            id: int,
            lr: float,
            input_size: int,
            output_size: int,
            d_phi: int,
            fc1_dims: int = 64,
            fc2_dims: int = 128,
            name: str = 'DDQN',
            use_layer_norm: bool = False,
            critic_loss: str = 'mse',
    ) -> None:
        """Initialize the SF-DDQN network.

        Args:
            id: Agent identifier.
            lr: Learning rate for both Adam optimizers.
            input_size: Observation space dimensionality (raw obs is KEPT).
            output_size: Number of discrete actions (n_actions).
            d_phi: Cumulant dimensionality (successor-feature width).
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Network name for checkpoint file naming.
            use_layer_norm: If True, LayerNorm after fc1 and fc2 (matches DDQN).
            critic_loss: 'mse' (default) or 'huber' — controls the psi TD loss fn.
        """
        super().__init__()

        self.name = name
        self.use_layer_norm = use_layer_norm
        self.n_actions = int(output_size)
        self.d_phi = int(d_phi)

        # Eval-time causal-ablation handle. When set to 'zero', psi() returns a
        # zeroed successor-feature tensor, so Q = psi . w == 0 identically and the
        # policy degenerates to argmax over ties — the definitional causal test
        # from the pre-reg (zeroing psi zeroes Q). 'freeze_mean' replaces psi with
        # its per-batch mean (removes the temporal/state-dependent signal while
        # keeping the mean value scale). 'none' (default) is an exact no-op.
        self.psi_ablation = 'none'

        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        # Output the flattened psi tensor: (n_actions * d_phi).
        self.psi_head = nn.Linear(fc2_dims, self.n_actions * self.d_phi)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(fc1_dims)
            self.ln2 = nn.LayerNorm(fc2_dims)

        # Learned reward-weight vector w in R^d_phi. Initialised to 1/d_phi so the
        # initial Q = psi . w is the mean of psi's components — a bounded, neutral
        # starting readout that keeps the value scale small before w is trained.
        self.w = nn.Parameter(T.full((self.d_phi,), 1.0 / self.d_phi))

        # Two optimizers: psi (trunk + head) trained by its own TD loss; w trained
        # by reward regression. Separating them keeps the value-scale-setting w
        # update from injecting gradient noise into the psi feature learning.
        trunk_params = list(self.fc1.parameters()) + list(self.fc2.parameters()) \
            + list(self.psi_head.parameters())
        if self.use_layer_norm:
            trunk_params += list(self.ln1.parameters()) + list(self.ln2.parameters())
        self.psi_optimizer = optim.Adam(trunk_params, lr=float(lr), weight_decay=1e-4)
        self.w_optimizer = optim.Adam([self.w], lr=float(lr))

        # Kept for API compatibility with DDQN callers that reference `.optimizer`
        # (e.g. diagnostics, checkpoint code). Points at the psi optimizer — the
        # primary value-learning path.
        self.optimizer = self.psi_optimizer

        self.loss = nn.SmoothL1Loss() if str(critic_loss).lower() == 'huber' else nn.MSELoss()

        self.device = get_device()
        self.to(self.device)

    def psi(self, state: T.Tensor) -> T.Tensor:
        """Compute successor features psi(s, .) for all actions.

        Args:
            state: Observation tensor of shape (*, input_size).

        Returns:
            psi tensor of shape (*, n_actions, d_phi).
        """
        x = self.fc1(state)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x = F.relu(x)
        flat = self.psi_head(x)  # (*, n_actions * d_phi)
        psi = flat.view(*flat.shape[:-1], self.n_actions, self.d_phi)
        if self.psi_ablation == 'zero':
            # Causal test: zero psi -> Q = psi . w == 0 by construction.
            psi = T.zeros_like(psi)
        elif self.psi_ablation == 'freeze_mean' and psi.dim() >= 2:
            # Remove the state/temporal signal: replace psi with the batch mean,
            # broadcast back over the batch. Keeps the mean value scale.
            psi = psi.mean(dim=0, keepdim=True).expand_as(psi)
        return psi

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Compute Q-values Q(s, a) = psi(s, a) . w for all actions.

        Returns a tensor of shape (*, n_actions), so this is a drop-in replacement
        for DDQN.forward at every action-selection / argmax call site.
        """
        psi = self.psi(state)                 # (*, n_actions, d_phi)
        return T.matmul(psi, self.w)          # (*, n_actions)

    def penultimate(self, state: T.Tensor) -> T.Tensor:
        """Post-ReLU fc2 activations (feature vector before the psi head)."""
        x = self.fc1(state)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        return F.relu(x)

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Saves the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name, '...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Loads the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))
