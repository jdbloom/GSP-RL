"""DDPG Actor and Critic networks for continuous action spaces.

Provides DDPGActorNetwork (deterministic policy) and DDPGCriticNetwork
(state-action value function). Both use fan-in weight initialization.
Also serves as the base architecture composed by RDDPG wrappers in rddpg.py.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.optim as optim
from gsp_rl.src.networks import get_device
import numpy as np


def fanin_init(size, fanin=None):
    """Initialize weights using fan-in uniform distribution.

    Args:
        size: Shape tuple for the weight tensor.
        fanin: Fan-in value. Defaults to size[0] (input features).

    Returns:
        Tensor of given size with values uniform in [-1/sqrt(fanin), 1/sqrt(fanin)].
    """
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return T.Tensor(size).uniform_(-v, v)


class DDPGActorNetwork(nn.Module):
    """Deterministic policy network for DDPG.

    Architecture: Linear(input_size, 400) -> ReLU -> Linear(400, 300) -> ReLU
    -> Linear(300, output_size) -> Tanh * min_max_action.

    Output is bounded to [-min_max_action, min_max_action] via tanh scaling.
    Used directly for DDPG and as a component inside RDDPGActorNetwork.

    Attributes:
        min_max_action: Action space bound for tanh scaling.
        name: Formatted as '{name}_{id}_DDPG' for checkpoint files.
        DIAGNOSTIC_PROFILE: Declarative profile consumed by Actor._diagnose_network.
    """

    DIAGNOSTIC_PROFILE = {
        'fau_layers':      ['fc1', 'fc2'],
        'wnorm_layers':    ['fc1', 'fc2', 'mu'],
        'has_penultimate': True,
        'output_kind':     'continuous_action',
    }
    def __init__(
            self,
            id: int,
            lr: float,
            input_size: int,
            output_size: int,
            fc1_dims: int = 400,
            fc2_dims: int = 300,
            name: str = "DDPG_Actor",
            min_max_action: float = 1.0,
            weight_decay: float = 1e-4,
            init_w: float = 3e-3,
            use_layer_norm: bool = False,
            use_linear_output: bool = False,
            init_scheme: str = "fanin",
    ) -> None:
        """Initialize DDPG actor network.

        Args:
            id: Agent identifier, embedded in checkpoint name.
            lr: Learning rate for Adam optimizer.
            input_size: Observation dimensionality (or encoding size for RDDPG).
            output_size: Action space dimensionality.
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Base name for checkpoint files.
            min_max_action: Tanh output scaling factor (or clamp bound when
                use_linear_output=True).
            weight_decay: Adam weight decay. Default 1e-4 preserves legacy behavior;
                set to 0 in the GSP-head ablation to test the decoupled-decay hypothesis.
            init_w: Output layer weight init half-range. Default 3e-3 preserves legacy;
                override (e.g. 0.1) in the GSP-head ablation to escape the pull-to-mean mode.
            use_layer_norm: If True, insert LayerNorm after fc1 and fc2 (before each
                ReLU). Defaults to False (legacy). Enabled as Task 4 of the stability
                plan — LayerNorm in the trunk is the highest-evidence plasticity
                stabilizer in recent off-policy RL (Lyle et al. NeurIPS 2024, BRO 2024).
                Placement is the trunk (not pre-tanh) because the failure mode identified
                in the ddpg-vs-attention analysis is a dead-ReLU cascade, not an
                output-layer issue.
            use_linear_output: If True, replace tanh with a hard clamp to
                [-min_max_action, min_max_action]. Defaults to False (legacy tanh).
                Motivated by MARL communication literature (DIAL, CommNet, TarMAC)
                where unbounded/linearly-bounded messages during training avoid the
                gradient attenuation tanh introduces near ±1 and near 0.
            init_scheme: Hidden-layer weight initialization scheme. One of:
                - ``"fanin"`` (default): uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)].
                  Preserves legacy behavior for all existing runs.
                - ``"kaiming"``: Kaiming He normal init (std=sqrt(2/fan_in)) for fc1
                  and fc2. Produces ~2.4x larger weights than fanin for typical GSP
                  input sizes, which should reduce initial dead-ReLU fraction when the
                  input distribution is positive-bounded (e.g., prox values in [0, 0.5]).
                  The output layer (mu) always uses small uniform init_w regardless of
                  this setting — Kaiming does not apply to output projections.
        """
        super().__init__()

        self.device = get_device()

        self.min_max_action = min_max_action
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.use_layer_norm = use_layer_norm
        self.use_linear_output = use_linear_output
        self.init_scheme = init_scheme

        self.fc1 = nn.Linear(input_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, output_size)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(self.fc1_dims)
            self.ln2 = nn.LayerNorm(self.fc2_dims)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        self.init_weights(init_w)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = weight_decay)

        self.name = name+'_'+str(id)+'_DDPG'

        self.to(self.device)


    def init_weights(self, init_w: float) -> None:
        """Initializes weights of the network.

        Hidden layers (fc1, fc2) are initialized according to ``self.init_scheme``:
        - ``"fanin"``: uniform in [-1/sqrt(fan_in), 1/sqrt(fan_in)] (legacy default).
        - ``"kaiming"``: Kaiming He normal (mode='fan_in', nonlinearity='relu').

        The output layer (mu) always uses small uniform [-init_w, init_w] regardless
        of init_scheme — Kaiming does not apply to output projections.
        """
        if self.init_scheme == "kaiming":
            T.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
            T.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        elif self.init_scheme == "fanin_fixed":
            # Correct fan-in: index size[1] = in_features (PyTorch Linear (out, in) convention).
            # Legacy "fanin" scheme uses size[0] which is the OUT dim — that's the bug we're fixing.
            self.fc1.weight.data = fanin_init(
                self.fc1.weight.data.size(), fanin=self.fc1.weight.data.size()[1]
            )
            self.fc2.weight.data = fanin_init(
                self.fc2.weight.data.size(), fanin=self.fc2.weight.data.size()[1]
            )
        else:
            # Default: fanin uniform (legacy behavior — preserves all existing runs).
            # NOTE: misnamed — actually uses size[0] = OUT_features as fan-in.
            self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
            self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.mu.weight.data.uniform_(-init_w, init_w)

    def forward(self, x: T.Tensor, return_features: bool = False):
        """Compute deterministic action given state.

        Args:
            x: State tensor of shape (*, input_size).
            return_features: If True, returns (action, penultimate_features) tuple
                where penultimate_features is the post-ReLU activation of fc2 —
                the feature vector immediately before the output linear.
                This is the tensor targeted by VICReg variance+covariance
                regularization for Task 5 of the stability plan. Default False
                preserves the legacy Tensor-only return.

        Returns:
            Action tensor of shape (*, output_size), bounded by min_max_action.
            If return_features: (action, features) where features has shape
            (*, fc2_dims).
        """
        prob = self.fc1(x)
        if self.use_layer_norm:
            prob = self.ln1(prob)
        prob = self.relu(prob)
        prob = self.fc2(prob)
        if self.use_layer_norm:
            prob = self.ln2(prob)
        prob = self.relu(prob)
        penultimate = prob  # post-ReLU activation at fc2 is the "features" VICReg regularizes
        mu = self.mu(prob)
        if self.use_linear_output:
            mu = T.clamp(mu, -self.min_max_action, self.min_max_action)
        else:
            mu = self.min_max_action * self.tanh(mu)
        if return_features:
            return mu, penultimate
        return mu

    def penultimate(self, x: T.Tensor) -> T.Tensor:
        """Return post-ReLU activations of fc2 — the feature vector immediately
        before the output linear (mu). Alias to the existing return_features=True
        forward path but exposed under the standard diagnostics method name so
        compute_effective_rank can find it consistently across network classes.
        """
        prob = self.fc1(x)
        if self.use_layer_norm:
            prob = self.ln1(prob)
        prob = self.relu(prob)
        prob = self.fc2(prob)
        if self.use_layer_norm:
            prob = self.ln2(prob)
        return self.relu(prob)

    def save_checkpoint(self, path: str, intention=False) -> None:
        """ Saves the Model"""
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention=False) -> None:
        """ Saves the Model"""
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        self.load_state_dict(T.load(path + '_' + network_name))


############################################################################
# Critic Network for DDPG
############################################################################
class DDPGCriticNetwork(nn.Module):
    """State-action value function (Q-network) for DDPG.

    Architecture: Linear(input_size, 400) -> ReLU -> Linear(400, 300) -> ReLU
    -> Linear(300, 1). Concatenates state and action internally in forward().

    IMPORTANT: input_size must equal state_dim + action_dim because forward()
    calls T.cat([state, action], dim=-1) before the first layer.

    Attributes:
        name: Formatted as '{name}_{id}_DDPG' for checkpoint files.
        DIAGNOSTIC_PROFILE: Declarative profile consumed by Actor._diagnose_network.
    """

    DIAGNOSTIC_PROFILE = {
        'fau_layers':      ['fc1', 'fc2'],
        'wnorm_layers':    ['fc1', 'fc2', 'q'],
        'has_penultimate': False,
        'output_kind':     'q_scalar',
    }
    def __init__(self,
                 id: int,
                 lr: float,
                 input_size: int,
                 output_size: int,
                 fc1_dims: int = 400,
                 fc2_dims: int = 300,
                 name: str = "DDPG_Critic"
    ):
        """Initialize DDPG critic network.

        Args:
            id: Agent identifier, embedded in checkpoint name.
            lr: Learning rate for Adam optimizer.
            input_size: Must be state_dim + action_dim (state-action concat).
            output_size: Kept for interface consistency (output is always 1).
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Base name for checkpoint files.
        """
        super().__init__()

        self.device = get_device()
        self.lr = lr
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc1 = nn.Linear(input_size, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.relu = nn.ReLU()
        self.init_weights(3e-3)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.name = name+'_'+str(id)+'_DDPG'
        self.to(self.device)

    def init_weights(self, init_w: float) -> None:
        """
        Initializes the weights of the network
        """
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.q.weight.data.uniform_(-init_w, init_w)

    def forward(self, state: T.Tensor, action: T.Tensor) -> T.Tensor:
        """Compute Q-value for a state-action pair.

        Args:
            state: State tensor of shape (batch, state_dim).
            action: Action tensor of shape (batch, action_dim).

        Returns:
            Q-value tensor of shape (batch, 1).
        """
        action_value = self.fc1(T.cat([state, action], dim = -1))
        action_value = self.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = self.relu(action_value)
        action_value = self.q(action_value)
        return action_value

    def save_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Saves Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False) -> None:
        """ Loads Model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))