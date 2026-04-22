"""Deep Q-Network (DQN) for discrete action spaces.

Provides a simple feedforward Q-network: Linear(input_size, 64) -> ReLU ->
Linear(64, 128) -> ReLU -> Linear(128, output_size). Outputs raw Q-values
(no output activation). Used in pairs (q_eval, q_next) by NetworkAids.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gsp_rl.src.networks import get_device


class DQN(nn.Module):
    """Feedforward Q-network for DQN and value-based RL.

    Produces Q-values for each discrete action given a state observation.
    The network dict stores two instances: 'q_eval' (online) and 'q_next' (target).

    Attributes:
        fc1: First hidden layer (input_size -> fc1_dims).
        fc2: Second hidden layer (fc1_dims -> fc2_dims).
        fc3: Output layer (fc2_dims -> output_size), produces Q-values.
        optimizer: Adam with weight_decay=1e-4.
        loss: MSELoss instance for TD error computation.
        device: Auto-detected cuda:0 or cpu.
        DIAGNOSTIC_PROFILE: Declarative profile consumed by Actor._diagnose_network.
    """

    DIAGNOSTIC_PROFILE = {
        'fau_layers':      ['fc1', 'fc2'],
        'wnorm_layers':    ['fc1', 'fc2', 'fc3'],
        'grad_layers':     ['fc1', 'fc2', 'fc3'],
        'has_penultimate': True,
        'output_kind':     'q_values',
    }
    def __init__(
            self,
            id: int,
            lr: float,
            input_size: int,
            output_size: int,
            fc1_dims: int = 64,
            fc2_dims: int = 128,
            name: str = 'DQN',
            use_layer_norm: bool = False,
    ) -> None:
        """Initialize DQN network.

        Args:
            id: Agent identifier (unused in network, kept for interface consistency).
            lr: Learning rate for Adam optimizer.
            input_size: Observation space dimensionality.
            output_size: Number of discrete actions (Q-value per action).
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Network name for checkpoint file naming.
            use_layer_norm: If True, insert LayerNorm after fc1 and fc2 (before each
                ReLU). Defaults to False (legacy). See DDQN docstring for rationale.
        """
        super().__init__()

        self.name = name
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_size)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(fc1_dims)
            self.ln2 = nn.LayerNorm(fc2_dims)

        self.optimizer = optim.Adam(self.parameters(), lr = lr, weight_decay = 1e-4)

        self.loss = nn.MSELoss()

        self.device = get_device()

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Compute Q-values for all actions given a state.

        Args:
            state: Observation tensor of shape (*, input_size).

        Returns:
            Q-values tensor of shape (*, output_size).
        """
        x = self.fc1(state)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        x1 = F.relu(x)
        actions = self.fc3(x1)

        return actions

    def penultimate(self, state: T.Tensor) -> T.Tensor:
        """Return post-ReLU activations of fc2. See DDQN.penultimate for rationale."""
        x = self.fc1(state)
        if self.use_layer_norm:
            x = self.ln1(x)
        x = F.relu(x)
        x = self.fc2(x)
        if self.use_layer_norm:
            x = self.ln2(x)
        return F.relu(x)

    def save_checkpoint(self, path: str, intention: bool = False):
        """ Saves the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... saving', network_name,'...')
        T.save(self.state_dict(), path + '_' + network_name)

    def load_checkpoint(self, path: str, intention: bool = False):
        """ Loads the model """
        network_name = self.name
        if intention:
            network_name += "_intention"
        print('... loading', network_name, '...')
        if str(self.device) == 'cpu':
            self.load_state_dict(T.load(path + '_' + network_name, map_location=T.device('cpu')))
        else:
            self.load_state_dict(T.load(path + '_' + network_name))