"""Double Deep Q-Network (DDQN) for discrete action spaces.

Same architecture as DQN. The double-Q trick is implemented in the learn
method (NetworkAids.learn_DDQN), not in the network itself: q_eval selects
actions, q_next evaluates them, reducing overestimation bias.

See Also: docs/modules/networks.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gsp_rl.src.networks import get_device


class DDQN(nn.Module):
    """Feedforward Q-network for Double DQN.

    Architecturally identical to DQN. The double-Q decoupling happens in
    the learn loop, not in the network structure.

    Attributes:
        fc1: First hidden layer (input_size -> fc1_dims).
        fc2: Second hidden layer (fc1_dims -> fc2_dims).
        fc3: Output layer (fc2_dims -> output_size), produces Q-values.
        optimizer: Adam with weight_decay=1e-4.
        loss: MSELoss instance.
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
            name: str = 'DDQN',
            use_layer_norm: bool = False,
    ) -> None:
        """Initialize DDQN network.

        Args:
            id: Agent identifier.
            lr: Learning rate for Adam optimizer.
            input_size: Observation space dimensionality.
            output_size: Number of discrete actions.
            fc1_dims: First hidden layer width.
            fc2_dims: Second hidden layer width.
            name: Network name for checkpoint file naming.
            use_layer_norm: If True, insert LayerNorm after fc1 and fc2 (before each
                ReLU). Defaults to False (legacy). Controlled by ACTOR_USE_LAYER_NORM
                config flag — separate from the GSP head's GSP_USE_LAYER_NORM. The
                trunk placement follows BRO (NeurIPS 2024) and Lyle et al. (NeurIPS
                2024) which show LN in the trunk is the most effective off-policy RL
                plasticity stabilizer.
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
        """Return post-ReLU activations of fc2 — the feature vector immediately
        before the output layer. Used by diagnostics (effective rank).
        """
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
        print('... saving', network_name,'...')
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