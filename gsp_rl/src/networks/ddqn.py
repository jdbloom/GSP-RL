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
            critic_loss: str = 'mse',
            advantage_only_pred: tuple | None = None,
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
            critic_loss: 'mse' (default, legacy) or 'huber'. Huber (SmoothL1,
                beta=1.0) caps the per-sample gradient magnitude for |TD error|>1,
                the most effective divergence control under Adam where a global
                gradient rescale is normalized away. Controlled by the CRITIC_LOSS
                config flag.
            advantage_only_pred: None (default, legacy) or (start, width) — the
                column span of the spliced GSP prediction inside the input.
                When set, the net becomes a DUELING head with the split placed so
                the prediction reaches ONLY the advantage stream
                (GSP_SPLICE_ADVANTAGE_ONLY):
                  * A(s,a): the existing fc1→fc2→fc3 trunk over the FULL input
                    (obs + pred) — fc3's output is reinterpreted as advantages;
                  * V(s):   a parallel v_fc1→v_fc2→v_fc3 stream over the input
                    with the pred columns REMOVED, so the prediction cannot
                    influence the state-value by construction;
                  * Q = V + A - mean(A).
                Motivation (2026-07-09 Q-probe): with the flat head, the spliced
                prediction's effect on Q is ~99.8% common-mode (a state-value
                offset) — the gradient-cheapest absorption of an action-invariant
                input. The dueling split architecturally forbids common-mode
                absorption: the feature can only express itself as action
                preference. None = byte-identical legacy flat Q-head (no extra
                modules, no extra RNG draws, same op sequence).
        """
        super().__init__()

        self.name = name
        self.use_layer_norm = use_layer_norm
        self.advantage_only_pred = (
            None if advantage_only_pred is None
            else (int(advantage_only_pred[0]), int(advantage_only_pred[1]))
        )

        self.fc1 = nn.Linear(input_size, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, output_size)

        if self.use_layer_norm:
            self.ln1 = nn.LayerNorm(fc1_dims)
            self.ln2 = nn.LayerNorm(fc2_dims)

        if self.advantage_only_pred is not None:
            # Dueling value stream. Built AFTER the trunk so the trunk's RNG
            # draws are unchanged vs legacy; when advantage_only_pred is None
            # this whole block is skipped and construction is byte-identical.
            start, width = self.advantage_only_pred
            if width < 1 or start < 0 or start + width > input_size:
                raise ValueError(
                    f'{name}: advantage_only_pred span ({start}, {width}) does '
                    f'not fit inside input_size {input_size}'
                )
            v_input_size = input_size - width
            if v_input_size < 1:
                raise ValueError(
                    f'{name}: advantage_only_pred removes ALL input columns '
                    f'(input_size {input_size}, pred width {width}) — the value '
                    'stream would have no input'
                )
            self.v_fc1 = nn.Linear(v_input_size, fc1_dims)
            self.v_fc2 = nn.Linear(fc1_dims, fc2_dims)
            self.v_fc3 = nn.Linear(fc2_dims, 1)
            if self.use_layer_norm:
                self.v_ln1 = nn.LayerNorm(fc1_dims)
                self.v_ln2 = nn.LayerNorm(fc2_dims)
            # Instance-level profile override so the per-episode diagnostics
            # (weight norms / grad norms) cover the value stream from day one.
            self.DIAGNOSTIC_PROFILE = {
                **self.DIAGNOSTIC_PROFILE,
                'wnorm_layers': ['fc1', 'fc2', 'fc3', 'v_fc1', 'v_fc2', 'v_fc3'],
                'grad_layers': ['fc1', 'fc2', 'fc3', 'v_fc1', 'v_fc2', 'v_fc3'],
            }

        # float(lr): a YAML-sourced sci-notation LR (e.g. '3e-05') can arrive as
        # a str and crash Adam's `0.0 <= lr` check. Defensive at the call site.
        self.optimizer = optim.Adam(self.parameters(), lr = float(lr), weight_decay = 1e-4)

        self.loss = nn.SmoothL1Loss() if str(critic_loss).lower() == 'huber' else nn.MSELoss()

        self.device = get_device()

        self.to(self.device)

    def forward(self, state: T.Tensor) -> T.Tensor:
        """Compute Q-values for all actions given a state.

        Legacy (advantage_only_pred is None): the fc trunk's output IS Q.
        Dueling (GSP_SPLICE_ADVANTAGE_ONLY): the trunk output is reinterpreted
        as A(s,a); V(s) comes from the pred-excluded value stream; and
        Q = V + A - mean(A). mean(A) subtraction makes the decomposition
        identifiable (Wang et al. 2016) — V(s) == mean_a Q(s,a), so the spliced
        prediction provably cannot move the common-mode component of Q.

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
        if self.advantage_only_pred is None:
            return actions
        advantage = actions
        value = self.value_stream(state)
        return value + advantage - advantage.mean(dim=-1, keepdim=True)

    def value_stream(self, state: T.Tensor) -> T.Tensor:
        """Dueling V(s): forward the value stream on the pred-EXCLUDED input.

        Only valid when advantage_only_pred is set. The pred columns
        [start, start+width) are removed from the input before v_fc1, so no
        gradient path (and no functional path) exists from the spliced GSP
        prediction into V — the architectural guarantee under test.

        Args:
            state: Observation tensor of shape (*, input_size) — the FULL
                augmented input; the slicing happens here.

        Returns:
            State-value tensor of shape (*, 1).
        """
        if self.advantage_only_pred is None:
            raise RuntimeError(
                f'{self.name}: value_stream called on a non-dueling net '
                '(advantage_only_pred is None)'
            )
        start, width = self.advantage_only_pred
        v_in = T.cat([state[..., :start], state[..., start + width:]], dim=-1)
        v = self.v_fc1(v_in)
        if self.use_layer_norm:
            v = self.v_ln1(v)
        v = F.relu(v)
        v = self.v_fc2(v)
        if self.use_layer_norm:
            v = self.v_ln2(v)
        v = F.relu(v)
        return self.v_fc3(v)

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