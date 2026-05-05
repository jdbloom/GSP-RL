"""Network factory methods, learning algorithms, and hyperparameter management.

Contains the three-level class hierarchy base:
- Hyperparameters: Loads and stores all config values from YAML dict.
- NetworkAids(Hyperparameters): Factory methods (make_*_networks), learning
  algorithms (learn_*), action selection (*_choose_action), and memory
  management (sample_memory, store_transition).

Actor (in actor.py) inherits from NetworkAids, completing the chain:
Actor -> NetworkAids -> Hyperparameters.

See Also: docs/modules/actors.md, docs/algorithms.md
"""
from gsp_rl.src.networks import (
    DQN,
    DDQN,
    DDPGActorNetwork,
    DDPGCriticNetwork,
    RDDPGActorNetwork,
    RDDPGCriticNetwork,
    TD3ActorNetwork,
    TD3CriticNetwork,
    EnvironmentEncoder,
    AttentionEncoder
)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam

import numpy as np
import logging

_learn_logger = logging.getLogger("stelaris.learn")


def _check_nan(value, name):
    """Raise RuntimeError if value is NaN or Inf. Works with floats and tensors."""
    if isinstance(value, T.Tensor):
        if T.isnan(value).any() or T.isinf(value).any():
            raise RuntimeError(f"NaN detected in {name}: {value}")
    else:
        if not np.isfinite(value):
            raise RuntimeError(f"NaN detected in {name}: {value}")


Loss = nn.MSELoss()


def vicreg_variance_loss(h: T.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> T.Tensor:
    """VICReg variance term — hinge loss keeping per-dim std >= target_std.

    Part of Task 5 (Bardes, Ponce, LeCun ICLR 2022, arxiv 2105.04906).
    For a feature batch of shape (batch, D), computes std along batch dim for
    each of D features and penalizes those below target_std. Forces the
    representation to have a variance floor, attacking dimensional collapse.

    Args:
        h: Feature tensor of shape (batch, D).
        target_std: Minimum acceptable per-dim std. VICReg paper uses 1.0 but
            for scale-aware application in MSE regression we pass the running
            label std estimate to avoid saturating the tanh output.
        eps: Numerical floor for the std calculation.

    Returns:
        Scalar tensor loss. Zero when all dims already satisfy std >= target_std.
    """
    std = T.sqrt(h.var(dim=0) + eps)
    return T.mean(F.relu(target_std - std))


def vicreg_covariance_loss(h: T.Tensor) -> T.Tensor:
    """VICReg covariance term — off-diagonal penalty on the feature covariance.

    Part of Task 5 (Bardes, Ponce, LeCun ICLR 2022).
    Decorrelates feature dimensions by penalizing off-diagonal elements of the
    batch-wise covariance matrix. Normalized by D to make the coefficient
    approximately scale-independent across feature widths.

    Args:
        h: Feature tensor of shape (batch, D).

    Returns:
        Scalar tensor loss. Zero when features are perfectly decorrelated.
    """
    N, D = h.shape
    h_centered = h - h.mean(dim=0, keepdim=True)
    cov = (h_centered.T @ h_centered) / max(N - 1, 1)
    # Zero out the diagonal, sum of squares on off-diagonal, divide by D
    off_diag = cov - T.diag(T.diagonal(cov))
    return (off_diag ** 2).sum() / D


def gsp_l2er_loss(actor_net, states: T.Tensor, eps: float = 1e-8) -> T.Tensor:
    """Compute the differentiable effective-rank regularization loss for the GSP head.

    Runs a partial forward pass through the actor network's fc1 and fc2 layers,
    capturing the input activation tensor (pre-linear) at each layer. Then
    computes the soft effective rank of each layer's input covariance matrix as:

        erank(M) = exp( H(p) )    where H is Shannon entropy and
        p_i = s_i^2 / sum(s^2)   with s = svdvals(M)

    This is the Yang 2025 / He 2509.22335 soft-rank surrogate — fully
    differentiable via torch.linalg.svdvals, which backprops through the SVD.

    The returned loss is the *negative* sum of per-layer effective ranks:
        L2ER = -sum_l erank(input_cov_l)

    so that *subtracting* lambda * L2ER from the MSE loss is equivalent to
    maximizing effective rank (pushing the representations toward full rank).

    Only fc1 and fc2 inputs are regularized; the output projection (mu) is
    skipped — the output head is a scalar projector and its rank is inherently 1.

    Args:
        actor_net: A DDPGActorNetwork (must have .fc1, .fc2, .relu attributes).
        states: Input batch of shape (batch, input_size). Must be on the same
            device as actor_net.
        eps: Numerical floor for singular values to prevent log(0).

    Returns:
        Scalar tensor: -sum(erank_fc1, erank_fc2). Fully connected to the
        computation graph — backward() populates gradients on actor_net.parameters().
    """
    def _erank(M: T.Tensor) -> T.Tensor:
        """Soft effective rank of matrix M (batch, dim) via input covariance.

        torch.linalg.svdvals is not implemented on MPS (Apple Silicon) as of
        PyTorch 2.x. We move the matrix to CPU for the SVD, keeping the result
        on the original device so gradient flow is preserved through the .to()
        call. This is a small CPU detour (matrix is at most batch×dim, typically
        16×400) and does not break the autograd graph.
        """
        # Input covariance: (dim, dim) — centre the batch
        M_c = M - M.mean(dim=0, keepdim=True)
        # Use the batch matrix directly for SVD (batch, dim) rather than
        # forming (dim, dim) explicitly — avoids O(dim^2) memory and is
        # identical in spectral structure up to a 1/(N-1) scale that cancels
        # in the normalisation step.
        M_cpu = M_c.to('cpu')  # MPS fallback: svdvals requires CPU on Apple Silicon
        s = T.linalg.svdvals(M_cpu).to(M.device)
        s = T.clamp(s, min=eps)
        s2 = s ** 2
        p = s2 / s2.sum()
        # Shannon entropy (nats) then exp → effective rank
        H = -(p * p.log()).sum()
        return H.exp()

    # Partial forward: capture inputs at fc1 and fc2
    x_fc1 = states                                  # input to fc1
    pre_act1 = actor_net.fc1(x_fc1)
    if actor_net.use_layer_norm:
        pre_act1 = actor_net.ln1(pre_act1)
    x_fc2 = actor_net.relu(pre_act1)               # input to fc2

    erank_fc1 = _erank(x_fc1)
    erank_fc2 = _erank(x_fc2)

    # Return negative sum so *subtracting* lambda * L2ER maximises rank
    return -(erank_fc1 + erank_fc2)


class Hyperparameters:
    """Configuration container loaded from a YAML config dict.

    Maps YAML keys to instance attributes. Notable name mappings:
    - GSP_LEARNING_FREQUENCY -> self.gsp_learning_offset
    - REPLACE_TARGET_COUNTER -> self.replace_target_ctr

    Also initializes self.time_step = 0 (used by TD3 warmup).
    """
    def __init__(self, config):
        self.gamma = config['GAMMA']
        self.tau = config['TAU']
        self.alpha = config['ALPHA']
        self.beta = config['BETA']
        self.lr = config['LR']

        # Phase 4 — independent GSP head learning rate.
        # Default: same value as the trunk/actor LR (config['LR']), preserving
        # exact legacy behavior for all existing batches. When set to a different
        # value, the GSP prediction head's Adam optimizer uses gsp_head_lr while
        # the main action-network optimizer continues to use self.lr.
        # Only affects the GSP actor/head network; the GSP critic and target
        # networks are unchanged (they remain tied to self.lr).
        self.gsp_head_lr = float(config.get('GSP_HEAD_LR', self.lr))

        self.epsilon = config['EPSILON']
        self.eps_min = config['EPS_MIN']
        self.eps_dec = config['EPS_DEC']

        self.gsp_learning_offset = config['GSP_LEARNING_FREQUENCY'] #learn after every 1000 action network learning steps
        self.gsp_batch_size = config['GSP_BATCH_SIZE']

        self.batch_size = config['BATCH_SIZE']
        self.mem_size = config['MEM_SIZE']
        self.replace_target_ctr = config['REPLACE_TARGET_COUNTER']

        self.gsp_e2e_enabled = bool(config.get('GSP_E2E_ENABLED', False))
        self.gsp_e2e_lambda = float(config.get('GSP_E2E_LAMBDA', 1.0))
        self.gsp_e2e_linear_output = bool(config.get('GSP_E2E_LINEAR_OUTPUT', False))

        # H-13 closure: LayerNorm in the main DQN/DDQN action network's trunk.
        # Independent of GSP_USE_LAYER_NORM (which only affects the GSP head).
        # Default False preserves legacy behavior. See
        # docs/research/gsp-hypothesis-tracker.md H-13 for the rationale (j44 vs
        # j123: 96% vs 18% final success with same collapsed GSP head, suggesting
        # actor-side LN drives the difference).
        self.actor_use_layer_norm = bool(config.get('ACTOR_USE_LAYER_NORM', False))

        # Per-episode diagnostics instrumentation (FAU, weight norms, effective
        # rank, Q-gap, pred diversity). Default OFF to preserve legacy runs
        # untouched. Opt-in via ``DIAGNOSTICS_ENABLED: true``. Spec:
        # docs/specs/2026-04-17-diagnostics-instrumentation.md.
        self.diagnostics_enabled = bool(config.get('DIAGNOSTICS_ENABLED', False))
        self.diagnostics_freeze_episode = int(config.get('DIAGNOSTICS_FREEZE_EPISODE', 50))
        self.diagnostics_cadence = int(config.get('DIAGNOSTICS_CADENCE', 10))
        self.diagnostics_batch_size = int(config.get('DIAGNOSTICS_BATCH_SIZE', 1024))
        # Optional critic-side diagnostics (weight norms). Default OFF — adds latency
        # and most plasticity signals of interest are on the actor/policy network.
        # Set to True for targeted investigations (e.g., checking whether critic
        # weight norms grow unboundedly during DDPG/TD3 training).
        self.diagnose_critic = bool(config.get('DIAGNOSE_CRITIC', False))

        # Gradient zero fraction (He 2603.21173 OCP Thm 1) — cheap; default ON.
        # Tracks the fraction of weight gradient entries near zero per named layer.
        # Should co-vary with FAU under OCP continuity conditions; divergence
        # indicates a measurement artifact or gradient flow shut-off without
        # activation collapse. Layers probed are taken from DIAGNOSTIC_PROFILE
        # 'grad_layers' key (defaults to 'fau_layers' if not specified).
        self.diagnose_grad_zero = bool(config.get('DIAGNOSE_GRAD_ZERO', True))

        # Activation churn (Tang 2506.00592 C-CHAIN) — cheap once state_dict
        # snapshots exist; default ON. If no before/after snapshots are provided
        # to compute_diagnostics(), this metric is silently skipped. Caller is
        # responsible for snapshotting via copy.deepcopy(net.state_dict()) around
        # a training step.
        self.diagnose_churn = bool(config.get('DIAGNOSE_CHURN', True))

        # KFAC Hessian effective rank (He 2509.22335 Thm 6.2) — most expensive of
        # the three new metrics; requires a full forward+backward pass plus covariance
        # matrix construction per layer. Default OFF; opt-in for targeted
        # investigations of Hessian rank collapse. Layers probed are taken from
        # DIAGNOSTIC_PROFILE 'kfac_layers' key (defaults to 'fau_layers' if not
        # specified).
        self.diagnose_kfac = bool(config.get('DIAGNOSE_KFAC', False))

        # H-14 GSP-minus ablation flag. When True, the GSP head still runs (gets
        # trained, produces predictions) but those predictions are REPLACED WITH
        # ZERO before concatenation into the actor's augmented observation. This
        # is the QMIP-minus pattern — same architecture, same training, signal
        # removed — the clean test of "does the GSP prediction contribute?".
        # Applied in RL-CollectiveTransport's agent.make_agent_state; the flag
        # only matters if the host code reads it (gsp-rl does not use the value
        # itself). Default False preserves legacy behavior.
        self.gsp_zero_out_signal = bool(config.get('GSP_ZERO_OUT_SIGNAL', False))

        # Candidate A — change what the GSP head predicts. Default 'delta_theta'
        # is the legacy collective-Δθ target used in all dissertation runs (and
        # the target that produced the head collapse documented in H-13/H-14).
        # 'future_prox' retargets each robot's head to predict its own proximity
        # K=GSP_PREDICTION_HORIZON steps ahead — non-self-referential because
        # prox is determined by environment geometry + multi-agent action, not
        # directly chosen by the robot's own action. The flag is read here; the
        # delayed-label buffer that produces (state_t, prox_{t+K}) training pairs
        # lives in the host code (RL-CollectiveTransport agent.py).
        self.gsp_prediction_target = str(config.get('GSP_PREDICTION_TARGET', 'delta_theta'))
        self.gsp_prediction_horizon = int(config.get('GSP_PREDICTION_HORIZON', 5))

        # GSP_OUTPUT_KIND — controls how many targets the GSP head predicts.
        # Motivated by He 2509.22335 Theorem 6.2: rank(Hessian) <= P - k_τ*(I+O+1).
        # Increasing output dim O directly raises the achievable Hessian rank,
        # potentially breaking the rank-1 collapse pattern observed with O=1.
        #
        # Supported values:
        #   'delta_theta_1d'        (default) O=1  — legacy Δθ scalar, backward compat
        #   'future_prox_1d'                  O=1  — per-agent future proximity scalar
        #   'cyl_kinematics_3d'               O=3  — (cyl_Δx, cyl_Δy, cyl_Δθ) per step
        #   'cyl_kinematics_goal_4d'          O=4  — above + group_centroid_Δ_to_goal
        #   'time_to_goal_1d'                 O=1  — remaining steps to success (or 0)
        #
        # gsp_output_size_effective is the O to use when building the GSP head.
        # The legacy gsp_output_size kwarg (from config['GSP_OUTPUT_SIZE']) is kept
        # for backward compat on non-GSP_OUTPUT_KIND runs; this field overrides it
        # when GSP_OUTPUT_KIND is set to a non-default value.
        _GSP_OUTPUT_KIND_SIZES = {
            'delta_theta_1d': 1,
            'future_prox_1d': 1,
            'cyl_kinematics_3d': 3,
            'cyl_kinematics_goal_4d': 4,
            'time_to_goal_1d': 1,
        }
        self.gsp_output_kind = str(config.get('GSP_OUTPUT_KIND', 'delta_theta_1d'))
        if self.gsp_output_kind not in _GSP_OUTPUT_KIND_SIZES:
            raise ValueError(
                f"Unknown GSP_OUTPUT_KIND '{self.gsp_output_kind}'. "
                f"Valid values: {list(_GSP_OUTPUT_KIND_SIZES)}"
            )
        self.gsp_output_size_effective = _GSP_OUTPUT_KIND_SIZES[self.gsp_output_kind]

        # Weight initialization scheme for the GSP head's hidden layers.
        # 'fanin' (default) preserves legacy behavior for all in-flight runs.
        # 'kaiming' uses Kaiming He normal init — see DDPGActorNetwork docstring.
        self.gsp_init_scheme = str(config.get('GSP_INIT_SCHEME', 'fanin'))

        # Phase 3 — effective-rank regularization on the GSP head.
        # When > 0 the MSE loss is reduced by lambda_er * sum(effective_rank per
        # layer), pushing activations toward higher-rank (less collapsed)
        # representations. See He et al. 2509.22335 for the theoretical grounding
        # (OCP Theorem 1) and the Stelaris Phase-3 experiment plan.
        # Default 0.0 → strict no-op, all historical runs unaffected.
        self.gsp_l2er_lambda = float(config.get('GSP_L2ER_LAMBDA', 0.0))

        # C-CHAIN churn-minimizing regularization on the GSP head
        # (Tang et al. 2506.00592, ICML 2025). When > 0, after the main MSE
        # (+ optional L2-ER) optimizer step, a second backward+step penalizes
        # the L2 change in head outputs on the same mini-batch:
        #   L_cchain = λ * F.mse_loss(head(states), pre_step_outputs.detach())
        # This counteracts plasticity loss by limiting how much each mini-batch
        # shifts the network's function on the training distribution.
        # Default 0.0 → strict no-op; all historical runs are unaffected.
        self.gsp_cchain_lambda = float(config.get('GSP_CCHAIN_LAMBDA', 0.0))

        self.noise = config['NOISE']
        self.update_actor_iter = config['UPDATE_ACTOR_ITER']
        self.warmup = config['WARMUP']
        self.time_step = 0
        # H-phase5-4: when True, skip the GSP head's optimizer step entirely.
        # Head stays at random init for the run. Default False preserves all
        # prior behavior. Read in actor.py:502 in the learn() loop.
        self.gsp_head_frozen = bool(config.get('GSP_HEAD_FROZEN', False))

        # JEPA (Joint Embedding Predictive Architecture) latent-space head.
        # When enabled, the legacy scalar future_prox prediction is replaced
        # by: online encoder → predictor → latent MSE against EMA target encoder.
        # The actor receives the 32-d encoder output instead of the 1-d gsp_pred.
        # Default False — all existing runs are unaffected.
        self.gsp_jepa_enabled = bool(config.get('GSP_JEPA_ENABLED', False))
        self.gsp_encoder_dim = int(config.get('GSP_ENCODER_DIM', 32))
        self.gsp_encoder_ema_tau = float(config.get('GSP_ENCODER_EMA_TAU', 0.995))

class NetworkAids(Hyperparameters):
    """Network factory, learning algorithms, action selection, and memory management.

    All methods operate on a 'networks' dict (plain dict, not a class) that
    contains the neural networks, replay buffer, learning scheme string, and
    step counter. This dict is either self.networks or self.gsp_networks,
    passed explicitly to allow the same learn/action methods to serve both
    the main action network and the GSP prediction network.
    """
    def __init__(self, config):
        super().__init__(config)
    def make_DQN_networks(self, nn_args):
        return {'q_eval':DQN(**nn_args), 'q_next':DQN(**nn_args)}
    
    def make_DDQN_networks(self, nn_args):
        return {'q_eval':DDQN(**nn_args), 'q_next':DDQN(**nn_args)}
    
    def make_DDPG_networks(self, actor_nn_args, critic_nn_args):
        DDPG_networks = {
                        'actor': DDPGActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': DDPGActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic': DDPGCriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic': DDPGCriticNetwork(**critic_nn_args, name = 'target_critic_1')}
        return DDPG_networks

    def make_TD3_networks(self, actor_nn_args, critic_nn_args):
        TD3_networks = {
                        'actor': TD3ActorNetwork(**actor_nn_args, name = 'actor'),
                        'target_actor': TD3ActorNetwork(**actor_nn_args, name = 'target_actor'),
                        'critic_1': TD3CriticNetwork(**critic_nn_args, name = 'critic_1'),
                        'target_critic_1': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_1'),
                        'critic_2': TD3CriticNetwork(**critic_nn_args, name = 'critic_2'),
                        'target_critic_2': TD3CriticNetwork(**critic_nn_args, name = 'target_critic_2')}
        return TD3_networks
    
    def make_RDDPG_networks(self, lstm_nn_args, actor_nn_args, critic_nn_args):
        shared_ee = EnvironmentEncoder(**lstm_nn_args)
        RDDPG_networks = {
            'actor': RDDPGActorNetwork(shared_ee, DDPGActorNetwork(**actor_nn_args, name='actor')),
            'target_actor': RDDPGActorNetwork(EnvironmentEncoder(**lstm_nn_args), DDPGActorNetwork(**actor_nn_args, name='target_actor')),
            'critic': RDDPGCriticNetwork(shared_ee, DDPGCriticNetwork(**critic_nn_args, name = 'critic')),
            'target_critic':RDDPGCriticNetwork(EnvironmentEncoder(**lstm_nn_args), DDPGCriticNetwork(**critic_nn_args, name = 'target_critic'))
        }
        return RDDPG_networks
    
    def make_Environmental_Encoder(self, nn_args):
        lstm_networks = {'ee': EnvironmentEncoder(**nn_args)}
        return lstm_networks

    def make_Attention_Encoder(self, nn_args):
        Attention_networks = {'attention': AttentionEncoder(**nn_args)}
        return Attention_networks

    def update_DDPG_network_parameters(self, tau, networks):
        # Update Actor Network
        for target_param, param in zip(networks['target_actor'].parameters(), networks['actor'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        # Update Critic Network
        for target_param, param in zip(networks['target_critic'].parameters(), networks['critic'].parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        
        return networks

    def update_TD3_network_parameters(self, tau, networks):
        actor_params = networks['actor'].named_parameters()
        critic_1_params = networks['critic_1'].named_parameters()
        critic_2_params = networks['critic_2'].named_parameters()
        target_actor_params = networks['target_actor'].named_parameters()
        target_critic_1_params = networks['target_critic_1'].named_parameters()
        target_critic_2_params = networks['target_critic_2'].named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + (1-tau)*target_actor[name].clone()

        networks['target_critic_1'].load_state_dict(critic_1)
        networks['target_critic_2'].load_state_dict(critic_2)
        networks['target_actor'].load_state_dict(actor)

        return networks

    def DQN_DDQN_choose_action(self, observation, networks):
        state = T.tensor(observation, dtype = T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(state)
        return T.argmax(action_values).item()

    def DQN_DDQN_choose_action_batch(self, observations, networks):
        """Batched action selection for DQN/DDQN. Returns list of action indices."""
        states = T.tensor(np.array(observations), dtype=T.float).to(networks['q_eval'].device)
        action_values = networks['q_eval'].forward(states)
        return T.argmax(action_values, dim=1).cpu().tolist()
    
    def DDPG_choose_action(self, observation, networks):
        if networks['learning_scheme'] == 'RDDPG':
            # if using LSTM we need to add an extra dimension
            state = T.tensor(np.array(observation), dtype=T.float).to(networks['actor'].device)
            mu, _ = networks['actor'].forward(state)
            return mu.unsqueeze(0)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            return networks['actor'].forward(state).unsqueeze(0)
        
    
    def DDPG_choose_action_batch(self, observations, networks):
        """Batched action selection for DDPG. Returns (batch, output_size) numpy array."""
        if networks['learning_scheme'] == 'RDDPG':
            # RDDPG uses sequences — cannot batch across robots (stateful LSTM)
            raise NotImplementedError("RDDPG cannot be batched — use sequential choose_action")
        states = T.tensor(np.array(observations), dtype=T.float).to(networks['actor'].device)
        return networks['actor'].forward(states)

    def TD3_choose_action_batch(self, observations, networks, n_actions):
        """Batched action selection for TD3. Returns list of (1, output_size) numpy arrays."""
        if self.time_step < self.warmup:
            batch_size = len(observations)
            mus = T.tensor(np.random.normal(scale=self.noise, size=(batch_size, n_actions)),
                           dtype=T.float).to(networks['actor'].device)
        else:
            states = T.tensor(np.array(observations), dtype=T.float).to(networks['actor'].device)
            mus = networks['actor'].forward(states).to(networks['actor'].device)
        noise = T.tensor(np.random.normal(scale=self.noise, size=mus.shape),
                         dtype=T.float).to(networks['actor'].device)
        mus_prime = T.clamp(mus + noise, -networks['actor'].min_max_action,
                            networks['actor'].min_max_action)
        self.time_step += 1
        return mus_prime.cpu().detach().numpy()

    def TD3_choose_action(self, observation, networks, n_actions):
        if self.time_step < self.warmup:
            mu = T.tensor(np.random.normal(scale = self.noise,
                                           size = (n_actions,)),
                          dtype=T.float).to(networks['actor'].device)
        else:
            state = T.tensor(observation, dtype = T.float).to(networks['actor'].device)
            mu = networks['actor'].forward(state).to(networks['actor'].device)
        mu_prime = mu + T.tensor(np.random.normal(scale = self.noise), dtype = T.float).to(networks['actor'].device)
        mu_prime = T.clamp(mu_prime, -networks['actor'].min_max_action, networks['actor'].min_max_action)
        self.time_step += 1
        return mu_prime.unsqueeze(0).cpu().detach().numpy()
    
    def Attention_choose_action(self, observation, networks):
        return networks['attention'](observation).cpu().detach().numpy()

    
    def learn_DQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.int64))

        q_pred = networks['q_eval'](states)[indices, actions.type(T.LongTensor)]

        q_next = networks['q_next'](states_).max(dim=1)[0]

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next

        loss = networks['q_eval'].loss(q_target, q_pred).to(networks['q_eval'].device)
        loss.backward()
        _check_nan(loss, f"DQN loss at step {networks['learn_step_counter']}")

        networks['q_eval'].optimizer.step()
        networks['learn_step_counter'] += 1

        self.decrement_epsilon()

        return loss.item()
    

    def learn_DDQN(self, networks):
        networks['q_eval'].optimizer.zero_grad()

        states, actions, rewards, states_, dones = self.sample_memory(networks)

        indices = T.LongTensor(np.arange(self.batch_size).astype(np.int64))

        q_pred = networks['q_eval'](states)[indices, actions.type(T.LongTensor)]

        q_next = networks['q_next'](states_)
        q_eval = networks['q_eval'](states_)

        max_actions = T.argmax(q_eval, dim = 1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = networks['q_eval'].loss(q_target, q_pred).to(networks['q_eval'].device)

        loss.backward()
        _check_nan(loss, f"DDQN loss at step {networks['learn_step_counter']}")
        
        networks['q_eval'].optimizer.step()

        networks['learn_step_counter']+=1

        self.decrement_epsilon()

        return loss.item()

    def learn_DDQN_e2e(self, networks, gsp_networks):
        """End-to-end joint training of DDQN + GSP head.

        At each learn step:
        1. Sample 7-value batch from main replay (requires gsp_obs_size > 0).
        2. Re-run GSP head on gsp_obs WITH gradient to produce fresh prediction.
        3. Replace the stale GSP scalar in the stored state with the fresh value.
        4. Run DDQN on augmented state.
        5. Compute combined loss: ddqn_loss + lambda * MSE(fresh_gsp, label).
        6. Backward through both networks, clip GSP gradients, step both optimizers.

        The next-state Q-target uses STORED states_ as-is (no GSP re-run) wrapped
        in torch.no_grad() — stable targets are critical for DDQN convergence.

        Args:
            networks: Main DDQN networks dict (must contain 'q_eval', 'q_next',
                'replay', 'learning_scheme', 'learn_step_counter').
            gsp_networks: GSP networks dict (must contain 'actor', 'learning_scheme').

        Returns:
            dict with keys: ddqn_loss, gsp_mse_loss, total_loss, gsp_grad_norm,
                gsp_grad_norm_pre_clip, ddqn_grad_norm, gsp_input_grad,
                gsp_pred_mean, gsp_pred_std, gsp_label_mean, gsp_label_std.
        """
        e2e_lambda = float(getattr(self, 'gsp_e2e_lambda', 1.0))
        device = networks['q_eval'].device

        # --- 1. Sample 7 values directly from main replay ---
        result = networks['replay'].sample_buffer(self.batch_size)
        states_np, actions_np, rewards_np, states_np_, dones_np, gsp_obs_np, gsp_labels_np = result

        states = T.tensor(states_np, dtype=T.float32).to(device)
        actions = T.tensor(actions_np, dtype=T.float32).to(device)
        rewards = T.tensor(rewards_np, dtype=T.float32).to(device)
        states_ = T.tensor(states_np_, dtype=T.float32).to(device)
        dones = T.tensor(dones_np).to(device)
        gsp_obs = T.tensor(gsp_obs_np, dtype=T.float32).to(device)
        gsp_labels = T.tensor(gsp_labels_np, dtype=T.float32).to(device)

        # --- Zero both optimizers before any forward pass ---
        networks['q_eval'].optimizer.zero_grad()
        gsp_networks['actor'].optimizer.zero_grad()

        # --- 2. Re-run GSP head WITH gradient ---
        gsp_pred = gsp_networks['actor'].forward(gsp_obs)
        # gsp_pred shape: (batch, 1) or (batch,) — normalize to (batch, 1)
        if gsp_pred.dim() == 1:
            gsp_pred = gsp_pred.unsqueeze(1)

        # --- 3. Replace stale GSP scalar in state ---
        # State layout: [env_obs(input_size), gsp_scalar(1), optional_gk(...)]
        # self.input_size is the raw env obs dimensionality (e.g. 31).
        gsp_idx = self.input_size
        augmented = T.cat(
            [states[:, :gsp_idx], gsp_pred, states[:, gsp_idx + 1:]], dim=1
        )
        augmented.retain_grad()

        # --- 4. DDQN forward on augmented state ---
        indices = T.LongTensor(np.arange(self.batch_size).astype(np.int64))
        q_pred = networks['q_eval'](augmented)[indices, actions.type(T.LongTensor)]

        # --- 5. Target Q using STORED next-state (no GSP re-run) ---
        with T.no_grad():
            q_next = networks['q_next'](states_)
            q_eval_next = networks['q_eval'](states_)
            max_actions = T.argmax(q_eval_next, dim=1)
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next[indices, max_actions]

        # --- 6. Combined loss ---
        ddqn_loss = networks['q_eval'].loss(q_target, q_pred).to(device)

        if gsp_labels.dim() == gsp_pred.dim() - 1:
            gsp_labels = gsp_labels.unsqueeze(-1)
        else:
            gsp_labels = gsp_labels.view_as(gsp_pred)
        gsp_mse_loss = F.mse_loss(gsp_pred, gsp_labels)

        total_loss = ddqn_loss + e2e_lambda * gsp_mse_loss

        # --- 7. Backward + gradient clipping ---
        total_loss.backward()
        _check_nan(total_loss, f"E2E total loss at step {networks['learn_step_counter']}")

        # Pre-clip GSP grad norm for diagnostics
        gsp_params = list(gsp_networks['actor'].parameters())
        gsp_grad_norm_pre_clip = float(
            T.norm(T.stack([p.grad.norm() for p in gsp_params if p.grad is not None]))
        )

        T.nn.utils.clip_grad_norm_(gsp_networks['actor'].parameters(), max_norm=1.0)

        # Post-clip GSP grad norm
        gsp_grad_norm = float(
            T.norm(T.stack([p.grad.norm() for p in gsp_params if p.grad is not None]))
        )

        # DDQN Q-eval grad norm (before step)
        q_params = list(networks['q_eval'].parameters())
        ddqn_grad_norm = float(
            T.norm(T.stack([p.grad.norm() for p in q_params if p.grad is not None]))
        )

        # Gradient at the GSP input dimension of the augmented state
        gsp_input_grad = None
        if augmented.grad is not None:
            gsp_input_grad = float(augmented.grad[:, gsp_idx].abs().mean().item())

        # --- 8. Step both optimizers ---
        networks['q_eval'].optimizer.step()
        gsp_networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1
        self.decrement_epsilon()

        return {
            'ddqn_loss': ddqn_loss.item(),
            'gsp_mse_loss': gsp_mse_loss.item(),
            'total_loss': total_loss.item(),
            'gsp_grad_norm': gsp_grad_norm,
            'gsp_grad_norm_pre_clip': gsp_grad_norm_pre_clip,
            'ddqn_grad_norm': ddqn_grad_norm,
            'gsp_input_grad': gsp_input_grad,
            'gsp_pred_mean': float(gsp_pred.detach().mean().item()),
            'gsp_pred_std': float(gsp_pred.detach().std().item()),
            'gsp_label_mean': float(gsp_labels.detach().mean().item()),
            'gsp_label_std': float(gsp_labels.detach().std().item()),
        }

    def learn_DDPG(self, networks, gsp = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)
        target_actions = networks['target_actor'](states_)
        q_value_ = networks['target_critic'](states_, target_actions)

        target = T.unsqueeze(rewards, 1) + self.gamma*q_value_

        #Critic Update
        networks['critic'].optimizer.zero_grad()

        q_value = networks['critic'](states, actions)
        value_loss = Loss(q_value, target)
        value_loss.backward()
        _check_nan(value_loss, f"DDPG critic loss at step {networks['learn_step_counter']}")
        networks['critic'].optimizer.step()

        #Actor Update
        networks['actor'].optimizer.zero_grad()

        new_policy_actions = networks['actor'](states)
        actor_loss = -networks['critic'](states, new_policy_actions)
        actor_loss = actor_loss.mean()
        actor_loss.backward()
        _check_nan(actor_loss, f"DDPG actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()
    
    def learn_RDDPG(self, networks, gsp = False, recurrent = False):
        mem_result = self.sample_memory(networks)
        if len(mem_result) == 7:
            states, actions, rewards, states_, dones, h_batch, c_batch = mem_result
            device = networks['actor'].device
            h_0 = T.tensor(np.array(h_batch), dtype=T.float32).to(device)
            c_0 = T.tensor(np.array(c_batch), dtype=T.float32).to(device)
            # h_batch shape: (batch, num_layers, 1, hidden) -> (num_layers, batch, hidden)
            h_0 = h_0.squeeze(2).permute(1, 0, 2).contiguous()
            c_0 = c_0.squeeze(2).permute(1, 0, 2).contiguous()
            hidden_init = (h_0, c_0)
        else:
            states, actions, rewards, states_, dones = mem_result
            hidden_init = None

        # states: (batch, seq_len, obs_dim)
        # actions: (batch, seq_len, act_dim)
        seq_len = states.shape[1]
        burn_in_len = seq_len // 2
        train_len = seq_len - burn_in_len

        # Split into burn-in and training portions
        burn_states = states[:, :burn_in_len, :]
        train_states = states[:, burn_in_len:, :]
        burn_states_ = states_[:, :burn_in_len, :]
        train_states_ = states_[:, burn_in_len:, :]
        train_actions = actions[:, burn_in_len:, :]
        train_rewards = rewards[:, burn_in_len:]

        # Burn-in: refresh hidden state without gradients
        with T.no_grad():
            if burn_in_len > 0:
                # Run burn-in through actor encoder to get hidden state
                _, actor_hidden = networks['actor'].ee(burn_states, hidden=hidden_init)
                _, critic_hidden = networks['critic'].ee(burn_states, hidden=hidden_init)
                _, target_actor_hidden = networks['target_actor'].ee(burn_states_, hidden=hidden_init)
                _, target_critic_hidden = networks['target_critic'].ee(burn_states_, hidden=hidden_init)
            else:
                actor_hidden = hidden_init
                critic_hidden = hidden_init
                target_actor_hidden = hidden_init
                target_critic_hidden = hidden_init

        # Detach hidden states so burn-in gradients don't flow
        if actor_hidden is not None:
            actor_hidden = (actor_hidden[0].detach(), actor_hidden[1].detach())
            critic_hidden = (critic_hidden[0].detach(), critic_hidden[1].detach())
            target_actor_hidden = (target_actor_hidden[0].detach(), target_actor_hidden[1].detach())
            target_critic_hidden = (target_critic_hidden[0].detach(), target_critic_hidden[1].detach())

        # Target computation (no gradients)
        with T.no_grad():
            target_actions, _ = networks['target_actor'](train_states_, hidden=target_actor_hidden)
            q_value_, _ = networks['target_critic'](train_states_, target_actions, hidden=target_critic_hidden)
            # Use last timestep for Bellman target
            q_last_ = q_value_[:, -1, :]  # (batch, 1)
            r_last = train_rewards[:, -1]  # (batch,)
            target = r_last.unsqueeze(1) + self.gamma * q_last_  # (batch, 1)

        # Critic update
        networks['critic'].optimizer.zero_grad()
        q_value, _ = networks['critic'](train_states, train_actions, hidden=critic_hidden)
        q_last = q_value[:, -1, :]  # (batch, 1)
        value_loss = Loss(q_last, target)
        value_loss.backward()
        _check_nan(value_loss, f"RDDPG critic loss at step {networks['learn_step_counter']}")
        networks['critic'].optimizer.step()

        # Actor update
        networks['actor'].optimizer.zero_grad()
        new_policy_actions, _ = networks['actor'](train_states, hidden=actor_hidden)
        # Re-run critic with fresh hidden (detached) for actor loss
        actor_q_val, _ = networks['critic'](train_states, new_policy_actions, hidden=critic_hidden)
        actor_loss = -actor_q_val[:, -1, :].mean()
        actor_loss.backward()
        _check_nan(actor_loss, f"RDDPG actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        return actor_loss.item()

    def learn_TD3(self, networks, gsp = False, recurrent = False):
        states, actions, rewards, states_, dones = self.sample_memory(networks)

        with T.no_grad():
            target_actions = networks['target_actor'].forward(states_)
            noise = T.clamp(
                T.tensor(np.random.normal(0, 0.2, size=target_actions.shape).astype(np.float32)),
                -0.5, 0.5
            ).to(target_actions.device)
            target_actions = T.clamp(target_actions + noise, -self.min_max_action, self.min_max_action)

            q1_ = networks['target_critic_1'].forward(states_, target_actions)
            q2_ = networks['target_critic_2'].forward(states_, target_actions)

            q1_[dones] = 0.0
            q2_[dones] = 0.0

            q1_ = q1_.view(-1)
            q2_ = q2_.view(-1)

            critic_value_ = T.min(q1_, q2_)
            target = rewards + self.gamma * critic_value_

        q1 = networks['critic_1'].forward(states, actions).squeeze()
        q2 = networks['critic_2'].forward(states, actions).squeeze()

        networks['critic_1'].optimizer.zero_grad()
        networks['critic_2'].optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss

        critic_loss.backward()
        _check_nan(critic_loss, f"TD3 critic loss at step {networks['learn_step_counter']}")
        networks['critic_1'].optimizer.step()
        networks['critic_2'].optimizer.step()

        networks['learn_step_counter'] += 1

        if networks['learn_step_counter'] % self.update_actor_iter != 0:
            return 0, 0

        networks['actor'].optimizer.zero_grad()
        actor_q1_loss = networks['critic_1'].forward(states, networks['actor'].forward(states))
        actor_loss = -T.mean(actor_q1_loss)
        actor_loss.backward()
        _check_nan(actor_loss, f"TD3 actor loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        self.update_TD3_network_parameters(self.tau, networks)

        return actor_loss.item()

    def learn_attention(self, networks):
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return 0
        observations, labels = self.sample_attention_memory(networks)
        networks['learn_step_counter'] += 1
        networks['attention'].optimizer.zero_grad()
        pred_headings = networks['attention'](observations)
        loss = Loss(pred_headings, labels.unsqueeze(-1))
        loss.backward()
        _check_nan(loss, f"Attention loss at step {networks['learn_step_counter']}")
        networks['attention'].optimizer.step()
        return loss.item()

    def learn_gsp_mse(self, networks, recurrent: bool = False):
        """Train the GSP prediction network via direct supervised MSE.

        Replaces the DDPG/RDDPG actor-critic path for non-attention GSP variants.
        Samples (state, label) pairs from `networks['replay']`, forwards the state
        through the actor network, and minimizes MSE against the label. The label
        is stored in the action field of the replay buffer by convention — see
        RL-CollectiveTransport Main.py's store_gsp_transition call sites.

        Rationale: see docs/research/2026-04-13-gsp-information-collapse-analysis.md
        in the Stelaris repo. Training the GSP predictor as a DDPG actor-critic
        on a clipped negative-MSE reward produced an information-collapsed
        predictor whose output was worse than predicting the constant mean.
        Direct supervised MSE has a non-vanishing gradient `2(pred-label)` that
        drives the predictor toward the label regardless of how flat the reward
        landscape is.

        Task 5: Optional VICReg variance+covariance penalty on the penultimate
        feature vector (Bardes, Ponce, LeCun ICLR 2022). Guarded by
        self.gsp_vicreg_enabled (default False). When enabled, this ADDS two
        loss terms to the MSE loss targeting dimensional collapse of the
        encoder features — the failure mode that LayerNorm alone only
        partially addresses for MSE regression aux heads (Lyle 2024 +
        literature review 2026-04-15).

        Design notes (from the red-team audit):
        - target_std is scale-aware: defaults to running estimate of label std
          to avoid mismatched-scale saturation of the variance hinge
        - covariance coefficient normalized by feature dim
        - variance hinge: F.relu(target_std - pred_std).mean()
        """
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return None

        vicreg_enabled = getattr(self, 'gsp_vicreg_enabled', False)

        if recurrent:
            mem_result = self.sample_memory(networks)
            if len(mem_result) == 7:
                states, labels, _, _, _, _, _ = mem_result
            else:
                states, labels, _, _, _ = mem_result
            networks['actor'].optimizer.zero_grad()
            preds_out = networks['actor'](states, hidden=None)
            preds = preds_out[0] if isinstance(preds_out, tuple) else preds_out
            if preds.dim() == labels.dim() + 1:
                labels_shaped = labels.unsqueeze(-1)
            else:
                labels_shaped = labels.view_as(preds)
            mse_loss = F.mse_loss(preds, labels_shaped)
            # VICReg not yet supported for recurrent path (RDDPGActorNetwork
            # forward signature differs — would require a separate feature
            # extraction hook). Only apply to non-recurrent for now.
            loss = mse_loss
        else:
            states, labels, _, _, _ = self.sample_memory(networks)
            networks['actor'].optimizer.zero_grad()
            if vicreg_enabled:
                preds, features = networks['actor'].forward(states, return_features=True)
            else:
                preds = networks['actor'].forward(states)
                features = None
            # labels shape: (batch,) or (batch, 1). preds shape: (batch, 1).
            if labels.dim() == preds.dim() - 1:
                labels_shaped = labels.unsqueeze(-1)
            else:
                labels_shaped = labels.view_as(preds)
            mse_loss = F.mse_loss(preds, labels_shaped)
            loss = mse_loss

            if vicreg_enabled and features is not None:
                var_coef = float(getattr(self, 'gsp_vicreg_var_coef', 1.0))
                cov_coef = float(getattr(self, 'gsp_vicreg_cov_coef', 0.04))
                # Scale-aware target_std: match the batch's label std so the
                # variance hinge doesn't force features to saturate the
                # downstream tanh head. Clamp to >= 0.01 for numerical safety.
                with T.no_grad():
                    label_std = float(labels_shaped.std().clamp(min=0.01).item())
                var_loss = vicreg_variance_loss(features, target_std=label_std)
                cov_loss = vicreg_covariance_loss(features)
                loss = mse_loss + var_coef * var_loss + cov_coef * cov_loss

            # Phase 3 — L2-ER regularization.
            # L_total = MSE - lambda_er * sum(erank_per_layer)
            # Minimising L_total maximises effective rank at each layer, counteracting
            # dormancy / rank collapse. gsp_l2er_loss returns the positive erank_sum
            # so we subtract lambda * that quantity from the MSE loss.
            # Guard with hasattr so the path is robust against non-DDPG GSP heads
            # (attention variant does not have .fc1 / .fc2).
            l2er_lambda = float(getattr(self, 'gsp_l2er_lambda', 0.0))
            if l2er_lambda > 0.0 and hasattr(networks['actor'], 'fc1'):
                l2er_erank_sum = -gsp_l2er_loss(networks['actor'], states)
                # gsp_l2er_loss returns -(erank1+erank2); negating yields erank_sum > 0.
                loss = loss - l2er_lambda * l2er_erank_sum

        # Snapshot head outputs BEFORE the MSE backward+step.
        # We capture pre_outputs here (detached, no graph) so that after the
        # optimizer mutates the weights we can measure how much the function
        # changed on this same batch. This is C-CHAIN's "reference batch"
        # snapshot — Tang et al. 2506.00592 Eq. (3).
        cchain_lambda = float(getattr(self, 'gsp_cchain_lambda', 0.0))
        if cchain_lambda > 0.0 and not recurrent:
            with T.no_grad():
                pre_outputs = networks['actor'].forward(states).detach().clone()

        loss.backward()
        _check_nan(loss, f"GSP MSE loss at step {networks['learn_step_counter']}")
        networks['actor'].optimizer.step()

        # C-CHAIN auxiliary step (two-step formulation).
        # After the MSE optimizer step, re-run the head on the same batch and
        # penalize the L2 distance from the pre-step snapshot. A second
        # backward+step is used so the C-CHAIN gradient does NOT mix with the
        # MSE gradient inside the same computation graph — this matches the
        # paper's "run optimizer on churn loss separately" interpretation and
        # avoids modifying the MSE loss value that gets logged.
        # Guard: recurrent path skipped (RDDPGActorNetwork forward signature
        # differs and its use case does not yet have plasticity concerns).
        if cchain_lambda > 0.0 and not recurrent:
            post_outputs = networks['actor'].forward(states)
            if post_outputs.dim() != pre_outputs.dim():
                # Normalize shape (batch,) → (batch, 1) to match pre_outputs
                post_outputs = post_outputs.unsqueeze(-1) if post_outputs.dim() == 1 else post_outputs
                pre_outputs = pre_outputs.unsqueeze(-1) if pre_outputs.dim() == 1 else pre_outputs
            cchain_loss = cchain_lambda * F.mse_loss(post_outputs, pre_outputs)
            networks['actor'].optimizer.zero_grad()
            cchain_loss.backward()
            _check_nan(cchain_loss, f"GSP C-CHAIN loss at step {networks['learn_step_counter']}")
            networks['actor'].optimizer.step()

        networks['learn_step_counter'] += 1

        # Phase 4 — loss-step correlation diagnostic.
        # Compute Pearson correlation between the FRESH forward-pass predictions
        # (the same preds that produced the MSE loss) and the replay-buffer labels.
        # This is intentionally different from gsp_pred_target_corr in hdf5_logger,
        # which accumulates actor-input-path predictions over a full episode (a
        # different code path with a 1-timestep lag). Computing per-batch here and
        # aggregating in the caller lets us compare "is the loss-path head actually
        # learning?" vs "is the actor-input path measurement broken?"
        #
        # Safety contract:
        # - Uses T.no_grad() / detach — zero gradient graph impact.
        # - NaN/zero-variance guard: returns float("nan") when undefined.
        # - Shape agnostic: flattens both arrays before corrcoef.
        # - Recurrent path: preds/labels_shaped not available in that scope,
        #   so we skip and return nan for consistency.
        batch_corr: float = float("nan")
        if not recurrent:
            with T.no_grad():
                _pred_np = preds.detach().cpu().numpy().flatten()
                _lbl_np = labels_shaped.detach().cpu().numpy().flatten()
                if _pred_np.size > 1:
                    _STD_TOL = 1e-12
                    _p_std = float(np.nanstd(_pred_np))
                    _l_std = float(np.nanstd(_lbl_np))
                    if _p_std > _STD_TOL and _l_std > _STD_TOL:
                        _mask = np.isfinite(_pred_np) & np.isfinite(_lbl_np)
                        if _mask.sum() > 1:
                            batch_corr = float(np.corrcoef(_pred_np[_mask], _lbl_np[_mask])[0, 1])

        return loss.item(), batch_corr

    def _update_jepa_target_encoder(self, tau: float) -> None:
        """EMA update: target_p ← tau * target_p + (1 - tau) * online_p.

        Args:
            tau: EMA decay coefficient (e.g. 0.995). Higher = slower update.
        """
        with T.no_grad():
            for online_p, target_p in zip(
                self.gsp_encoder_online.parameters(),
                self.gsp_encoder_target.parameters(),
            ):
                target_p.data.mul_(tau).add_(online_p.data, alpha=1.0 - tau)

    def learn_gsp_jepa(self, networks: dict):
        """Train the JEPA latent-space GSP head.

        Samples (state_t, state_{t+k}) pairs from the JEPA replay buffer
        (state_t in the 'state' slot, state_{t+k} in the 'action' slot by
        convention). Computes:

            z_t     = encoder_online(state_t)           # online encoding
            z_pred  = predictor(z_t)                    # predicted future latent
            z_target = encoder_target(state_{t+k}).detach()   # EMA target

            loss_pred = MSE(z_pred, z_target)           # latent prediction loss

        Optional VICReg variance + covariance penalties on z_t are added
        when self.gsp_vicreg_enabled is True (reusing existing helpers).

        After backward + optimizer step, the target encoder is updated via EMA.

        Returns:
            Tuple (loss_float, latent_stats_dict) where latent_stats_dict has:
                {var: float, rank: float, pred_mse: float}
        """
        if networks['replay'].mem_ctr < self.gsp_batch_size:
            return None

        vicreg_enabled = getattr(self, 'gsp_vicreg_enabled', False)
        tau = float(getattr(self, 'gsp_encoder_ema_tau', 0.995))
        enc_device = self.gsp_encoder_online.device

        # Sample directly from the JEPA replay buffer rather than going through
        # sample_memory(), which requires a 'actor' or 'q_eval' key in networks
        # to determine device. JEPA networks dict has neither — device comes from
        # the encoder module itself.
        result = networks['replay'].sample_buffer(self.gsp_batch_size)
        raw_states, raw_future, _, _, _ = result[0], result[1], result[2], result[3], result[4]
        states = T.tensor(raw_states, dtype=T.float32).to(enc_device)
        # future_states: stored in the 'action' slot by convention (state_{t+k})
        future_states = T.tensor(raw_future, dtype=T.float32).to(enc_device)

        # Forward through online encoder + predictor
        z_t = self.gsp_encoder_online(states)
        z_pred = self.gsp_predictor(z_t)

        # Target: forward through frozen target encoder
        with T.no_grad():
            z_target = self.gsp_encoder_target(future_states)

        loss_pred = F.mse_loss(z_pred, z_target)
        loss = loss_pred

        # Optional VICReg on online encoder output z_t
        if vicreg_enabled:
            var_coef = float(getattr(self, 'gsp_vicreg_var_coef', 1.0))
            cov_coef = float(getattr(self, 'gsp_vicreg_cov_coef', 0.04))
            # target_std: 1.0 (standard VICReg default) — latent lives in
            # unbounded linear space so label-std normalization is not needed.
            var_loss = vicreg_variance_loss(z_t, target_std=1.0)
            cov_loss = vicreg_covariance_loss(z_t)
            loss = loss_pred + var_coef * var_loss + cov_coef * cov_loss

        self._jepa_online_optimizer.zero_grad()
        loss.backward()
        _check_nan(loss, f"JEPA loss at step {networks['learn_step_counter']}")
        self._jepa_online_optimizer.step()

        # EMA update of target encoder
        self._update_jepa_target_encoder(tau)

        networks['learn_step_counter'] += 1

        # Compute latent statistics (no grad)
        with T.no_grad():
            latent_var = float(z_t.var(dim=0).mean().item())
            # Approximate rank: number of singular values above 1% of max
            z_cpu = z_t.detach().cpu()
            try:
                sv = T.linalg.svdvals(z_cpu)
                rank = float((sv > sv[0] * 0.01).sum().item())
            except Exception:
                rank = float("nan")
            pred_mse = float(loss_pred.item())

        latent_stats = {
            'var': latent_var,
            'rank': rank,
            'pred_mse': pred_mse,
        }
        return loss.item(), latent_stats

    def decrement_epsilon(self):
        self.epsilon = max(self.epsilon-self.eps_dec, self.eps_min)

    def store_transition(self, s, a, r, s_, d, networks, gsp_obs=None, gsp_label=None):
        networks['replay'].store_transition(s, a, r, s_, d, gsp_obs=gsp_obs, gsp_label=gsp_label)
    
    def store_attention_transition(self, s, y, networks):
        networks['replay'].store_transition(s, y)

    def sample_memory(self, networks):
        result = networks['replay'].sample_buffer(self.batch_size)
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            device = networks['q_eval'].device
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG', 'TD3'}:
            device = networks['actor'].device

        if len(result) == 7:
            # Two sources of 7-value returns:
            # 1. SequenceReplayBuffer: items 5 and 6 are h_batch, c_batch — lists
            #    of tuples (hidden states), not numpy arrays.
            # 2. ReplayBuffer with gsp_obs_size > 0: items 5 and 6 are numpy
            #    arrays (gsp_obs, gsp_labels). learn_DDQN_e2e calls sample_buffer
            #    directly to get these — legacy callers only need the first 5.
            extra5 = result[5]
            if isinstance(extra5, np.ndarray):
                # E2E replay path: discard gsp_obs and gsp_labels for legacy callers.
                states, actions, rewards, states_, dones = (
                    result[0], result[1], result[2], result[3], result[4]
                )
                states = T.tensor(states, dtype=T.float32).to(device)
                actions = T.tensor(actions, dtype=T.float32).to(device)
                rewards = T.tensor(rewards, dtype=T.float32).to(device)
                states_ = T.tensor(states_, dtype=T.float32).to(device)
                dones = T.tensor(dones).to(device)
                return states, actions, rewards, states_, dones
            else:
                # Sequence replay path: return all 7 (h_batch, c_batch are tensors/tuples).
                states, actions, rewards, states_, dones, h_batch, c_batch = result
                states = T.tensor(states, dtype=T.float32).to(device)
                actions = T.tensor(actions, dtype=T.float32).to(device)
                rewards = T.tensor(rewards, dtype=T.float32).to(device)
                states_ = T.tensor(states_, dtype=T.float32).to(device)
                dones = T.tensor(dones).to(device)
                return states, actions, rewards, states_, dones, h_batch, c_batch
        else:
            states, actions, rewards, states_, dones = result
            states = T.tensor(states, dtype=T.float32).to(device)
            actions = T.tensor(actions, dtype=T.float32).to(device)
            rewards = T.tensor(rewards, dtype=T.float32).to(device)
            states_ = T.tensor(states_, dtype=T.float32).to(device)
            dones = T.tensor(dones).to(device)
            return states, actions, rewards, states_, dones

    def sample_attention_memory(self, networks):
        observations, labels = networks['replay'].sample_buffer(self.batch_size)
        observations = T.tensor(observations, dtype = T.float32).to(networks['attention'].device)
        labels = T.tensor(labels, dtype = T.float32).to(networks['attention'].device)
        return observations, labels