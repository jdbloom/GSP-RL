"""Main agent class for GSP-RL.

Provides the Actor class -- the primary public API for creating RL agents.
Actor builds and manages both the main action-selection network and the
optional GSP (Global State Prediction) network. Inherits learning
algorithms and network factories from NetworkAids.

Inheritance: Actor -> NetworkAids -> Hyperparameters.

See Also: docs/modules/actors.md
"""
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as Adam

import numpy as np
import math

from gsp_rl.src.buffers import(
    ReplayBuffer,
    SequenceReplayBuffer,
    AttentionSequenceReplayBuffer
)
from gsp_rl.src.actors.learning_aids import NetworkAids
from gsp_rl.src.actors.feature_stats import RunningStandardizer
from gsp_rl.src.networks.jepa import JEPAEncoder, JEPAPredictor


class Actor(NetworkAids):
    """Top-level agent class that builds networks, selects actions, and learns.

    Manages two network dicts:
    - self.networks: Main action-selection network (DQN/DDQN/DDPG/TD3/RDDPG).
    - self.gsp_networks: Optional GSP prediction network (DDPG-GSP/RDDPG-GSP/A-GSP).

    When GSP is enabled, the GSP prediction is concatenated with the raw
    observation to form an augmented input (network_input_size = input_size +
    gsp_output_size) for the main action network.

    See docs/class-graph.md for the networks dict schema per algorithm.
    """
    def __init__(
            self,
            id: int,
            config: dict,
            network: str,
            input_size: int,
            output_size: int,  
            min_max_action: int,
            meta_param_size: int,
            gsp: bool = False,
            recurrent_gsp: bool = False,
            attention: bool = False, 
            recurrent_hidden_size: int = 256,
            recurrent_embedding_size: int = 256,
            recurrent_num_layers = 5,
            gsp_input_size: int = 6,
            gsp_output_size: int = 1,
            gsp_min_max_action: float = 1.0,
            gsp_look_back: int = 2,
            gsp_sequence_length: int = 5
        ) -> None:
        """
        id: int -> the id of the agent
        input_size: int -> the size of the observation space coming from the environment
        output_size: int -> the size of the expected action space
        meta_param_size: int -> the encoding size for LSTM
        gsp: bool -> flag to use DDPG-GSP
        recurrent_gsp: bool -> flag to use RDDPG-GSP
        attention: bool -> flag to use A-GSP
        gsp_input_size: int -> the input size to the gsp network
        gsp_output_size: int -> the output size of the gsp network
        gsp_look_back: int -> ...
        seq_len: int -> length of sequence to use as input to A-GSP
        """
        super().__init__(config)

        self.id = id
        self.input_size = input_size
        self.output_size = output_size

        self.min_max_action = min_max_action
        self.meta_param_size = meta_param_size

        self.action_space = [i for i in range(self.output_size)]
        self.failure_action_code = len(self.action_space)

        self.gsp = gsp
        self.recurrent_gsp = recurrent_gsp
        self.attention_gsp = attention
        
        self.gsp_network_input = gsp_input_size
        # GSP_OUTPUT_KIND overrides the gsp_output_size kwarg when a non-default
        # kind is configured. The effective output size is always read from
        # self.gsp_output_size_effective (set in Hyperparameters.__init__ by the
        # GSP_OUTPUT_KIND flag). For backward compat, 'delta_theta_1d' leaves the
        # effective size at 1 and the gsp_output_size kwarg wins via the fallback.
        _effective = getattr(self, 'gsp_output_size_effective', gsp_output_size)
        if getattr(self, 'gsp_output_kind', 'delta_theta_1d') != 'delta_theta_1d':
            self.gsp_network_output = _effective
        else:
            # Default path: use the kwarg (legacy behavior).
            self.gsp_network_output = gsp_output_size
        self.gsp_min_max_action = gsp_min_max_action
        self.gsp_look_back = gsp_look_back
        self.gsp_sequence_length = gsp_sequence_length

        # GSP_E2E_NORMALIZE_FEATURE (opt-in): now that the feature width K
        # (gsp_network_output) is resolved, build the shared RunningStandardizer.
        # NetworkAids.__init__ parsed the flag and left gsp_feature_stats=None (it
        # runs before gsp_network_output is known). ONE instance is shared by the
        # acting splice (RL-CT Agent.make_agent_state) and the learn splices
        # (learn_DDQN_e2e / learn_TD3_e2e) through this same self. Flag off → stays
        # None → both splices byte-identical to today.
        if getattr(self, 'gsp_e2e_normalize_feature', False):
            _feat_dim = int(self.gsp_network_output or 1)
            self.gsp_feature_stats = RunningStandardizer(
                dim=_feat_dim,
                # 0 (default) = legacy all-history Welford; > 0 = EMA mode with
                # that half-life in learn-step updates (see feature_stats.py).
                ema_halflife=getattr(self, 'gsp_e2e_normalize_ema_halflife', 0.0),
            )

        # Task 0 ablation knobs for the GSP head. Defaults preserve legacy behavior.
        # Read from the agent_config.yml so experiments can flip these per-job.
        self.gsp_weight_decay = float(config.get('GSP_WEIGHT_DECAY', 1e-4))
        self.gsp_init_w = float(config.get('GSP_INIT_W', 3e-3))
        # Trunk capacity knobs — affect only the GSP head's MLP hidden layers.
        self.gsp_fc1_dims = int(config.get('GSP_FC1_DIMS', 400))
        self.gsp_fc2_dims = int(config.get('GSP_FC2_DIMS', 300))
        # Task 4: LayerNorm trunk placement on the GSP head. Default False preserves legacy.
        self.gsp_use_layer_norm = bool(config.get('GSP_USE_LAYER_NORM', False))
        # Weight init scheme for the GSP head hidden layers. Default 'fanin' preserves
        # legacy. 'kaiming' uses Kaiming He normal (std=sqrt(2/fan_in)) for ReLU networks.
        # Motivation: fanin init is too narrow for positive-bounded inputs (prox in [0,0.5]),
        # leaving 65-72% of fc1 units dormant from episode 1 (init-investigation j189-194).
        self.gsp_init_scheme = str(config.get('GSP_INIT_SCHEME', 'fanin'))
        # Task 5: VICReg variance+covariance penalty on the GSP head penultimate features.
        # Default disabled. var_coef 1.0 and cov_coef 0.04 follow VICReg paper Table 9;
        # target_std is derived dynamically from label batch std in learn_gsp_mse.
        # See docs/research/gsp-hypothesis-tracker.md H-05 for rationale.
        self.gsp_vicreg_enabled = bool(config.get('GSP_VICREG_ENABLED', False))
        self.gsp_vicreg_var_coef = float(config.get('GSP_VICREG_VAR_COEF', 1.0))
        self.gsp_vicreg_cov_coef = float(config.get('GSP_VICREG_COV_COEF', 0.04))

        self.recurrent_hidden_size = recurrent_hidden_size
        self.recurrent_embedding_size = recurrent_embedding_size
        self.recurrent_num_layers = recurrent_num_layers

        self.network_input_size = self.input_size
        if self.gsp:
            if getattr(self, 'gsp_jepa_enabled', False):
                # JEPA path: actor receives the full latent vector (encoder_dim)
                # instead of the legacy scalar/K-dim GSP prediction.
                # Latent-primary (GSP_ACTOR_LATENT_PRIMARY): the raw env-obs block
                # is DROPPED from the Q-net input, so the actor is forced to route
                # through the latent. network_input_size then starts from 0 (only
                # latent + trailing neighbor/global block) instead of self.input_size.
                # The runtime augmented-obs builder (RL-CollectiveTransport
                # make_agent_state) and the coupled splice (learn_DDQN_jepa_coupled)
                # drop env_obs in lockstep so the three input dims stay coherent.
                if getattr(self, 'gsp_actor_latent_primary', False):
                    self.network_input_size = 0
                self.network_input_size += getattr(self, 'gsp_encoder_dim', 32)
            else:
                # For multi-dim GSP output (gsp_output_kind != delta_theta_1d),
                # the actor's augmented obs grows by the full output vector width,
                # not just 1. agent.py's make_agent_state handles the concatenation.
                self.network_input_size += self.gsp_network_output
        if self.attention_gsp:  
            self.attention_observation = [[0 for _ in range(self.gsp_network_input)] for _ in range(self.gsp_sequence_length)]
        elif self.recurrent_gsp:
            self.recurrent_gsp_network_input = self.gsp_network_input

        self.build_networks(network)
        self.gsp_networks = None

        # JEPA latent-space encoder path. When enabled, skip the legacy
        # DDPG-based GSP build and instantiate encoder + predictor instead.
        # The target encoder is an EMA copy of the online encoder (frozen).
        self.gsp_encoder_online = None
        self.gsp_encoder_target = None
        self.gsp_predictor = None
        self._jepa_online_optimizer = None
        self._jepa_predictor_optimizer = None

        if gsp:
            if getattr(self, 'gsp_jepa_enabled', False):
                import copy
                _enc_dim = getattr(self, 'gsp_encoder_dim', 32)
                _head_lr = getattr(self, 'gsp_head_lr', self.lr)
                self.gsp_encoder_online = JEPAEncoder(
                    input_dim=self.gsp_network_input,
                    latent_dim=_enc_dim,
                    simnorm=getattr(self, 'gsp_jepa_simnorm', False),
                    simnorm_group_size=getattr(
                        self, 'gsp_jepa_simnorm_group_size', 8
                    ),
                )
                # Target encoder: EMA copy — weights frozen, no gradient
                self.gsp_encoder_target = copy.deepcopy(self.gsp_encoder_online)
                for param in self.gsp_encoder_target.parameters():
                    param.requires_grad = False
                # Action-condition the predictor when GSP_JEPA_ACTION_COND is set.
                # action_dim=0 (default) yields the byte-identical legacy predictor.
                _pred_action_dim = 0
                if getattr(self, 'gsp_jepa_action_cond', False):
                    _pred_action_dim = int(getattr(self, 'gsp_jepa_action_dim', 0))
                self.gsp_predictor = JEPAPredictor(
                    latent_dim=_enc_dim, action_dim=_pred_action_dim
                )
                # Optimizers: online encoder + predictor share one optimizer
                self._jepa_online_optimizer = T.optim.Adam(
                    list(self.gsp_encoder_online.parameters())
                    + list(self.gsp_predictor.parameters()),
                    lr=_head_lr,
                )
                # JEPA requires a replay buffer for (state_t, state_{t+k}) pairs.
                # We reuse the standard ReplayBuffer; the future state is stored
                # in the 'action' slot by convention (matches future_prox label path).
                self.gsp_networks = {}
                self.gsp_networks['learning_scheme'] = 'JEPA'
                self.gsp_networks['replay'] = ReplayBuffer(
                    self.mem_size, self.gsp_network_input,
                    self.gsp_network_input, 'Continuous',
                    recency_halflife=self.recency_halflife,
                )
                self.gsp_networks['learn_step_counter'] = 0
                self.gsp_networks['output_size'] = _enc_dim
            else:
                if attention:
                    self.build_gsp_network('attention')
                self.build_gsp_network('DDPG')

        # Information-collapse diagnostic: last GSP learner training loss.
        # NOTE: this is the loss returned by the GSP learner's inner learn step, which means:
        #   - For DDPG/RDDPG/TD3 GSP schemes: actor loss (a critic-derived policy-gradient
        #     signal), NOT the prediction MSE against delta-theta. A collapsed predictor may
        #     not produce an anomalous value here, since the critic's value landscape can
        #     support multiple policy solutions.
        #   - For the attention GSP scheme: genuine prediction MSE against the label.
        # For prediction-collapse detection, prefer the raw per-step squared error captured
        # in RL-CollectiveTransport as `gsp_squared_error` plus the episode-level
        # `gsp_output_std` / `gsp_pred_target_corr` attrs computed in the Stelaris HDF5Logger.
        # Populated by learn_gsp() whenever a GSP learning step fires; reset to None at the
        # start of each learn() call so callers can distinguish "no GSP step this tick" from
        # "GSP step ran".
        self.last_gsp_loss: float | None = None
        # Populated by learn() when the e2e path fires; reset to None each learn() call.
        self.last_e2e_diagnostics = None
        # Phase 4 loss-step correlation diagnostic. Accumulates one float per
        # GSP learn step (the Pearson corr between fresh forward-pass preds and
        # replay-buffer labels for that batch). Collected by Main.py at episode
        # end to produce mean/std attrs in HDF5. Main.py is responsible for
        # clearing this list after consuming it (not reset per-tick like
        # last_gsp_loss, because it accumulates across an episode).
        self.last_gsp_loss_step_corr_samples: list = []
        # JEPA latent stats from the most recent learn_gsp_jepa call. Dict with
        # keys {var, rank, pred_mse}. Reset to None each learn() tick (like
        # last_gsp_loss). Main.py reads this to call hdf5_writer.record_jepa_*.
        self.last_gsp_jepa_stats: dict | None = None

    def build_networks(self, learning_scheme):
        if learning_scheme == 'None':
            self.networks = {'learning_scheme': '', 'learn_step_counter': 0}
        if learning_scheme == 'DQN':
            nn_args = {
                'id':self.id,
                'lr':self.lr,
                'output_size':self.output_size,
                'input_size':self.network_input_size,
                'use_layer_norm': getattr(self, 'actor_use_layer_norm', False),
            }
            self.networks = self.build_DQN(nn_args)
            self.networks['learning_scheme'] = 'DQN'
            # gsp_obs is stored in the main replay for BOTH the legacy e2e path
            # and the coupled-JEPA path (GSP_JEPA_COUPLE_VALUE): both re-encode the
            # raw GSP input WITH gradient inside the actor learn step.
            _needs_gsp_obs = self.gsp_e2e_enabled or getattr(
                self, 'gsp_jepa_couple_value', False
            )
            gsp_obs_sz = self.gsp_network_input if _needs_gsp_obs else 0
            # E2E label width == head output width (K). For the size-K trajectory
            # target the head predicts a K-vector, so the co-indexed E2E label
            # column must be K wide too (learn_DDQN_e2e regresses pred_K vs
            # label_K). Legacy scalar target -> K==1, byte-identical.
            gsp_label_sz = int(self.gsp_network_output) if self.gsp_e2e_enabled else 1
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete', gsp_obs_size=gsp_obs_sz, recency_halflife=self.recency_halflife, gsp_label_size=gsp_label_sz)
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDQN':
            nn_args = {
                'id':self.id,
                'lr':self.lr,
                'output_size':self.output_size,
                'input_size':self.network_input_size,
                'use_layer_norm': getattr(self, 'actor_use_layer_norm', False),
                'critic_loss': getattr(self, 'critic_loss', 'mse'),
            }
            # Successor-Features head (GSP_SF_ENABLED): swap the scalar-Q DDQN pair
            # for the DDQN_SF pair (psi + learned reward-weight w), and allocate the
            # replay's phi column so the per-step cumulant is stored. Default OFF —
            # unset GSP_SF_ENABLED keeps the exact legacy DDQN build.
            _sf = getattr(self, 'gsp_sf_enabled', False)
            _phi_sz = 0
            if _sf:
                nn_args['d_phi'] = self.gsp_sf_phi_dim
                _phi_sz = self.gsp_sf_phi_dim
                self.networks = self.make_DDQN_SF_networks(nn_args)
            else:
                self.networks = self.build_DDQN(nn_args)
            self.networks['learning_scheme'] = 'DDQN'
            # gsp_obs is stored in the main replay for BOTH the legacy e2e path
            # and the coupled-JEPA path (GSP_JEPA_COUPLE_VALUE): both re-encode the
            # raw GSP input WITH gradient inside the actor learn step.
            _needs_gsp_obs = self.gsp_e2e_enabled or getattr(
                self, 'gsp_jepa_couple_value', False
            )
            gsp_obs_sz = self.gsp_network_input if _needs_gsp_obs else 0
            # E2E label width == head output width (K) — see the DQN branch above.
            gsp_label_sz = int(self.gsp_network_output) if self.gsp_e2e_enabled else 1
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete', gsp_obs_size=gsp_obs_sz, recency_halflife=self.recency_halflife, phi_size=_phi_sz, gsp_label_size=gsp_label_sz)
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDPG':
            actor_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':self.network_input_size,
                'lr': self.lr,
                'min_max_action':self.min_max_action}
            critic_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':self.network_input_size + actor_nn_args['output_size'],
                'lr': self.lr
                }
            self.networks = self.build_DDPG(actor_nn_args, critic_nn_args)
            self.networks['learning_scheme'] = 'DDPG'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous', recency_halflife=self.recency_halflife)
            self.networks['output_size'] = self.output_size
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == "RDDPG":
            lstm_nn_args = {
                'lr': self.lr,
                'input_size': self.network_input_size,
                'output_size': self.meta_param_size,
                'embedding_size': self.recurrent_embedding_size,
                'hidden_size': self.recurrent_hidden_size,
                'num_layers': self.recurrent_num_layers,
                'batch_size': self.batch_size
            }
            actor_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':lstm_nn_args['output_size'],
                'lr': self.lr,
                'min_max_action':self.min_max_action}
            critic_nn_args = {
                'id':self.id,
                'output_size':self.output_size,
                'input_size':lstm_nn_args['output_size'] + actor_nn_args['output_size'],
                'lr': self.lr
                }
            self.networks = self.build_RDDPG()
            self.networks['learning_scheme'] = 'RDDPG'
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous', recency_halflife=self.recency_halflife)
            self.networks['output_size'] = self.output_size
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'TD3':
            actor_nn_args = {
                'id':self.id,
                'alpha':self.alpha,
                'input_size': self.network_input_size,
                'fc1_dims':400,
                'fc2_dims':300,
                'output_size':self.output_size,
                'min_max_action': self.min_max_action
            }
            critic_nn_args = {
                'id':self.id,
                'beta':self.beta,
                'input_size':self.network_input_size+actor_nn_args['output_size'],
                'fc1_dims':400,
                'fc2_dims':300,
                'output_size':self.output_size}
            self.networks = self.build_TD3(actor_nn_args, critic_nn_args)
            self.networks['learning_scheme'] = 'TD3'
            # Cross-head e2e: the GSP arrays are stored co-indexed in the MAIN
            # replay (same mechanism as DDQN e2e, line ~297), so learn_TD3_e2e's
            # single 7-return sample_buffer stays aligned (gsp_obs[i]/gsp_labels[i]
            # correspond to states[i]). Without gsp_obs_size the continuous buffer
            # returns only 5 values and the e2e unpack fails at runtime.
            _needs_gsp_obs = self.gsp_e2e_enabled
            gsp_obs_sz = self.gsp_network_input if _needs_gsp_obs else 0
            # E2E label width == head output width (K, or 2K for
            # cyl_displacement_traj) — see the DQN branch above. Previously this
            # path omitted gsp_label_size (default 1), which broadcast-crashed any
            # vector-label E2E target on TD3. Legacy scalar target -> width 1,
            # byte-identical.
            gsp_label_sz = int(self.gsp_network_output) if self.gsp_e2e_enabled else 1
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous', gsp_obs_size=gsp_obs_sz, recency_halflife=self.recency_halflife, gsp_label_size=gsp_label_sz)
            self.networks['output_size'] = self.output_size
            self.networks['learn_step_counter'] = 0
        else:
            print("removed the exception")
            #raise Exception('[ERROR] Learning scheme is not recognised: '+ learning_scheme)


    def build_DQN(self, nn_args):
        return self.make_DQN_networks(nn_args)
    
    def build_DDQN(self, nn_args):
        return self.make_DDQN_networks(nn_args)
    
    def build_DDPG(self, actor_nn_args, critic_nn_args):
        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)

    def build_RDDPG(self, lstm_nn_args, actor_nn_args, critic_nn_args):
        return self.make_RDDPG_networks(lstm_nn_args, actor_nn_args, critic_nn_args)

    def build_TD3(self, actor_nn_args, critic_nn_args):
        return self.make_TD3_networks(actor_nn_args, critic_nn_args)

    def build_gsp_network(self, learning_scheme:str | None =None):
        self.gsp_networks = None
        if self.attention_gsp:
            nn_args = {
                'input_size': self.gsp_network_input,
                'output_size': self.gsp_network_output,
                'min_max_action': self.gsp_min_max_action,
                'encode_size': 2,
                'embed_size':256, 
                'hidden_size':256,
                'heads':8,
                'forward_expansion':4,
                'dropout':0,
                'max_length':self.gsp_sequence_length
            }
            self.gsp_networks = self.make_Attention_Encoder(nn_args)
            self.gsp_networks['learning_scheme'] = 'attention'
            self.gsp_networks['replay'] = AttentionSequenceReplayBuffer(num_observations = self.gsp_network_input, seq_len = 5)
            self.gsp_networks['learn_step_counter'] = 0
        else:
            if learning_scheme == 'DDPG':
                if self.recurrent_gsp:
                    self.gsp_networks = self.build_RDDPG_gsp()
                    self.gsp_networks['learning_scheme'] = 'RDDPG'
                    self.gsp_networks['output_size'] = self.gsp_network_output
                    #self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, 1, 'Continuous', use_gsp = True)
                    self.gsp_networks['replay'] = SequenceReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, self.gsp_sequence_length, hidden_size=self.recurrent_hidden_size, num_layers=self.recurrent_num_layers)
                    #SequenceReplayBuffer(max_sequence=100, num_observations = self.gsp_network_input, num_actions = 1, seq_len = 5)
                    self.gsp_networks['learn_step_counter'] = 0
                else:
                    self.gsp_networks = self.build_DDPG_gsp()
                    self.gsp_networks['learning_scheme'] = 'DDPG'
                    self.gsp_networks['output_size'] = self.gsp_network_output
                    self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, 'Continuous', recency_halflife=self.recency_halflife)
                    self.gsp_networks['learn_step_counter'] = 0
            elif learning_scheme == 'TD3':
                if self.recurrent_gsp:
                    self.gsp_networks = self.build_RTD3_gsp()
                    self.gsp_networks['learning_scheme'] = 'RTD3'
                    self.gsp_networks['output_size']  = self.gsp_network_output
                    self.gsp_networks['replay'] = SequenceReplayBuffer(max_sequence=100, num_observations = self.gsp_network_input, num_actions = self.gsp_network_output, seq_len = 5)
                    self.gsp_networks['learn_step_counter'] = 0
                else:
                    self.gsp_networks = self.build_TD3_gsp()
                    self.gsp_networks['learning_scheme'] = 'TD3'
                    self.gsp_networks['output_size'] = 1
                    self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, 'Continuous', recency_halflife=self.recency_halflife)
                    self.gsp_networks['learn_step_counter'] = 0
            else:
                raise Exception('[Error] gsp learning scheme is not recognised: '+learning_scheme)

    def build_DDPG_gsp(self):
        actor_nn_args = {
            'id':self.id,
            'input_size':self.gsp_network_input,
            'output_size':self.gsp_network_output,
            # Phase 4: use gsp_head_lr (independent of trunk/actor LR).
            # Default: same as self.lr (from config['LR']), so existing batches are
            # bit-for-bit identical. Override via GSP_HEAD_LR in the experiment YAML.
            # The GSP critic LR intentionally stays at self.lr — only the actor/head
            # that produces predictions (and is trained via supervised MSE) gets the
            # independent rate.
            'lr': getattr(self, 'gsp_head_lr', self.lr),
            'min_max_action':self.min_max_action,
            # Task 0 ablation knobs — defaults preserve legacy behavior exactly.
            # Only the GSP-head actor network receives these; the main policy actor
            # path (make_DDPG_networks called from e.g. Main.py algorithm setup) is
            # unchanged because it constructs its own actor_nn_args without these keys.
            'weight_decay': getattr(self, 'gsp_weight_decay', 1e-4),
            'init_w': getattr(self, 'gsp_init_w', 3e-3),
            # Trunk capacity — wider MLP for the GSP head to test the feature-learning
            # collapse hypothesis identified in the 2026-04-14 trajectory deep-dive.
            'fc1_dims': getattr(self, 'gsp_fc1_dims', 400),
            'fc2_dims': getattr(self, 'gsp_fc2_dims', 300),
            # Task 4: LayerNorm in the GSP head trunk. Placement: after fc1 and fc2,
            # before each ReLU. Default False preserves legacy.
            'use_layer_norm': getattr(self, 'gsp_use_layer_norm', False),
            # Task 5: linear (no tanh) output activation for e2e regression head.
            # Default False preserves legacy tanh-bounded output.
            'use_linear_output': getattr(self, 'gsp_e2e_linear_output', False),
            # Weight init scheme for the GSP head's hidden layers.
            # 'fanin' (default) preserves legacy. 'kaiming' uses He normal init.
            'init_scheme': getattr(self, 'gsp_init_scheme', 'fanin'),
        }
        critic_nn_args = {
            'id':self.id,
            'input_size':self.gsp_network_input+self.gsp_network_output,
            'output_size': 1,
            'lr': self.lr
        }
        return self.make_DDPG_networks(actor_nn_args, critic_nn_args)
    
    def build_RDDPG_gsp(self):
        lstm_nn_args = {
            'lr': self.lr,
            'input_size': self.gsp_network_input,
            'output_size': self.meta_param_size,
            'embedding_size': self.recurrent_embedding_size,
            'hidden_size': self.recurrent_hidden_size,
            'num_layers': self.recurrent_num_layers,
            'batch_size': self.batch_size
        }
        actor_nn_args = {
            'id':self.id,
            'input_size':lstm_nn_args['output_size'],
            'output_size':self.gsp_network_output,
            'lr': self.lr,
            'min_max_action':self.min_max_action
        }
        critic_nn_args = {
            'id':self.id,
            'input_size':lstm_nn_args['output_size']+actor_nn_args['output_size'],
            'output_size': 1,
            'lr': self.lr
        }
        # print('[INPUT]: ', lstm_nn_args['input_size'])
        # print('[LSTM OUTPUT]', lstm_nn_args['output_size'])
        # print('[DDPG INPUT]', actor_nn_args['input_size'])
        # print('[DDPG OUTPUT]', actor_nn_args['output_size'])
        return self.make_RDDPG_networks(lstm_nn_args, actor_nn_args, critic_nn_args)

    def update_network_parameters(self, tau = None):
        if tau is None:
            tau = self.tau
        # Update gsp Networks 
        if self.gsp:
            if self.gsp_networks['learning_scheme'] == 'DDPG' or self.gsp_networks['learning_scheme'] == 'RDDPG':
                self.gsp_networks = self.update_DDPG_network_parameters(tau, self.gsp_networks)
            elif self.gsp_networks['learning_scheme'] == 'TD3' or self.gsp_networks['learning_scheme'] == 'RTD3':
                self.gsp_networks = self.update_TD3_network_parameters(tau, self.gsp_networks)
        # Update Action Selection Networks
        if self.networks['learning_scheme'] == 'DDPG':
            self.networks = self.update_DDPG_network_parameters(tau, self.networks)
        elif self.networks['learning_scheme'] == 'TD3':
            self.networks = self.update_TD3_network_parameters(tau, self.networks)

    def replace_target_network(self):
        """Update q_next toward q_eval.

        SOFT_TARGET_TAU == 0 (default): hard copy every REPLACE_TARGET_COUNTER
        learn steps — bit-identical to all prior behavior.

        SOFT_TARGET_TAU > 0: apply a Polyak soft update every learn step
        (q_next ← tau*q_eval + (1-tau)*q_next per parameter) and skip the
        periodic hard reset entirely. Mirrors the DDPG/TD3 copy_ pattern from
        update_DDPG_network_parameters().
        """
        tau = self.soft_target_tau
        if tau > 0.0:
            # Soft Polyak update every step — hard reset does NOT fire.
            for target_param, online_param in zip(
                self.networks['q_next'].parameters(),
                self.networks['q_eval'].parameters(),
            ):
                target_param.data.copy_(
                    target_param.data * (1.0 - tau) + online_param.data * tau
                )
        else:
            # Hard copy on the counter cadence — exact legacy behavior.
            if self.networks['learn_step_counter'] % self.replace_target_ctr == 0:
                self.networks['q_next'].load_state_dict(
                    self.networks['q_eval'].state_dict()
                )
 
    def choose_action(self, observation, networks, test=False):        
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            if test or np.random.random()>self.epsilon:
                actions = self.DQN_DDQN_choose_action(observation, networks)
            else:
                actions = np.random.choice(self.action_space)
            return actions
        elif networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
            actions = self.DDPG_choose_action(observation, networks)
            if not test:
                actions+=T.normal(0.0, self.noise, size = (1, networks['output_size'])).to(networks['actor'].device)
            actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
            return actions[0].cpu().detach().numpy()
        elif networks['learning_scheme'] == 'TD3':
            actions = self.TD3_choose_action(observation, networks, self.output_size)
            return actions[0]
        elif networks['learning_scheme'] == 'attention':
            self.attention_observation.append(observation)
            self.attention_observation.pop(0)
            #print(type(observation))
            observation = np.array(self.attention_observation)
            #print(type(observation))
            observation = T.Tensor(observation).to(networks['attention'].device)
            #print(type(observation))
            return self.Attention_choose_action(observation.unsqueeze(0), networks)
        else:
            raise Exception('[ERROR]: Learning scheme not recognised for action selection ' + networks['learning_scheme'])
    
    def choose_actions_batch(self, observations, networks, test=False):
        """Batched action selection for multiple observations in one forward pass.

        Only supports stateless action networks (DQN, DDQN, DDPG, TD3 — but
        see the TD3 caveat below: its batched semantics DIVERGE from
        sequential beyond fp drift).
        Does NOT support RDDPG or attention — those have state/memory concerns.

        #53 Sub-project B call-site contract (BATCHED_ACTOR_FORWARD): the host
        (RL-CollectiveTransport) routes its per-robot acting loop and the
        stateless DDPG-scheme GSP-head prediction loop through this method —
        one stacked (R, D) forward through the shared CTDE net instead of R
        sequential (D,) forwards — ONLY when the opt-in BATCHED_ACTOR_FORWARD
        flag is set (default False = the byte-identical sequential
        choose_action path). This path is BASELINE-CHANGING, not inert:
          - float-reduction order differs (batched vs per-row matmul), so
            outputs match sequential only within fp tolerance (~1e-6); on a
            near-tie the DQN/DDQN argmax can flip;
          - the R per-robot epsilon-greedy gate draws collapse to ONE
            np.random.random() draw per step (all-explore or all-exploit);
          - DDPG exploration noise is drawn as one (R, K) T.normal instead of
            R sequential (1, K) draws (same count, different stream order).
        Activation is gated on the pre-registered n-seed noise-floor
        re-baseline (docs/superpowers/specs/2026-06-30-training-loop-
        optimization-design.md §7).

        TD3 caveat — NOT covered by the float-drift-only equivalence claim
        above. TD3 batched acting has DIFFERENT warmup/exploration SEMANTICS
        vs the sequential loop, a semantic divergence rather than fp drift:
        TD3_choose_action_batch advances self.time_step once per BATCH
        instead of once per robot (R sequential calls advance it R times, so
        the warmup phase ends R× sooner in env steps under batching), and it
        draws np.random warmup/exploration noise in different shapes and
        stream order (one (R, n_actions) np.random.normal draw vs R separate
        per-robot draws). No host routes TD3 through this method today; any
        future TD3 use requires its OWN equivalence work (warmup accounting +
        RNG-contract tests + re-baseline) before activation.

        Works for the GSP prediction head too: pass self.gsp_networks (the
        stateless 'DDPG' scheme built by build_DDPG_gsp, any output width K);
        clamping intentionally uses self.min_max_action — the same bound the
        sequential choose_action applies to gsp_networks today.

        Args:
            observations: list of observation arrays, one per agent.
            networks: network dict (self.networks or stateless self.gsp_networks).
            test: if True, greedy (no exploration noise/epsilon).

        Returns:
            list of actions, one per observation.
        """
        if networks['learning_scheme'] in {'DQN', 'DDQN'}:
            if test or np.random.random() > self.epsilon:
                return self.DQN_DDQN_choose_action_batch(observations, networks)
            else:
                return [np.random.choice(self.action_space) for _ in observations]
        elif networks['learning_scheme'] == 'DDPG':
            actions = self.DDPG_choose_action_batch(observations, networks)
            if not test:
                actions = actions + T.normal(0.0, self.noise,
                    size=(len(observations), networks['output_size'])).to(networks['actor'].device)
            actions = T.clamp(actions, -self.min_max_action, self.min_max_action)
            return actions.cpu().detach().numpy()
        elif networks['learning_scheme'] == 'TD3':
            return self.TD3_choose_action_batch(observations, networks, self.output_size)
        else:
            raise NotImplementedError(
                f"Batched action selection not supported for {networks['learning_scheme']}. "
                f"Use choose_action() for RDDPG/attention networks.")

    def learn(self):
        # Reset per-tick diagnostic signals. None means "no step ran this tick".
        self.last_gsp_loss = None
        self.last_e2e_diagnostics = None
        self.last_gsp_jepa_stats = None

        # TODO Not sure why we have n_agents*batch_size + batch_size
        if self.networks['replay'].mem_ctr < self.batch_size: # (self.n_agents*self.batch_size + self.batch_size):
                return

        if self.gsp:
            # H-phase5-4: when GSP_HEAD_FROZEN is true, skip the GSP head's
            # optimizer step entirely. Head stays at random init for the
            # entire run. Tests whether reward-shaping wins require head
            # learning at all, or whether reward density alone explains the
            # effect. Default false preserves all prior behavior.
            if not self.gsp_head_frozen:
                if self.networks['learn_step_counter'] % self.gsp_learning_offset == 0:
                    #print('[DEBUG] Learning Attention', self.networks['learn_step_counter'])
                    self.learn_gsp()

        # Coupled-JEPA path (GSP_JEPA_COUPLE_VALUE): re-encode gsp_obs WITH
        # gradient, splice the fresh latent into the augmented state, and train
        # the encoder jointly on the DDQN value loss + JEPA self-prediction loss.
        # Takes precedence over the scalar e2e path when JEPA is enabled.
        if (
            self.networks['learning_scheme'] == 'DDQN'
            and self.gsp
            and getattr(self, 'gsp_jepa_enabled', False)
            and getattr(self, 'gsp_jepa_couple_value', False)
        ):
            self.replace_target_network()
            result = self.learn_DDQN_jepa_coupled(self.networks, self.gsp_networks)
            self.last_e2e_diagnostics = result
            if result is not None:
                self.last_gsp_loss = result.get('jepa_pred_mse')
                return result.get('ddqn_loss')
            return None

        if self.networks['learning_scheme'] == 'DDQN' and self.gsp_e2e_enabled and self.gsp:
            self.replace_target_network()
            result = self.learn_DDQN_e2e(self.networks, self.gsp_networks)
            self.last_e2e_diagnostics = result
            if result is not None:
                self.last_gsp_loss = result.get('gsp_mse_loss')
                return result.get('ddqn_loss')  # Main.py expects a scalar, not a dict
            return None

        # Cross-head charter arm: the same prediction-into-the-actor coupling on
        # the TD3 continuous head. Target updates happen inside learn_TD3_e2e (only
        # on actor-update steps), so no replace/update call here — mirrors the plain
        # TD3 branch below. Main.py reads last_e2e_diagnostics for the h5 logger.
        if self.networks['learning_scheme'] == 'TD3' and self.gsp_e2e_enabled and self.gsp:
            result = self.learn_TD3_e2e(self.networks, self.gsp_networks)
            self.last_e2e_diagnostics = result
            if result is not None:
                self.last_gsp_loss = result.get('gsp_mse_loss')
                return result.get('critic_loss')  # Main.py expects a scalar, not a dict
            return None

        # Successor-Features path (GSP_SF_ENABLED): Q = psi . w, psi trained by its
        # own TD loss and w by reward regression. Mutually exclusive with the
        # coupled-JEPA / e2e paths above (both return before reaching here).
        if (
            self.networks['learning_scheme'] == 'DDQN'
            and getattr(self, 'gsp_sf_enabled', False)
        ):
            self.replace_target_network()
            result = self.learn_DDQN_sf(self.networks)
            self.last_e2e_diagnostics = result
            if result is not None:
                return result.get('sf_psi_loss')
            return None

        if self.networks['learning_scheme'] == 'DQN':
            self.replace_target_network()

            return self.learn_DQN(self.networks)

        elif self.networks['learning_scheme'] == 'DDQN':
            self.replace_target_network()
            return self.learn_DDQN(self.networks)

        elif self.networks['learning_scheme'] == 'DDPG':
            self.update_network_parameters()
            return self.learn_DDPG(self.networks)

        elif self.networks['learning_scheme'] == 'TD3':
            # Target update moved inside learn_TD3 — only on actor update steps
            return self.learn_TD3(self.networks)

    def learn_gsp(self):
        if self.gsp_networks['replay'].mem_ctr < self.gsp_batch_size:
                return
        # HISTORICAL NOTE: this used to dispatch to learn_DDPG / learn_RDDPG /
        # learn_TD3 with gsp=True for non-attention variants, training the GSP
        # predictor as a DDPG actor-critic on a clipped negative-MSE reward.
        # That produced an information-collapsed predictor whose output was
        # empirically worse than predicting the constant mean. Replaced on
        # 2026-04-13 with direct supervised MSE for all non-attention variants.
        # See Stelaris docs/research/2026-04-13-gsp-information-collapse-analysis.md
        # for root cause analysis.
        loss = None
        scheme = self.gsp_networks['learning_scheme']
        if scheme == 'JEPA':
            # Under value-coupling, learn_DDQN_jepa_coupled (called from learn())
            # already trains the encoder's self-prediction loss jointly with the
            # value loss on the SAME encoder, and sets last_gsp_jepa_stats. Running
            # the standalone self-prediction step here would (a) double-train the
            # self-prediction relative to the value loss, and (b) crash under
            # action-conditioning — this JEPA buffer stores (state_t, state_{t+k})
            # with no action slot, so it cannot pass the action tensor the
            # action-conditioned predictor (action_dim>0) requires. Skip it; the
            # coupled step subsumes it.
            if getattr(self, 'gsp_jepa_couple_value', False):
                return
            loss = self.learn_gsp_jepa(self.gsp_networks)
        elif scheme == 'attention':
            loss = self.learn_attention(self.gsp_networks)
        elif scheme == 'RDDPG':
            loss = self.learn_gsp_mse(self.gsp_networks, recurrent=True)
        elif scheme in {'DDPG', 'TD3'}:
            loss = self.learn_gsp_mse(self.gsp_networks, recurrent=False)
        if loss is not None:
            # learn_gsp_mse returns (loss_float, batch_corr_float).
            # learn_gsp_jepa returns (loss_float, latent_stats_dict).
            # learn_attention returns a plain float — keep the tuple dispatch.
            if isinstance(loss, tuple):
                loss_val = loss[0]
                self.last_gsp_loss = float(loss_val)
                extra = loss[1]
                if isinstance(extra, dict):
                    # JEPA path: store latent stats for Main.py to record.
                    self.last_gsp_jepa_stats = extra
                elif isinstance(extra, float):
                    batch_corr = extra
                    # Accumulate per-batch loss-step correlations across all GSP learn
                    # steps within this episode. Main.py reads
                    # last_gsp_loss_step_corr_samples at episode end, computes
                    # mean/std, and passes them to hdf5_writer. Attribute is
                    # initialised in __init__ and cleared by Main.py at episode end.
                    if not math.isnan(batch_corr):
                        self.last_gsp_loss_step_corr_samples.append(batch_corr)
            else:
                self.last_gsp_loss = float(loss)

    def store_agent_transition(self, s, a, r, s_, d, gsp_obs=None, gsp_label=None, phi=None):
        self.store_transition(s, a, r, s_, d, self.networks, gsp_obs=gsp_obs, gsp_label=gsp_label, phi=phi)
    
    def store_gsp_transition(self, s, a, r, s_, d):
        if self.attention_gsp:
            self.store_attention_transition(s, a, self.gsp_networks)
        else:
            self.store_transition(s, a, r, s_, d, self.gsp_networks)

    def freeze_diagnostic_batch(self, gsp_obs_pool=None):
        """Sample + freeze the eval batch used for all per-episode diagnostic
        computations through the rest of training.

        Called once when ``self.diagnostics_freeze_episode`` is reached AND the
        replay buffer has at least ``self.diagnostics_batch_size`` entries.

        Args:
            gsp_obs_pool: optional numpy array of shape (M, gsp_input_size) of
                recent GSP head input vectors (typically captured by Main.py
                from the live training loop). If ``None``, GSP diagnostics will
                be skipped (only actor diagnostics computed).

        Stores on self:
            diag_actor_eval_batch: (batch_size, actor_input_size) float32
            diag_gsp_eval_batch:   (batch_size, gsp_input_size) float32 or None
        """
        if not self.diagnostics_enabled:
            return
        rng = np.random.default_rng(int(self.id) * 7919 + 13)
        replay = self.networks['replay']
        max_mem = min(replay.mem_ctr, replay.mem_size)
        if max_mem < self.diagnostics_batch_size:
            # Not enough samples yet; caller should retry next episode.
            return
        idx = rng.choice(max_mem, self.diagnostics_batch_size, replace=False)
        self.diag_actor_eval_batch = replay.state_memory[idx].astype(np.float32)

        self.diag_gsp_eval_batch = None
        if gsp_obs_pool is not None and len(gsp_obs_pool) >= self.diagnostics_batch_size:
            pool = np.asarray(gsp_obs_pool, dtype=np.float32)
            # Validate shape before committing — a shape-mismatched pool would later
            # crash in compute_diagnostics → GSP head forward, 40+ min into a run.
            # Fail loud with a printed warning and skip GSP head diagnostics rather
            # than crash the whole training subprocess. The actor diagnostics still
            # run. See docs/research/2026-04-22-b008-diag-pool-shape-postmortem.md.
            expected = int(getattr(self, 'gsp_network_input', 0))
            if pool.ndim != 2 or (expected > 0 and pool.shape[1] != expected):
                print(
                    f"[freeze_diagnostic_batch] WARN: gsp_obs_pool shape "
                    f"{pool.shape} does not match GSP head input size "
                    f"({expected},); skipping GSP head diagnostics for the rest "
                    f"of this run. Fix the caller to supply shape "
                    f"(N, {expected}) samples."
                )
            else:
                gsp_idx = rng.choice(len(pool), self.diagnostics_batch_size, replace=False)
                self.diag_gsp_eval_batch = pool[gsp_idx]

    # ---------------------------------------------------------------------------------
    # Diagnostics helpers
    # ---------------------------------------------------------------------------------

    def _main_network(self, networks_dict: dict):
        """Return the primary network from ``networks_dict`` based on learning_scheme.

        - DQN, DDQN  -> ``networks_dict['q_eval']``
        - DDPG, TD3, RDDPG -> ``networks_dict['actor']``
        - attention   -> ``networks_dict.get('actor') or networks_dict.get('attention')``

        Returns ``None`` if the expected key is absent.
        """
        scheme = networks_dict.get('learning_scheme', '')
        if scheme in ('DQN', 'DDQN'):
            return networks_dict.get('q_eval')
        if scheme in ('DDPG', 'TD3', 'RDDPG'):
            return networks_dict.get('actor')
        if scheme == 'attention':
            return networks_dict.get('actor') or networks_dict.get('attention')
        return None

    def _critic_network(self, networks_dict: dict):
        """Return the primary critic from ``networks_dict`` for DDPG/TD3/RDDPG.

        - DDPG, RDDPG -> ``networks_dict.get('critic')``
        - TD3          -> ``networks_dict.get('critic_1')`` (first critic only;
                          see TD3CriticNetwork.DIAGNOSTIC_PROFILE for limitation note)
        - Others       -> ``None``
        """
        scheme = networks_dict.get('learning_scheme', '')
        if scheme in ('DDPG', 'RDDPG'):
            return networks_dict.get('critic')
        if scheme == 'TD3':
            return networks_dict.get('critic_1')
        return None

    @staticmethod
    def _diagnose_network(
        net,
        batch: T.Tensor,
        prefix: str,
        profile: dict,
        diagnose_grad_zero: bool = False,
        diagnose_kfac: bool = False,
        churn_before_state_dict=None,
        churn_after_state_dict=None,
        diagnose_churn: bool = False,
    ) -> dict:
        """Run diagnostics on ``net`` using its ``DIAGNOSTIC_PROFILE``.

        Dispatches to the appropriate diagnostic functions based on the profile
        fields. All diagnostic imports are deferred to this method.

        Args:
            net: ``nn.Module`` with a ``DIAGNOSTIC_PROFILE`` class attribute.
            batch: Eval batch tensor, typically ``(N, input_dim)``.
            prefix: Key prefix for the output dict (e.g., ``'diag_actor'``).
            profile: The ``DIAGNOSTIC_PROFILE`` dict from the network class (or
                fetched from the instance). Must have keys: ``fau_layers``,
                ``wnorm_layers``, ``has_penultimate``, ``output_kind``.
            diagnose_grad_zero: if True, compute gradient zero fraction per layer
                (He 2603.21173 OCP Thm 1). Uses 'grad_layers' profile key
                (falls back to 'fau_layers' if not present).
            diagnose_kfac: if True, compute KFAC Hessian effective rank per layer
                (He 2509.22335 Thm 6.2). Uses 'kfac_layers' profile key
                (falls back to 'fau_layers' if not present).
            churn_before_state_dict: state_dict snapshot before a training step.
                Required (along with churn_after_state_dict and diagnose_churn=True)
                for churn computation.
            churn_after_state_dict: state_dict snapshot after a training step.
            diagnose_churn: if True and both state_dict snapshots are provided,
                compute activation churn (Tang 2506.00592 C-CHAIN).

        Returns:
            Dict of ``{prefix + '_' + metric_key: float}`` entries.
        """
        from gsp_rl.src.actors.diagnostics import (
            compute_fau,
            compute_overactive_fau,
            compute_weight_norms,
            compute_effective_rank,
            compute_q_action_gap,
            compute_hidden_norm,
            compute_attention_entropy,
            compute_grad_zero_fraction,
            compute_churn,
            compute_kfac_hessian_erank,
        )
        import torch.nn.functional as _F

        out: dict = {}
        fau_layers = profile.get('fau_layers', [])
        wnorm_layers = profile.get('wnorm_layers', [])
        has_penultimate = profile.get('has_penultimate', False)
        output_kind = profile.get('output_kind', '')
        # New profile keys — fall back to fau_layers when not explicitly set.
        grad_layers = profile.get('grad_layers', fau_layers)
        kfac_layers = profile.get('kfac_layers', fau_layers)

        # FAU and overactive FAU — only if ReLU layers declared
        if fau_layers:
            for k, v in compute_fau(net, batch, fau_layers).items():
                out[f'{prefix}_{k}'] = v
            for k, v in compute_overactive_fau(net, batch, fau_layers).items():
                out[f'{prefix}_{k}'] = v

        # Weight norms — always run if layers declared
        if wnorm_layers:
            for k, v in compute_weight_norms(net, wnorm_layers).items():
                out[f'{prefix}_{k}'] = v

        # Effective rank — only for networks with a penultimate() method
        if has_penultimate and hasattr(net, 'penultimate'):
            out[f'{prefix}_erank_penult'] = compute_effective_rank(
                net, batch, penultimate_fn='penultimate'
            )

        # Q-action gap — only for discrete Q-value outputs
        if output_kind == 'q_values':
            for k, v in compute_q_action_gap(net, batch).items():
                # Historical key schema: top-level 'diag_q_*' for actor Q-gap,
                # 'diag_gsp_q_*' for GSP Q-gap. We use prefix stripping to maintain
                # backward compat: replace 'diag_actor_' prefix with 'diag_' for the
                # q-gap keys so DDQN runs (j142, j150-170) see the same key names as before.
                if prefix == 'diag_actor':
                    out[f'diag_{k}'] = v
                else:
                    out[f'{prefix}_{k}'] = v

        # LSTM hidden norm — for LSTM-based encoders
        if output_kind == 'lstm_hidden':
            out[f'{prefix}_hidden_norm'] = compute_hidden_norm(net, batch)

        # Attention entropy — for attention-based encoders
        if output_kind == 'attention':
            out[f'{prefix}_attention_entropy'] = compute_attention_entropy(net, batch)

        # --- New metrics ---

        # Gradient zero fraction (cheap — default ON via diagnose_grad_zero flag)
        if diagnose_grad_zero and grad_layers:
            for k, v in compute_grad_zero_fraction(
                net, _F.mse_loss, batch, grad_layers
            ).items():
                out[f'{prefix}_{k}'] = v

        # Activation churn (cheap once snapshots exist — silently skip if missing)
        if (diagnose_churn
                and churn_before_state_dict is not None
                and churn_after_state_dict is not None):
            # Churn is measured on the full network output (always) plus per-layer
            # if fau_layers are declared (consistent with what we already hook).
            churn_layer_names = fau_layers if fau_layers else None
            for k, v in compute_churn(
                net, batch, churn_before_state_dict, churn_after_state_dict,
                layer_names=churn_layer_names,
            ).items():
                out[f'{prefix}_{k}'] = v

        # KFAC Hessian effective rank (expensive — default OFF via diagnose_kfac flag)
        if diagnose_kfac and kfac_layers:
            for k, v in compute_kfac_hessian_erank(net, batch, kfac_layers).items():
                out[f'{prefix}_{k}'] = v

        return out

    @staticmethod
    def _compute_actor_usage_metrics(net, batch: T.Tensor, pred_slice: slice) -> dict:
        """Measure whether the actor's Q-network depends on the GSP prediction
        dims (the ``pred_slice`` columns of its augmented input).

        ``pred_slice`` is an OFFSET-based slice — ``slice(base, base + width)``
        where ``base = self.input_size`` (raw env_obs width) — NOT a tail slice.
        This is correct-by-construction even when trailing columns follow the
        pred block (e.g. ``make_agent_state`` concatenates
        ``(env_obs, gsp_slot, global_knowledge)``, putting global_knowledge AFTER
        the prediction). See the call site in ``compute_diagnostics``.

        Two metrics (2026-07-04 actor-usage pre-registration):

        - ``diag_gsp_actor_saliency``: input-saliency ratio. On a detached clone
          of the frozen eval batch with ``requires_grad_(True)``, backprop the
          scalar ``Q(x).max(dim=1).values.sum()`` (for vector outputs) and take
          ``g = |x.grad|``. The ratio is ``mean(g[:, pred]) / mean(g[:, other])``.
          >> 1 means the Q-value is driven by the prediction; ~0 means ignored.
        - ``diag_gsp_actor_saliency_abs``: the raw ``mean(g[:, pred])`` (absolute
          pred saliency, un-normalized).
        - ``diag_gsp_actor_wnorm_pred_rel``: first Linear layer weight ``W`` has
          shape ``[hidden, in]``; ``col_norm = W.norm(dim=0)`` is the per-input
          contribution. Emit ``mean(col_norm[pred]) / mean(col_norm)``.

        The saliency computation runs a backward pass on a SEPARATE graph over a
        cloned/detached batch — it never touches the network's ``.grad`` buffers
        used by the real optimizer, and it restores the prior grad-enabled state.
        On NaN/inf, emits ``diag_gsp_actor_saliency = 0.0`` plus
        ``diag_gsp_actor_saliency_nan = 1.0``.
        """
        out: dict = {}
        in_dim = net.fc1.weight.shape[1]
        # Boolean mask for the non-pred ("other") columns.
        other_mask = T.ones(in_dim, dtype=T.bool, device=net.fc1.weight.device)
        other_mask[pred_slice] = False

        # --- M1: dQ/d(pred) saliency ---
        # Run on a detached clone so the real optimizer's gradient buffers are
        # untouched. torch.enable_grad() forces autograd even inside a caller's
        # no-grad diagnostics context.
        prev_grad_enabled = T.is_grad_enabled()
        try:
            with T.enable_grad():
                x = batch.detach().clone().requires_grad_(True)
                q = net(x)
                if q.dim() > 1:
                    scalar = q.max(dim=1).values.sum()
                else:
                    scalar = q.sum()
                grad = T.autograd.grad(scalar, x, retain_graph=False, create_graph=False)[0]
            g = grad.abs()
            sal_pred = g[:, pred_slice].mean()
            sal_other = g[:, other_mask].mean()
            ratio = float(sal_pred / (sal_other + 1e-8))
            abs_pred = float(sal_pred)
            if not (math.isfinite(ratio) and math.isfinite(abs_pred)):
                out['diag_gsp_actor_saliency'] = 0.0
                out['diag_gsp_actor_saliency_abs'] = 0.0
                out['diag_gsp_actor_saliency_nan'] = 1.0
            else:
                out['diag_gsp_actor_saliency'] = ratio
                out['diag_gsp_actor_saliency_abs'] = abs_pred
        finally:
            # Never leak the diagnostic backward's grads onto the network's params.
            net.zero_grad(set_to_none=True)
            T.set_grad_enabled(prev_grad_enabled)

        # --- M3: weight-on-pred relative norm ---
        with T.no_grad():
            col_norm = net.fc1.weight.norm(dim=0)  # [in]
            wnorm_rel = float(col_norm[pred_slice].mean() / (col_norm.mean() + 1e-8))
        out['diag_gsp_actor_wnorm_pred_rel'] = wnorm_rel

        return out

    def compute_diagnostics(
        self,
        gsp_predictions_this_episode=None,
        actor_before_state_dict=None,
        actor_after_state_dict=None,
        gsp_before_state_dict=None,
        gsp_after_state_dict=None,
    ) -> dict:
        """Run all diagnostic functions against the frozen eval batches.

        Uses ``DIAGNOSTIC_PROFILE`` on each network class to determine which
        metrics apply. Dispatches via ``_diagnose_network``.

        Returns a dict of namespaced scalar metrics ready to pass to
        ``HDF5Logger.record_episode_diagnostics``. Keys prefixed ``diag_``.

        Args:
            gsp_predictions_this_episode: optional 1-D numpy array of per-timestep
                GSP predictions from the episode, used to compute prediction
                diversity (Shannon entropy of binned predictions).
            actor_before_state_dict: optional state_dict snapshot of the actor
                network taken BEFORE the most recent training step. Required for
                churn computation (along with actor_after_state_dict). Caller
                snapshots via ``copy.deepcopy(net.state_dict())``.
            actor_after_state_dict: optional state_dict snapshot of the actor
                network taken AFTER the most recent training step.
            gsp_before_state_dict: same as actor_before_state_dict but for the
                GSP head network.
            gsp_after_state_dict: same as actor_after_state_dict but for the
                GSP head network.

        Returns:
            Empty dict if diagnostics aren't enabled or the eval batch hasn't
            been frozen yet; otherwise a dict with the full diagnostic set.
        """
        from gsp_rl.src.actors.diagnostics import compute_gsp_pred_diversity

        if not self.diagnostics_enabled:
            return {}
        if not hasattr(self, 'diag_actor_eval_batch') or self.diag_actor_eval_batch is None:
            return {}

        out: dict = {}

        # Read the three new flags (default to safe values if somehow absent)
        diag_grad_zero = getattr(self, 'diagnose_grad_zero', True)
        diag_churn = getattr(self, 'diagnose_churn', True)
        diag_kfac = getattr(self, 'diagnose_kfac', False)

        # --- Actor network diagnostics ---
        main_net = self._main_network(self.networks)
        if main_net is not None:
            device = next(main_net.parameters()).device
            actor_batch = T.from_numpy(self.diag_actor_eval_batch).to(device)
            profile = getattr(type(main_net), 'DIAGNOSTIC_PROFILE', None)
            if profile is not None:
                out.update(self._diagnose_network(
                    main_net, actor_batch, 'diag_actor', profile,
                    diagnose_grad_zero=diag_grad_zero,
                    diagnose_kfac=diag_kfac,
                    churn_before_state_dict=actor_before_state_dict,
                    churn_after_state_dict=actor_after_state_dict,
                    diagnose_churn=diag_churn,
                ))

            # Actor-usage metrics (M1 saliency + M3 weight-on-pred). Only meaningful
            # when GSP is enabled — that's when the actor's augmented input carries
            # a GSP prediction slot. The slot is OFFSET-based, not tail-based:
            # make_agent_state (RL-CollectiveTransport agent.py) concatenates
            # (env_obs, gsp_slot, global_knowledge), so the pred begins immediately
            # after env_obs at index self.input_size (the raw pre-GSP env width,
            # actor.py:79) and has width gsp_network_output (or gsp_encoder_dim under
            # JEPA). When global_knowledge is present the pred sits in the MIDDLE of
            # the augmented vector, NOT at the tail — a tail slice would mismeasure.
            # Guard on the fc1 attribute so networks without a leading Linear layer
            # are skipped safely.
            if self.gsp and hasattr(main_net, 'fc1'):
                if getattr(self, 'gsp_jepa_enabled', False):
                    pred_width = int(getattr(self, 'gsp_encoder_dim', 32))
                else:
                    pred_width = int(self.gsp_network_output)
                # Latent-primary drops the raw env_obs block, so the latent (pred)
                # slot begins at index 0 of the Q-net input; otherwise it begins
                # right after env_obs at self.input_size. Keeping this in sync with
                # network_input_size / the coupled splice makes the actor-usage
                # saliency (the prereg's temporal-prediction-USE metric) measurable
                # under latent-primary instead of hitting the NaN sentinel.
                if getattr(self, 'gsp_actor_latent_primary', False):
                    base = 0
                else:
                    base = int(self.input_size)  # raw env_obs width; pred starts here
                in_dim = main_net.fc1.weight.shape[1]
                if pred_width > 0 and base >= 0 and base + pred_width <= in_dim:
                    pred_slice = slice(base, base + pred_width)
                    out.update(self._compute_actor_usage_metrics(
                        main_net, actor_batch, pred_slice
                    ))
                else:
                    # The pred slot can't sit where expected (misconfigured widths
                    # or an fc1 input size that doesn't match input_size + pred).
                    # Emit the NaN sentinel rather than silently mismeasure.
                    out['diag_gsp_actor_saliency'] = 0.0
                    out['diag_gsp_actor_saliency_nan'] = 1.0

        # --- Critic network diagnostics (opt-in via DIAGNOSE_CRITIC) ---
        if getattr(self, 'diagnose_critic', False):
            critic_net = self._critic_network(self.networks)
            if critic_net is not None:
                c_device = next(critic_net.parameters()).device
                # Critic takes (state, action) but diagnostics probe state-only forward;
                # we skip critic FAU/erank (critic forward signature differs) and only
                # run weight norms, which don't require a forward pass.
                c_profile = getattr(type(critic_net), 'DIAGNOSTIC_PROFILE', None)
                if c_profile is not None:
                    from gsp_rl.src.actors.diagnostics import compute_weight_norms
                    for k, v in compute_weight_norms(
                        critic_net, c_profile.get('wnorm_layers', [])
                    ).items():
                        out[f'diag_critic_{k}'] = v

        # --- GSP head diagnostics ---
        if self.gsp_networks is not None and hasattr(self, 'diag_gsp_eval_batch') and self.diag_gsp_eval_batch is not None:
            gsp_main = self._main_network(self.gsp_networks)
            if gsp_main is not None:
                gsp_device = next(gsp_main.parameters()).device
                gsp_batch = T.from_numpy(self.diag_gsp_eval_batch).to(gsp_device)
                gsp_profile = getattr(type(gsp_main), 'DIAGNOSTIC_PROFILE', None)
                if gsp_profile is not None:
                    out.update(self._diagnose_network(
                        gsp_main, gsp_batch, 'diag_gsp', gsp_profile,
                        diagnose_grad_zero=diag_grad_zero,
                        diagnose_kfac=diag_kfac,
                        churn_before_state_dict=gsp_before_state_dict,
                        churn_after_state_dict=gsp_after_state_dict,
                        diagnose_churn=diag_churn,
                    ))

            # GSP critic (opt-in)
            if getattr(self, 'diagnose_critic', False):
                gsp_critic = self._critic_network(self.gsp_networks)
                if gsp_critic is not None:
                    gc_profile = getattr(type(gsp_critic), 'DIAGNOSTIC_PROFILE', None)
                    if gc_profile is not None:
                        from gsp_rl.src.actors.diagnostics import compute_weight_norms
                        for k, v in compute_weight_norms(
                            gsp_critic, gc_profile.get('wnorm_layers', [])
                        ).items():
                            out[f'diag_gsp_critic_{k}'] = v

        # --- GSP prediction diversity (this-episode entropy, not eval-batch based) ---
        if gsp_predictions_this_episode is not None:
            preds = np.asarray(gsp_predictions_this_episode, dtype=np.float32).ravel()
            if preds.size > 0:
                out['diag_gsp_pred_diversity'] = compute_gsp_pred_diversity(preds)

        return out

    def reset_gsp_sequence(self):
        self.gsp_sequence = [np.zeros(self.gsp_network_input) for i in range(self.gsp_sequence_length)]
    
    def add_gsp_sequence(self, obs):
        self.gsp_sequence.append(obs)
        self.gsp_sequence.pop(0)

    def save_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].save_checkpoint(path)

        elif self.networks['learning_scheme'] == 'DDPG':
            self.networks['actor'].save_checkpoint(path)
            self.networks['target_actor'].save_checkpoint(path)
            self.networks['critic'].save_checkpoint(path)
            self.networks['target_critic'].save_checkpoint(path)

        elif self.networks['learning_scheme'] == 'TD3':
            self.networks['actor'].save_checkpoint(path)
            self.networks['target_actor'].save_checkpoint(path)
            self.networks['critic_1'].save_checkpoint(path)
            self.networks['target_critic_1'].save_checkpoint(path)
            self.networks['critic_2'].save_checkpoint(path)
            self.networks['target_critic_2'].save_checkpoint(path)
        if self.attention_gsp:
            if self.gsp_networks['learning_scheme'] == 'attention':
                self.gsp_networks['attention'].save_checkpoint(path)
        elif self.gsp and getattr(self, 'gsp_jepa_enabled', False):
            # JEPA path: save encoder_online + predictor + target_encoder via torch.save
            import torch
            jepa_state = {
                'encoder_online': self.gsp_encoder_online.state_dict(),
                'encoder_target': self.gsp_encoder_target.state_dict(),
                'predictor': self.gsp_predictor.state_dict(),
            }
            torch.save(jepa_state, f"{path}_jepa.pt")
        elif self.gsp:
            self.gsp_networks['actor'].save_checkpoint(path, self.gsp)
            self.gsp_networks['target_actor'].save_checkpoint(path, self.gsp)
            if self.gsp_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
                self.gsp_networks['critic'].save_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic'].save_checkpoint(path, self.gsp)
            elif self.gsp_networks['learning_scheme'] in {'TD3', 'RTD3'}:
                self.gsp_networks['critic_1'].save_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_1'].save_checkpoint(path, self.gsp)
                self.gsp_networks['critic_2'].save_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_2'].save_checkpoint(path, self.gsp)
        # GSP_E2E_NORMALIZE_FEATURE: the running feature stats are part of the
        # policy (the actor is calibrated to the standardized input scale), so
        # they checkpoint with the networks. Without this, a fresh-process eval
        # reconstructs the standardizer cold and standardize() is the identity
        # — the 2026-07-10 eval-restore incident that voided an ablation batch.
        # Known limitation: in RL-CT independent-learning mode every robot
        # saves to the SAME path, so this file is last-writer-wins — exactly
        # like the id-less DDQN network checkpoint it sits next to. The
        # restored (weights, stats) pair stays self-consistent; fixing the
        # collision means id-suffixing BOTH, together.
        if getattr(self, 'gsp_feature_stats', None) is not None:
            self.gsp_feature_stats.save(f"{path}_feature_stats.npz")

    def load_model(self, path):
        if self.networks['learning_scheme'] == 'DQN' or self.networks['learning_scheme'] == 'DDQN':
            self.networks['q_eval'].load_checkpoint(path)
            #print('-------------------- Weights ------------------')
            #for param in self.q_eval.parameters():
            #    print(param.data)
        elif self.networks['learning_scheme'] == 'DDPG':
            self.networks['actor'].load_checkpoint(path)
            self.networks['target_actor'].load_checkpoint(path)
            self.networks['critic'].load_checkpoint(path)
            self.networks['target_critic'].load_checkpoint(path)

        elif self.networks['learning_scheme'] == 'TD3':
            self.networks['actor'].load_checkpoint(path)
            self.networks['target_actor'].load_checkpoint(path)
            self.networks['critic_1'].load_checkpoint(path)
            self.networks['target_critic_1'].load_checkpoint(path)
            self.networks['critic_2'].load_checkpoint(path)
            self.networks['target_critic_2'].load_checkpoint(path)
        
        if self.attention_gsp:
            if self.gsp_networks['learning_scheme'] == 'attention':
                self.gsp_networks['attention'].load_checkpoint(path)
        elif self.gsp and getattr(self, 'gsp_jepa_enabled', False):
            import os, torch
            jepa_path = f"{path}_jepa.pt"
            if os.path.exists(jepa_path):
                state = torch.load(jepa_path, map_location='cpu')
                self.gsp_encoder_online.load_state_dict(state['encoder_online'])
                self.gsp_encoder_target.load_state_dict(state['encoder_target'])
                self.gsp_predictor.load_state_dict(state['predictor'])
        elif self.gsp:
            self.gsp_networks['actor'].load_checkpoint(path, self.gsp)
            self.gsp_networks['target_actor'].load_checkpoint(path, self.gsp)
            if self.gsp_networks['learning_scheme'] in {'DDPG', 'RDDPG'}:
                self.gsp_networks['critic'].load_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic'].load_checkpoint(path, self.gsp)
            elif self.gsp_networks['learning_scheme'] in {'TD3', 'RTD3'}:
                self.gsp_networks['critic_1'].load_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_1'].load_checkpoint(path, self.gsp)
                self.gsp_networks['critic_2'].load_checkpoint(path, self.gsp)
                self.gsp_networks['target_critic_2'].load_checkpoint(path, self.gsp)
        # Restore the feature-standardizer stats saved by save_model. Missing
        # file → stats stay cold (pre-persistence checkpoints); the eval-side
        # warm-up (RL-CT) is the fallback for those. A corrupt/truncated file
        # (save_model is not atomic; daemon restarts kill mid-checkpoint) must
        # degrade to the same cold-stats fallback, not crash the whole load —
        # but loudly, so the eval config can be pointed at the warm-up.
        if getattr(self, 'gsp_feature_stats', None) is not None:
            import os
            _stats_path = f"{path}_feature_stats.npz"
            if os.path.exists(_stats_path):
                try:
                    self.gsp_feature_stats.restore(_stats_path)
                except ValueError:
                    # dim mismatch — a real config error, never swallow.
                    raise
                except Exception as exc:
                    print(
                        f"[feature_stats] WARNING: could not restore "
                        f"{_stats_path} ({exc!r}); stats stay cold — enable "
                        f"GSP_EVAL_FEATURE_STATS_WARMUP_EPISODES for this eval"
                    )

    def save_gsp_head_snapshot(self, path: str) -> None:
        """Save ONLY the GSP prediction network's weights to `path`.

        Lightweight snapshot used by the training loop to capture intermediate
        GSP-head states for post-hoc best-checkpoint selection. Unlike save_model
        this writes a single file and does not touch the main actor/critic.

        No-op if GSP is disabled.
        """
        if not self.gsp or self.gsp_networks is None:
            return
        if self.gsp_networks.get('learning_scheme') == 'attention':
            net = self.gsp_networks.get('attention')
        else:
            net = self.gsp_networks.get('actor')
        if net is None:
            return
        T.save(net.state_dict(), path)

    def load_gsp_head_snapshot(self, path: str) -> None:
        """Load GSP-head weights previously saved by save_gsp_head_snapshot.

        Restores the primary GSP network (actor or attention) from `path`.
        The target_actor is left alone — this is intended for frozen test-phase
        evaluation, not resumed training.
        """
        if not self.gsp or self.gsp_networks is None:
            return
        if self.gsp_networks.get('learning_scheme') == 'attention':
            net = self.gsp_networks.get('attention')
        else:
            net = self.gsp_networks.get('actor')
        if net is None:
            return
        device = next(net.parameters()).device
        state = T.load(path, map_location=device)
        net.load_state_dict(state)


if __name__=='__main__':
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])

    
    print('[TESTING] DQN')
    agent.build_networks('DQN')
    agent.networks['learn_step_counter'] = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])

    print('[TESTING] DDQN')
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])
    agent.build_networks('DDQN')
    agent.networks['learn_step_counter'] = agent.replace_target_ctr
    agent.replace_target_network()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['q_eval'])
    print(agent.networks['q_next'])
    
    print('[TESTING] DDPG and param update')
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])
    agent.build_networks('DDPG')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [None, agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())

    print('[TESTING] TD3')
    agent_args = {'id':1, 'input_size':32, 'output_size':2, 'options_per_action':3, 'n_agents':1, 'n_chars':2, 'meta_param_size':2, 
                 'gsp':False, 'recurrent_gsp':False, 'gsp_look_back':2}
    agent = Actor(**agent_args)
    #agent.epsilon = agent.eps_min
    observation = np.zeros(agent_args['input_size'])
    agent.build_networks('TD3')
    agent.update_network_parameters()
    observation = np.random.random(size = agent_args['input_size'])
    done = False
    for i in range(200):
        action = [None, agent.choose_action(observation, agent.networks)]
        reward = np.random.random()
        new_obs = np.random.random(size = agent_args['input_size'])
        agent.store_transition(observation, action, reward, new_obs, done, agent.networks)
        observation = new_obs
    print('[LOSS]', agent.learn())
    print(agent.networks['actor'])
    print(agent.networks['critic_1'])
    print(agent.networks['critic_2'])
    

    print('[TESTING] gsp DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, gsp=True, recurrent_gsp = False)
    agent.build_networks('DDPG')
    agent.build_gsp_network('DDPG')
    agent.update_network_parameters()
    print(agent.gsp_networks['actor'])
    print(agent.gsp_networks['critic'])
    

    print('[TESTING] Recurrent gsp DDPG')
    agent = Actor(1, 32, 2, 3, 1, 2, 2, gsp=True, recurrent_gsp = True)
    agent.build_networks('DDPG')
    agent.build_gsp_network('DDPG')
    agent.update_network_parameters()
    print(agent.gsp_networks['actor'])
    print(agent.gsp_networks['critic'])
    print(agent.gsp_networks['ee'])