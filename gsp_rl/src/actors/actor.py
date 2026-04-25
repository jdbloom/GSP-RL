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
        if gsp:
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
            gsp_obs_sz = self.gsp_network_input if getattr(self, 'gsp_e2e_enabled', False) else 0
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete', gsp_obs_size=gsp_obs_sz)
            self.networks['learn_step_counter'] = 0
        elif learning_scheme == 'DDQN':
            nn_args = {
                'id':self.id,
                'lr':self.lr,
                'output_size':self.output_size,
                'input_size':self.network_input_size,
                'use_layer_norm': getattr(self, 'actor_use_layer_norm', False),
            }
            self.networks = self.build_DDQN(nn_args)
            self.networks['learning_scheme'] = 'DDQN'
            gsp_obs_sz = self.gsp_network_input if getattr(self, 'gsp_e2e_enabled', False) else 0
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, 1, 'Discrete', gsp_obs_size=gsp_obs_sz)
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
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous')
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
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous')
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
            self.networks['replay'] = ReplayBuffer(self.mem_size, self.network_input_size, self.output_size, 'Continuous')
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
                    self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, 'Continuous')
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
                    self.gsp_networks['replay'] = ReplayBuffer(self.mem_size, self.gsp_network_input, self.gsp_network_output, 'Continuous')
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
        if self.networks['learn_step_counter'] % self.replace_target_ctr==0:
            self.networks['q_next'].load_state_dict(self.networks['q_eval'].state_dict())
 
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

        Only supports stateless action networks (DQN, DDQN, DDPG, TD3).
        Does NOT support RDDPG or attention — those have state/memory concerns.

        Args:
            observations: list of observation arrays, one per agent.
            networks: network dict (self.networks).
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

        # TODO Not sure why we have n_agents*batch_size + batch_size
        if self.networks['replay'].mem_ctr < self.batch_size: # (self.n_agents*self.batch_size + self.batch_size):
                return

        if self.gsp:
            if self.networks['learn_step_counter'] % self.gsp_learning_offset == 0:
                #print('[DEBUG] Learning Attention', self.networks['learn_step_counter'])
                self.learn_gsp()

        if self.networks['learning_scheme'] == 'DDQN' and getattr(self, 'gsp_e2e_enabled', False) and self.gsp:
            self.replace_target_network()
            result = self.learn_DDQN_e2e(self.networks, self.gsp_networks)
            self.last_e2e_diagnostics = result
            if result is not None:
                self.last_gsp_loss = result.get('gsp_mse_loss')
                return result.get('ddqn_loss')  # Main.py expects a scalar, not a dict
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
        if scheme == 'attention':
            loss = self.learn_attention(self.gsp_networks)
        elif scheme == 'RDDPG':
            loss = self.learn_gsp_mse(self.gsp_networks, recurrent=True)
        elif scheme in {'DDPG', 'TD3'}:
            loss = self.learn_gsp_mse(self.gsp_networks, recurrent=False)
        if loss is not None:
            # Keep the tuple-skip guard for safety in case learn_attention's
            # return type ever changes; learn_gsp_mse returns a plain float.
            if isinstance(loss, tuple):
                return
            self.last_gsp_loss = float(loss)

    def store_agent_transition(self, s, a, r, s_, d, gsp_obs=None, gsp_label=None):
        self.store_transition(s, a, r, s_, d, self.networks, gsp_obs=gsp_obs, gsp_label=gsp_label)
    
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