from gsp_rl.src.actors import Actor
import os
import yaml

containing_folder = os.path.dirname(os.path.realpath(__file__))
config_path = os.path.join(containing_folder, 'config.yml')

with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# DQN/DDQN default hidden dims (from DQN/DDQN constructors)
DQN_FC1_DIMS = 64
DQN_FC2_DIMS = 128

# DDPG/TD3 default hidden dims (from DDPGActorNetwork/DDPGCriticNetwork constructors)
DDPG_FC1_DIMS = 400
DDPG_FC2_DIMS = 300

def test_build_networks_DQN():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
    for name, param in networks['q_eval'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DQN_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DQN_FC2_DIMS, DQN_FC1_DIMS))
        if name == 'fc3.weight':
            assert(shape == (nn_args['output_size'], DQN_FC2_DIMS))

    for name, param in networks['q_next'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DQN_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DQN_FC2_DIMS, DQN_FC1_DIMS))
        if name == 'fc3.weight':
            assert(shape == (nn_args['output_size'], DQN_FC2_DIMS))

def test_build_networks_DDQN():
    nn_args = {
            'id':1,
            'network': 'DDQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
    for name, param in networks['q_eval'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DQN_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DQN_FC2_DIMS, DQN_FC1_DIMS))
        if name == 'fc3.weight':
            assert(shape == (nn_args['output_size'], DQN_FC2_DIMS))

    for name, param in networks['q_next'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DQN_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DQN_FC2_DIMS, DQN_FC1_DIMS))
        if name == 'fc3.weight':
            assert(shape == (nn_args['output_size'], DQN_FC2_DIMS))

def test_build_networks_DDPG():
    nn_args = {
            'id':1,
            'network': 'DDPG',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'mu.weight':
            assert(shape == (nn_args['output_size'], DDPG_FC2_DIMS))

    for name, param in networks['target_actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'mu.weight':
            assert(shape == (nn_args['output_size'], DDPG_FC2_DIMS))

    for name, param in networks['critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'q.weight':
            assert(shape == (1, DDPG_FC2_DIMS))

    for name, param in networks['target_critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'q.weight':
            assert(shape == (1, DDPG_FC2_DIMS))


def test_build_networks_TD3():
    nn_args = {
            'id':1,
            'network': 'TD3',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'mu.weight':
            assert(shape == (nn_args['output_size'], DDPG_FC2_DIMS))

    for name, param in networks['target_actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'mu.weight':
            assert(shape == (nn_args['output_size'], DDPG_FC2_DIMS))

    for name, param in networks['critic_1'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'q1.weight':
            assert(shape == (1, DDPG_FC2_DIMS))

    for name, param in networks['target_critic_1'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'q1.weight':
            assert(shape == (1, DDPG_FC2_DIMS))

    for name, param in networks['critic_2'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'q1.weight':
            assert(shape == (1, DDPG_FC2_DIMS))

    for name, param in networks['target_critic_2'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc1.weight':
            assert(shape == (DDPG_FC1_DIMS, nn_args['input_size']+nn_args['output_size']))
        if name == 'fc2.weight':
            assert(shape == (DDPG_FC2_DIMS, DDPG_FC1_DIMS))
        if name == 'q1.weight':
            assert(shape == (1, DDPG_FC2_DIMS))

def test_build_gsp_networks_DDPG():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':True,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.gsp_networks
    for name, param in networks['actor'].named_parameters():
        shape = tuple(param.size())
        if name == 'mu.weight':
            assert(shape == (nn_args['gsp_output_size'], DDPG_FC2_DIMS))

    for name, param in networks['critic'].named_parameters():
        shape = tuple(param.size())
        if name == 'q.weight':
            assert(shape == (1, DDPG_FC2_DIMS))

def test_build_gsp_networks_Attention():
    nn_args = {
            'id':1,
            'network': 'DQN',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':True,
            'recurrent_gsp':False,
            'attention': True,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    networks = actor.gsp_networks
    for name, param in networks['attention'].named_parameters():
        shape = tuple(param.size())
        if name == 'fc_out.weight':
            assert(shape[0] == nn_args['gsp_output_size'])


def test_build_networks_TD3_e2e_replay_carries_gsp_arrays():
    """Regression: with GSP_E2E_ENABLED the TD3 (continuous) build must allocate
    the main replay with gsp_obs_size > 0, so sample_buffer returns the 7-tuple
    learn_TD3_e2e unpacks. The original TD3 build omitted gsp_obs_size, so the
    continuous buffer returned 5 values and the e2e learn step crashed with
    'not enough values to unpack (expected 7, got 5)' in the real env."""
    import numpy as np
    e2e_config = dict(config)
    e2e_config['GSP_E2E_ENABLED'] = True
    nn_args = {
            'id':1,
            'network': 'TD3',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':True,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': e2e_config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    replay = actor.networks['replay']
    assert replay.gsp_obs_size == nn_args['gsp_input_size'], (
        "TD3 e2e replay must be built with gsp_obs_size == gsp_input_size"
    )
    # Fill and confirm the REAL sample arity is 7 (not the 5 that crashed).
    obs_w = replay.state_memory.shape[1]
    for _ in range(actor.batch_size + 5):
        replay.store_transition(
            np.zeros(obs_w, dtype=np.float32),
            np.zeros(nn_args['output_size'], dtype=np.float32),
            0.0,
            np.zeros(obs_w, dtype=np.float32),
            False,
            gsp_obs=np.zeros(nn_args['gsp_input_size'], dtype=np.float32),
            gsp_label=np.zeros(1, dtype=np.float32),
        )
    result = replay.sample_buffer(actor.batch_size)
    assert len(result) == 7, (
        "TD3 e2e main replay must return 7 values (incl. gsp_obs, gsp_labels)"
    )


def test_build_networks_TD3_non_e2e_replay_has_no_gsp_arrays():
    """Guard the inverse: a plain TD3 build (no e2e) must NOT allocate gsp arrays,
    preserving the legacy 5-value continuous buffer."""
    nn_args = {
            'id':1,
            'network': 'TD3',
            'input_size':32,
            'output_size':2,
            'meta_param_size':2,
            'gsp':False,
            'recurrent_gsp':False,
            'attention': False,
            'gsp_input_size': 6,
            'gsp_output_size': 1,
            'gsp_look_back':2,
            'gsp_sequence_length': 5,
            'config': config,
            'min_max_action':1.0,
    }
    actor = Actor(**nn_args)
    assert actor.networks['replay'].gsp_obs_size == 0, (
        "Plain TD3 (no e2e) must keep the legacy continuous buffer (gsp_obs_size==0)"
    )
