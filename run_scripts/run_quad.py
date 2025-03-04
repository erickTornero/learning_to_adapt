import sys
sys.path.insert(1, '.')
from learning_to_adapt.dynamics.meta_mlp_dynamics import MetaMLPDynamicsModel
from learning_to_adapt.trainers.mb_trainer import Trainer
from learning_to_adapt.policies.mpc_controller import MPCController
from learning_to_adapt.samplers.sampler import Sampler
from learning_to_adapt.logger import logger
from learning_to_adapt.envs_vrep.normalized_env import normalize
from learning_to_adapt.utils.utils import ClassEncoder
from learning_to_adapt.samplers.model_sample_processor import ModelSampleProcessor
from learning_to_adapt.envs_vrep.quadrotor_vrep_env import QuadrotorVrepEnv
import json
import os

EXP_NAME = 'grbalquad_early_stop2'

def run_experiment(config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    port_policy = 21001
    ports = [19999, 20001, 22001,23001,24001]
    #ports = [25001, 26001]
    #env = normalize(config['env'](reset_every_episode=True, task=config['task'], port=19999))
    #_envs = [normalize(config['env'](reset_every_episode=True, port=pt)) for pt in ports]
    # For the policy & etc
    _env = normalize(config['env'](reset_every_episode=True, task=config['task'], port= port_policy))
    dynamics_model = MetaMLPDynamicsModel(
        name="dyn_model",
        env=_env,
        meta_batch_size=config['meta_batch_size'],
        inner_learning_rate=config['inner_learning_rate'],
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes_model'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity_model'],
        batch_size=config['adapt_batch_size'],
    )

    policy = MPCController(
        name="policy",
        env=_env,
        dynamics_model=dynamics_model,
        discount=config['discount'],
        n_candidates=config['n_candidates'],
        horizon=config['horizon'],
        use_cem=config['use_cem'],
        num_cem_iters=config['num_cem_iters'],
    )

    sampler = Sampler(
        env=_env,
        policy=policy,
        n_parallel=config['n_parallel'],
        max_path_length=config['max_path_length'],
        num_rollouts=config['num_rollouts'],
        adapt_batch_size=config['adapt_batch_size'],  # Comment this out and it won't adapt during rollout
        ports = ports
    )

    sample_processor = ModelSampleProcessor(recurrent=True)

    algo = Trainer(
        env=_env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        initial_random_samples=config['initial_random_samples'],
        dynamics_model_max_epochs=config['dynamic_model_epochs'],
    )

    algo.train()

if __name__ == '__main__':
    
    config = {
        # Environment
        'env': QuadrotorVrepEnv,
        'max_path_length': 250,
        'task': None,
        'normalize': True,
            'n_itr': 120,
        'discount': 1.,

        # Policy:
        'n_candidates': 500,
        'horizon': 10,
        'use_cem': False,
        'num_cem_iters': 5,

        # Training:
        'num_rollouts': 5,
        'valid_split_ratio': 0.1,
        'rolling_average_persitency': 0.99,
        'initial_random_samples': True,

        # Dynamics Model
        'meta_batch_size': 10,
        'hidden_nonlinearity_model': 'relu',
        'learning_rate': 1e-3,
        'inner_learning_rate': 0.001,
        'hidden_sizes_model': (512, 512, 512),
        'dynamic_model_epochs': 100,
        'adapt_batch_size': 16,    

        #  Other
        'n_parallel': 5,    
    }
    run_experiment(config)