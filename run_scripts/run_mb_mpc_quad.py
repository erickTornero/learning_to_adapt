import sys
sys.path.insert(1, '.')
from learning_to_adapt.dynamics.mlp_dynamics import MLPDynamicsModel
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

EXP_NAME = 'mb_mpc_quad_2'


def run_experiment(config):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME + '/' + config.get('exp_name', '')
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last')
    json.dump(config, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)


    port_policy = 21001
    ports = [19999, 20001, 22001,23001,24001]

    _env = normalize(config['env'](reset_every_episode=True, task=config['task'], port=port_policy))

    dynamics_model = MLPDynamicsModel(
        name="dyn_model",
        env=_env,
        learning_rate=config['learning_rate'],
        hidden_sizes=config['hidden_sizes'],
        valid_split_ratio=config['valid_split_ratio'],
        rolling_average_persitency=config['rolling_average_persitency'],
        hidden_nonlinearity=config['hidden_nonlinearity'],
        batch_size=config['batch_size'],
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
        num_rollouts=config['num_rollouts'],
        max_path_length=config['max_path_length'],
        n_parallel=config['n_parallel'],
        ports=ports
    )

    sample_processor = ModelSampleProcessor(recurrent=False)

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
    # -------------------- Define Variants -----------------------------------

    config = {
            # Environment
            'env': QuadrotorVrepEnv,
            'task': None,

            # Policy
            'n_candidates': 2000,
            'horizon': 20,
            'use_cem': False,
            'num_cem_iters': 5,
            'discount': 1.,

            # Sampling
            'max_path_length': 250,
            'num_rollouts': 5,
            'initial_random_samples': True,

            # Training
            'n_itr': 100,
            'learning_rate': 1e-4,
            'batch_size': 128,
            'dynamic_model_epochs': 100,
            'valid_split_ratio': 0.1,
            'rolling_average_persitency': 0.99,

            # Dynamics Model
            'hidden_sizes': (512, 512),
            'hidden_nonlinearity': 'relu',


            #  Other
            'n_parallel': 5,
            }

    run_experiment(config)