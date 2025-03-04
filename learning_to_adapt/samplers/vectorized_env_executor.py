import numpy as np
import pickle as pickle
from multiprocessing import Process, Pipe
from learning_to_adapt.envs_vrep.normalized_env import normalize
from learning_to_adapt.envs_vrep.wrapper_quad.wrapper_vrep import VREPQuad
from learning_to_adapt.envs_vrep.quadrotor_vrep_env import QuadrotorVrepEnv
import copy


class IterativeEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Internally, the environments are executed iteratively.

    """

    def __init__(self, env, num_rollouts, max_path_length):
        self._num_envs = num_rollouts
        self.envs = np.asarray([copy.deepcopy(env) for _ in range(self._num_envs)])
        self.ts = np.zeros(len(self.envs), dtype='int')  # time steps
        self.max_path_length = max_path_length

    def step(self, actions):
        """
        Steps the wrapped environments with the provided actions

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool),
             env_infos (dict). Each list is of length meta_batch_size x envs_per_task
             (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        """
        Init 5 envs different ports
        Change Deep copy to create a new env with different ports, enable parallel mode
        """
        
        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)]

        # stack results split to obs, rewards, ...
        obs, rewards, dones, env_infos = list(map(list, zip(*all_results)))

        # reset env when done or max_path_length reached
        dones = np.asarray(dones)
        self.ts += 1
        dones = np.logical_or(self.ts >= self.max_path_length, dones)

        for i in np.argwhere(dones).flatten():
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0

        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments

        Returns:
            (list): list of (np.ndarray) with the new initial observaremoteions.
        """
        obses = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return obses

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs

class ParallelVrepExecutor(object):
    """
    Wrap multiples instances of vrep without loss the id connection
    """
    
    def __init__(self, num_rollouts, max_path_length, ports):
        """
        Initialize Pipes and Process
        """
        n_parallel      =   len(ports)
        
        self._num_envs  =   n_parallel
        self.ports      =   ports
        self.env_       =   None

        assert num_rollouts == self._num_envs
        self.remotes, self.work_remotes =   zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)   
        self.ps = [
            Process(target=self.worker, args=(work_remote, remote, max_path_length, seed, port)) 
            for work_remote, remote, seed, port in zip(self.work_remotes, self.remotes, seeds, ports)
        ]

        for p in self.ps:
            p.daemon = True
            p.start()
        
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions_):
        
        """
        Step for each environment
        """

        for remote, action_list in zip(self.remotes, actions_):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]
        #import pdb
        #pdb.set_trace()
        #print(results)
        obs, rws, dones, env_infos = map(lambda x: x, zip(*results))
        #print(obs)
        #print(rws)    
        #print(dones)
        #print(env_infos)
        return obs, rws, dones, env_infos
    
    def reset(self):
        """
        Reset all environments
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        
        #print(self.remotes[0].recv())
        #print(self.remotes[1].recv())
        observations = [np.asarray(remote.recv(), np.float32) for remote in self.remotes]
        return observations

    def worker(self, remote, parent_remote, max_path_length, seed, port_):
        #env = VREPQuad(port=port_)
        env = QuadrotorVrepEnv(task='rotorless', reset_every_episode=True, port=port_)
        if port_ == self.ports[0]:
            self.env_ = env
        np.random.seed(seed)
        #print(max_path_length)
        #max_path_length = max_path_length[0]
        ts = 0
        while True:
            cmd, data = remote.recv()

            if cmd == 'step':
                action  = data
                nextobs, rw, done, info = env.step(action)
                ts = ts + 1
                if done or ts >= max_path_length:
                    done = True
                    env.reset()
                    ts = 0
                """Send the next observation"""
                remote.send((nextobs, rw, done, info))
            elif cmd =='reset':
                """
                Reset the environment associated with the worker
                """
                obs = env.reset()
                remote.send(obs)
            else:
                print('Warning: Receiving unknown command!!')

    @property
    def getenv(self):
        return self.env_

    @property
    def num_envs(self):
        return self._num_envs


class ParallelEnvExecutor(object):
    """
    Wraps multiple environments of the same kind and provides functionality to reset / step the environments
    in a vectorized manner. Thereby the environments are distributed among meta_batch_size processes and
    executed in parallel.

    """

    def __init__(self, env, n_parallel, num_rollouts, max_path_length):
        assert num_rollouts % n_parallel == 0
        self.envs_per_proc = int(num_rollouts/n_parallel)
        self._num_envs = n_parallel * self.envs_per_proc
        self.n_parallel = n_parallel
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_parallel)])
        seeds = np.random.choice(range(10**6), size=n_parallel, replace=False)

        self.ps = [
            Process(target=worker, args=(work_remote, remote, pickle.dumps(env), self.envs_per_proc, max_path_length, seed))
            for (work_remote, remote, seed) in zip(self.work_remotes, self.remotes, seeds)]  # Why pass work remotes?

        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions):
        """
        Executes actions on each env

        Args:
            actions (list): lists of actions, of length meta_batch_size x envs_per_task

        Returns
            (tuple): a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of meta_envs)
        """
        assert len(actions) == self.num_envs

        # split list of actions in list of list of actions per meta tasks
        chunks = lambda l, n: [l[x: x + n] for x in range(0, len(l), n)]
        actions_per_meta_task = chunks(actions, self.envs_per_proc)

        # step remote environments
        for remote, action_list in zip(self.remotes, actions_per_meta_task):
            remote.send(('step', action_list))

        results = [remote.recv() for remote in self.remotes]

        obs, rewards, dones, env_infos = map(lambda x: sum(x, []), zip(*results))

        return obs, rewards, dones, env_infos

    def reset(self):
        """
        Resets the environments of each worker

        Returns:
            (list): list of (np.ndarray) with the new initial observations.
        """
        for remote in self.remotes:
            remote.send(('reset', None))
        return sum([remote.recv() for remote in self.remotes], [])

    def set_tasks(self, tasks=None):
        """
        Sets a list of tasks to each worker

        Args:
            tasks (list): list of the tasks for each worker
        """
        for remote, task in zip(self.remotes, tasks):
            remote.send(('set_task', task))
        for remote in self.remotes:
            remote.recv()

    @property
    def num_envs(self):
        """
        Number of environments

        Returns:
            (int): number of environments
        """
        return self._num_envs


def worker(remote, parent_remote, env_pickle, n_envs, max_path_length, seed):
    """
    Instantiation of a parallel worker for collecting samples. It loops continually checking the task that the remote
    sends to it.

    Args:
        remote (multiprocessing.Connection):
        parent_remote (multiprocessing.Connection):
        env_pickle (pkl): pickled environment
        n_envs (int): number of environments per worker
        max_path_length (int): maximum path length of the task
        seed (int): random seed for the worker
    """
    parent_remote.close()

    envs = [pickle.loads(env_pickle) for _ in range(n_envs)]
    np.random.seed(seed)

    ts = np.zeros(n_envs, dtype='int')

    while True:
        # receive command and data from the remote
        cmd, data = remote.recv()

        # do a step in each of the environment of the worker
        if cmd == 'step':
            all_results = [env.step(a) for (a, env) in zip(data, envs)]
            obs, rewards, dones, infos = map(list, zip(*all_results))
            ts += 1
            for i in range(n_envs):
                if dones[i] or (ts[i] >= max_path_length):
                    dones[i] = True
                    obs[i] = envs[i].reset()
                    ts[i] = 0
            remote.send((obs, rewards, dones, infos))

        # reset all the environments of the worker
        elif cmd == 'reset':
            obs = [env.reset() for env in envs]
            ts[:] = 0
            remote.send(obs)

        # set the specified task for each of the environments of the worker
        elif cmd == 'set_task':
            for env in envs:
                env.set_task(data)
            remote.send(None)

        # close the remote and stop the worker
        elif cmd == 'close':
            remote.close()
            break

        else:
            raise NotImplementedError
