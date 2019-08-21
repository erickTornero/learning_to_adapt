import numpy as np
from learning_to_adapt.envs_vrep.wrapper_quad.wrapper_vrep import VREPQuad
from learning_to_adapt.utils.serializable import Serializable
from learning_to_adapt.logger import logger
import os


class QuadrotorVrepEnv(VREPQuad, Serializable):
    def __init__(self, task='rotorless', reset_every_episode=False, port=19999):
        Serializable.quick_init(self, locals())
        ## missing line: task mask?
        self.reset_every_episode    =   reset_every_episode
        self.first  = True
        VREPQuad.__init__(self, ip='127.0.0.1', port=port)

        task = None if task == 'None' else task
        
        # Allow to disable a rotor
        self.task_mask = np.ones(self.action_space.shape)

        assert task in [None, 'rotorless']

        self.task = task
        self.actiondim = self.action_space.shape[0]
        # Rotor 0 always OK

    def step(self, action):
        action  =   self.task_mask * action
        print('in-step-', self.clientID)
        # Call the super method step
        ob, rw, done, info = super().step(action)
        # TODO: Force Done to be True??, check!
        done = False
        return ob, rw, done, info


    def reward(self, obs, action, next_obs):
        """
        Define the reward function just from states
        for Model Predictive Control
        """
        assert obs.ndim     == 2
        assert obs.shape    == next_obs.shape
        assert obs.shape[0] == next_obs.shape[0]

        # TODO: Check how to define the reward function
        # define it from states
        targetpos   =   self.targetpos
        currpos     =   obs[9:12]

        distance    =   targetpos - currpos
        distance    =   np.sqrt(distance * distance)

        reward      =   4.0 - 1.25 * distance

        return reward   



    def reset_task(self, value=None):
        if self.task == 'rotorless':
            failedrotor = value if value is not None else np.random.randint(0, self.actiondim)
            # if failedrotor == 0 rotor any rotor fails
            self.task_mask = np.ones(self.action_space.shape)
            if failedrotor > 0:
                self.task_mask[failedrotor] = 0
        elif self.task is None:
            pass
        else:
            raise NotImplementedError
    
    # TODO: Check! Bellow
    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))


if __name__ == '__main__':
    env = QuadrotorVrepEnv(task='rotorless')
    while True:
        env.reset()
        env.reset_task()
        for _ in range(1000):
            env.step(env.action_space.sample())
            #env.render()