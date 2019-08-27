from learning_to_adapt.samplers.base import SampleProcessor
from learning_to_adapt.utils import tensor_utils
import numpy as np


class ModelSampleProcessor(SampleProcessor):
    def __init__(
            self,
            baseline=None,
            discount=0.99,
            gae_lambda=1,
            normalize_adv=False,
            positive_adv=False,
            recurrent=False
    ):

        self.baseline = baseline
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.normalize_adv = normalize_adv
        self.positive_adv = positive_adv
        self.recurrent = recurrent

    def process_samples(self, paths, log=False, log_prefix=''):
        """ Compared with the standard Sampler, ModelBaseSampler.process_samples provides 3 additional data fields
                - observations_dynamics
                - next_observations_dynamics
                - actions_dynamics
            since the dynamics model needs (obs, act, next_obs) for training, observations_dynamics and actions_dynamics
            skip the last step of a path while next_observations_dynamics skips the first step of a path
        """

        assert len(paths) > 0
        recurrent = self.recurrent
        # compute discounted rewards - > returns
        returns = []
        for idx, path in enumerate(paths):
            from IPython.core.debugger import set_trace
            set_trace()
            path["returns"] = tensor_utils.discount_cumsum(path["rewards"], self.discount)
            returns.append(path["returns"])



        # 8) log statistics if desired
        self._log_path_stats(paths, log=log, log_prefix=log_prefix)

        observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][:-1] for path in paths], recurrent)
        next_observations_dynamics = tensor_utils.concat_tensor_list([path["observations"][1:] for path in paths], recurrent)
        actions_dynamics = tensor_utils.concat_tensor_list([path["actions"][:-1] for path in paths], recurrent)
        timesteps_dynamics = tensor_utils.concat_tensor_list([np.arange((len(path["observations"]) - 1)) for path in paths])

        from IPython.core.debugger import set_trace
        set_trace()
        rewards = tensor_utils.concat_tensor_list([path["rewards"][:-1] for path in paths], recurrent)
        returns = tensor_utils.concat_tensor_list([path["returns"] for path in paths], recurrent)

        samples_data = dict(
            observations=observations_dynamics,
            next_observations=next_observations_dynamics,
            actions=actions_dynamics,
            timesteps=timesteps_dynamics,
            rewards=rewards,
            returns=returns,
        )

        return samples_data
    
    def process_samples_es(self, paths, log=False, log_prefix=''):
        assert len(paths)> 0
        recurrent = self.recurrent

        returns = []
        for idx, path in enumerate(paths):
            dones = list(path['dones'])
            start_index = 0
            end_index = len(dones)
            
            pathreturns = []
            while True:
                try:
                    end_index = dones.index(True, start_index)
                    pathreturns.append(tensor_utils.discount_cumsum(path['rewards'][start_index:end_index + 1], self.discount))
                    start_index = end_index + 1
                except ValueError:
                    if start_index < len(dones):
                        pathreturns.append(tensor_utils.discount_cumsum(path['rewards'][start_index:], self.discount))
                    break
            path['returns'] = tensor_utils.concat_tensor_list(pathreturns, False)
            returns.append(path["returns"])
        
        from IPython.core.debugger import set_trace
        set_trace()
        
        list_obs    =   []
        list_nobs   =   []
        list_act    =   []
        list_timstp =   []
        list_rws    =   []
        list_rtrns  =   []

        for path in paths:
            dones = list(path['dones'])
            start_index = 0
            end_index   = len(dones)
            while True:
                try:
                    end_index   = dones.index(True, start_index)
                    list_obs.append(path['observations'][start_index:end_index])
                    list_nobs.append(path['observations'][start_index + 1:end_index + 1])
                    list_act.append(path['actions'][start_index:end_index])
                    list_timstp.append(np.arange(start_index, end_index))
                    list_rws.append(path['rewards'][start_index:end_index])
                    list_rtrns.append(path['returns'][start_index + 1:end_index + 1])
                    
                    start_index = end_index + 1
                except ValueError:
                    if start_index < len(dones):
                        list_obs.append(path['observations'][start_index:-1])
                        list_nobs.append(path['observations'][start_index + 1:])
                        list_act.append(path['actions'][start_index:-1])
                        list_timstp.append(np.arange(start_index,len(dones)-1))
                        list_rws.append(path['rewards'][start_index:-1])
                        list_rtrns.append(path['returns'][start_index + 1:])
                    break

        observations_dynamics = tensor_utils.concat_tensor_list(list_obs, recurrent)
        next_observations_dynamics = tensor_utils.concat_tensor_list(list_nobs, recurrent)
        actions_dynamics = tensor_utils.concat_tensor_list(list_act, recurrent)
        timesteps_dynamics = tensor_utils.concat_tensor_list(list_timstp)
        rewards = tensor_utils.concat_tensor_list(list_rws)
        returns = tensor_utils.concat_tensor_list(list_rtrns)

        samples_data = dict(
            observations=observations_dynamics,
            next_observations=next_observations_dynamics,
            actions=actions_dynamics,
            timesteps=timesteps_dynamics,
            rewards=rewards,
            returns=returns,
        )

        return samples_data


            



        

        

