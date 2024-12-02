import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
import yaml

class DbCBSPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
    ):


        super().__init__(
            observation_space,
            action_space,
        )

        self.trajectory_path = "trajectories/swap1_double_integrator_3d_opt.yaml"
        with open(self.trajectory_path, 'r') as file:
            self.trajectory  = yaml.safe_load(file)['result'][0]
        self.states = np.array(self.trajectory['states'])
        self.actions = np.array(self.trajectory['actions'])
        self.num_states = self.trajectory['num_states']
        self.mass = 0.1


    def _predict(self, obs, deterministic=False):

        actions = []
        # obs of every vec_env
        for env_obs in obs:
            print(env_obs)
            actions.append(
                torch.from_numpy(
                    self._get_action_from_trajectory(env_obs[0:6])
                )
            )

        actions = torch.stack(actions, dim=0)

        return actions


    def _get_action_from_trajectory(self, state):
        closest_state_idx = self._find_closest_state(state.numpy())

        if closest_state_idx >= self.num_states  - 1:
            control = np.zeros(3)
        else:
            control = self.actions[closest_state_idx] * self.mass

        return control

    def _find_closest_state(self, state):

        differences = self.states - state
        distances = np.sum(differences ** 2, axis=1)

        return np.argmin(distances)
