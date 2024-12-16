import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from src.dagger.pid_controller import PIDController
class PIDPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space
    ):
        self.pid_controller = PIDController(0.01)

        super().__init__(
            observation_space,
            action_space,
        )

    def _predict(self, obs, deterministic=False):
        # print("query expert")
        actions = []
        # obs of every vec_env
        for env_obs in obs:
            actions.append(
                self.pid_controller.get_action(env_obs[0:6])
            )

        actions = torch.stack(actions, dim=0)

        return actions

    def act(self, obs, deterministic=False):
        # print("query expert")
        actions = []
        # obs of every vec_env
        for env_obs in obs:
            # action = self.pid_controller.get_action(env_obs[0:6], env_obs[6:12])
            # if not isinstance(action, torch.Tensor):
            #     action = torch.from_numpy(action)
            # actions.append(action)
            actions.append(
                self.pid_controller.get_action(env_obs[0:6])
            )

        actions = np.stack(actions, axis=0)

        return actions


