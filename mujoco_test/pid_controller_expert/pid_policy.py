import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from mujoco_test.pid_controller_expert.pid_controller import PIDController


class PIDPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
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
                self.pid_controller.get_action(env_obs[0:6], env_obs[6:12])
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
                self.pid_controller.get_action(env_obs[0:6], env_obs[6:12])
            )

        actions = np.stack(actions, axis=0)

        return actions


    def goal_trajectory(self, t, radius=1.0, omega=2.0, z_amplitude=1.0, z_freq=0.5):

        x_d = radius * np.cos(omega * t) - radius
        y_d = radius * np.sin(omega * t)
        z_d = z_amplitude * np.sin(z_freq * t)
        vx_d = -radius * omega * np.sin(omega * t)
        vy_d = radius * omega * np.cos(omega * t)
        vz_d = z_amplitude * z_freq * np.cos(z_freq * t)

        return np.array([x_d, y_d, 0, vx_d, vy_d, 0])


