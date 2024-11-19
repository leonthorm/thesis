import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from pid_controller import PIDController


class PIDPolicy(BasePolicy):

    def __init__(
            self,
            target_state,
            observation_space: spaces.Space,
            action_space: spaces.Space,
    ):
        self.pid_controller = PIDController(0.01)
        self.target_state = target_state

        super().__init__(
            observation_space,
            action_space,
        )

    def _predict(self, obs, deterministic=False):

        return torch.from_numpy(self.pid_controller.get_action(obs.numpy().flatten(), self.target_state))


