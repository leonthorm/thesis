import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from pid_controller import PIDController
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

        self.trajectory_path = "db_cbs_trajectories/swap1_double_integrator_3d_opt.yaml"
        with open(self.trajectory_path, 'r') as file:
            self.trajectory  = yaml.safe_load(file)



    def _predict(self, obs, deterministic=False):

        actions = []
        # obs of every vec_env
        for env_obs in obs:
            actions.append(
                self.pid_controller.get_action(env_obs[0:6], env_obs[6:12])
            )

        actions = torch.stack(actions, dim=0)

        return actions


