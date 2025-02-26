import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from src.policies.pid_controller import PIDController


class PIDPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            mass: float = 0.1,
            dt: float = 0.01,
    ):
        self.mass = mass
        self.dt = dt
        self.pid_controller = PIDController(dt=self.dt,mass=self.mass)

        super().__init__(
            observation_space,
            action_space,
        )

        self.expert_queryed = 0
        self.act_queryed = 0
    def _predict(self, obs, deterministic=True):
        scale = 0.02
        dimensions = 3

        covariance_matrix = scale * np.eye(dimensions)

        mean = np.zeros(dimensions)


        actions = []
        # obs of every vec_env
        for env_obs in obs:
            noise = np.random.multivariate_normal(mean, covariance_matrix)
            actions.append(
                self.pid_controller.get_action(env_obs)
                + noise
            )



        actions = torch.stack(actions, dim=0)
        self.expert_queryed += 1
        # print("##################")
        # print("expert queryed", self.expert_queryed)
        # print("##################")
        return actions

    def act(self, obs, deterministic=False):
        scale = 0.02
        dimensions = 3

        covariance_matrix = scale * np.eye(dimensions)

        mean = np.zeros(dimensions)
        actions = []
        # obs of every vec_env
        noise = np.random.multivariate_normal(mean, covariance_matrix)
        # action = self.pid_controller.get_action(env_obs[0:6], env_obs[6:12])
        # if not isinstance(action, torch.Tensor):
        #     action = torch.from_numpy(action)
        # actions.append(action)
        actions.append(
            self.pid_controller.get_action(obs)
            + noise
        )


        # actions = np.stack(actions, axis=0)
        return self.pid_controller.get_action(obs) + noise

    def reset_controller(self):
        self.pid_controller = PIDController(dt=self.dt,mass=self.mass)



class DbCbsPIDPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            mass: float = 0.1,
            dt: float = 0.01,
    ):
        self.mass = mass
        self.dt = dt
        self.pid_controller = PIDController(dt=self.dt,mass=self.mass)

        super().__init__(
            observation_space,
            action_space,
        )

        self.expert_queryed = 0
        self.act_queryed = 0
    def _predict(self, obs, deterministic=True):
        scale = 0.02
        dimensions = 3
        covariance_matrix = scale * np.eye(dimensions)
        mean = np.zeros(dimensions)

        actions = []
        for env_obs in obs:
            env_actions = []
            for robot_obs in env_obs:
                noise = torch.tensor(
                    np.random.multivariate_normal(mean, covariance_matrix),
                    dtype=torch.float32
                )
                action = self.pid_controller.get_action(robot_obs) + noise
                env_actions.append(action)

            env_actions_tensor = torch.stack(env_actions, dim=0)
            actions.append(env_actions_tensor.flatten())
        actions = torch.stack(actions, dim=0)
        self.expert_queryed += 1
        # print("##################")
        # print("expert queryed", self.expert_queryed)
        # print("##################")
        return actions

class PIDPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            mass: float = 0.1,
            dt: float = 0.01,
    ):
        self.mass = mass
        self.dt = dt
        self.pid_controller = PIDController(dt=self.dt,mass=self.mass)

        super().__init__(
            observation_space,
            action_space,
        )

        self.expert_queryed = 0
        self.act_queryed = 0
    def _predict(self, obs, deterministic=True):
        scale = 0.02
        dimensions = 3

        covariance_matrix = scale * np.eye(dimensions)

        mean = np.zeros(dimensions)


        actions = []
        # obs of every vec_env
        for env_obs in obs:
            noise = np.random.multivariate_normal(mean, covariance_matrix)
            actions.append(
                self.pid_controller.get_action(env_obs)
                + noise
            )



        actions = torch.stack(actions, dim=0)
        self.expert_queryed += 1
        # print("##################")
        # print("expert queryed", self.expert_queryed)
        # print("##################")
        return actions

    def act(self, obs, deterministic=False):
        scale = 0.02
        dimensions = 3

        covariance_matrix = scale * np.eye(dimensions)

        mean = np.zeros(dimensions)
        actions = []
        # obs of every vec_env
        noise = np.random.multivariate_normal(mean, covariance_matrix)
        # action = self.pid_controller.get_action(env_obs[0:6], env_obs[6:12])
        # if not isinstance(action, torch.Tensor):
        #     action = torch.from_numpy(action)
        # actions.append(action)
        actions.append(
            self.pid_controller.get_action(obs)
            + noise
        )


        # actions = np.stack(actions, axis=0)
        return self.pid_controller.get_action(obs) + noise

    def reset_controller(self):
        self.pid_controller = PIDController(dt=self.dt,mass=self.mass)

class FeedForwardPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            mass: float = 0.1,
            dt: float = 0.01,
    ):

        super().__init__(
            observation_space,
            action_space,
        )

        self.expert_queryed = 0
        self.act_queryed = 0
    def _predict(self, obs, deterministic=True):
        scale = 0.02
        dimensions = 4
        covariance_matrix = scale * np.eye(dimensions)

        mean = np.zeros(dimensions)


        actions = []
        # obs of every vec_env
        for env_obs in obs:
            noise = np.random.multivariate_normal(mean, covariance_matrix)
            actions.append(
                env_obs[-4:]
                # + noise
            )

        actions = torch.stack(actions, dim=0)
        self.expert_queryed += 1
        # print("##################")
        # print("expert queryed", self.expert_queryed)
        # print("##################")
        return actions
