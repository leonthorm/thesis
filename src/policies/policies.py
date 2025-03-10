import numpy as np
import torch
from stable_baselines3.common.policies import BasePolicy
from gymnasium import spaces
from src.policies.pid_controller import PIDController
from src.policies.quad3d_payload_controller import Quad3dPayloadController
from src.util.helper import derivative, reconstruct_coltrans_state


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


class ColtransPolicy(BasePolicy):

    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            n_robots: int,
            n_venvs: int,
            dt: float = 0.01,
    ):

        super().__init__(
            observation_space,
            action_space,
        )
        self.dt = dt
        self.n_robots = n_robots
        self.n_venvs = n_venvs
        self.controller = dict()
        self._init_controller()
        self.expert_queryed = 0
        self.tick = 0
        self.act_queryed = 0

    def _predict(self, obs, deterministic=True):
        scale = 0.02
        dimensions = 4
        covariance_matrix = scale * np.eye(dimensions)

        mean = np.zeros(dimensions)

        actions = []
        # obs of every vec_env
        for env_idx, env_obs in enumerate(obs):
            noise = np.random.multivariate_normal(mean, covariance_matrix)
            actions.append(
                self._get_contol(env_idx, env_obs)
                # + noise
            )

        actions = torch.stack(actions, dim=0)
        self.expert_queryed += 1
        # print("##################")
        # print("expert queryed", self.expert_queryed)
        # print("##################")
        return actions

    def _get_contol(self, env_idx, obs, compAcc=False):

        state, state_d, actions_d, acc =  reconstruct_coltrans_state(obs)
        ref_start_idx = 3
        refArray = np.asarray(state_d, dtype=float)
        refArray = np.insert(refArray, ref_start_idx + 3, acc[0])
        refArray = np.insert(refArray, ref_start_idx + 4, acc[1])
        refArray = np.insert(refArray, ref_start_idx + 5, acc[2])
        state_d = refArray.copy()
        u = []
        for r_idx, ctrl in self.controller[env_idx].items():
            r_idx = int(r_idx)
            ui = ctrl.controllerLeePayload(
                actions_d,
                state_d,
                state,
                self.expert_queryed,
                r_idx,
                compAcc,
                acc,
            )
            u.append(np.array(ui) * (0.0356 * 9.81 / 4.))
            # u.append(np.array(actions_d) * (0.0356 * 9.81 / 4.))
            self.controller[env_idx][str(r_idx)] = ctrl
        u = np.stack(u)
        return torch.tensor(u)

    def _init_controller(self):
        gains = [
            (12, 10, 0),
            (14, 12, 0),
            (0.03, 0.0012, 0.0),
            (100, 100, 100),
            (1500),
        ]
        # todo: read from config file (test_quad3dpayload_n.py)
        params = {
            "mi": 0.0356,
            "mp": 0.01,
            "Ji": [16.571710e-6, 16.655602e-6, 29.261652e-6],
            "num_robots": self.n_robots,
            "l": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
            "payloadType": "point",
            "nocableTracking": True,
            "robot_radius": 0.1,
        }
        # self.controller = Quad3dPayloadController(params, gains)
        for env in range(self.n_venvs):
            self.controller[env] = dict()
            for i in range(self.n_robots):
                self.controller[env][str(i)] = Quad3dPayloadController(params, gains)


