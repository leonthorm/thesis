import os

import numpy as np
from typing import Dict, Optional, Tuple, Union
from gymnasium.envs.mujoco import MujocoEnv
from numpy.typing import NDArray
from gymnasium.spaces import Box
from pyarrow.lib import table_to_blocks
from torch.fx.experimental.migrate_gradual_types.constraint import is_dim
from werkzeug.exceptions import PreconditionRequired

from pid_controller import PIDController

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}
path = os.getcwd()

class PointMassEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        # target_state: NDArray[np.float32],
        xml_file: str = path+"/dynamics/point_mass.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = {},
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.0,
        ctrl_cost_weight: float = 0.1,

        ** kwargs,
    ):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=observation_space,
            render_mode="human",
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }
        self.target_state = np.concatenate(
            [np.random.uniform(-0.5, 0.5, 3),
            [0.0, 0.0, 0.0]], axis=0
        )
        print("target_state", self.target_state)
        self.model.site_pos[self.model.site_bodyid[0]] = self.target_state[0:3]
        self.pid_controller = PIDController(self.dt)
        self._ctrl_cost_weight = ctrl_cost_weight
        self.steps=0

    def step(self, action):

        position_before = self.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)
        position_after = self.data.qpos.copy()

        distance_to_target = np.linalg.norm(self.target_state[0:3] - position_after)

        reward, reward_info = self._get_rew(position_before, position_after, action)

        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "distance_to_target": distance_to_target,
            **reward_info,
        }
        if self.render_mode == "human":
            self.render()

        return (
            self._get_obs(),
            reward,
            self._is_done(distance_to_target),
            False,
            info
        )

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos,
             self.data.qvel,
             self.target_state]
        ).ravel()

    def _is_done(self, distance_to_target):

        return distance_to_target < 0.05 or distance_to_target > 5

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_rew(self, position_before, position_after, action):

        de_weight = 10.0
        ve_weight = 0.001
        distance_before = np.linalg.norm(self.target_state[0:3] - position_before)
        distance_after = np.linalg.norm(self.target_state[0:3] - position_after)

        distance_reward = distance_before - distance_after

        velocity_current = self.data.qvel.copy()
        velocity_error = -np.linalg.norm(velocity_current)

        ctrl_cost = self.control_cost(action)

        reward = distance_reward*de_weight - ctrl_cost + velocity_error * ve_weight

        if distance_after < 0.1:
            reward += 10.0

        reward_info = {
            "distance_reward": distance_reward,
            "control_cost": -ctrl_cost,
            "velocity_error": velocity_error,
        }
        print(reward)
        print(reward_info)

        return reward, reward_info

    def get_dt(self):
        return self.dt