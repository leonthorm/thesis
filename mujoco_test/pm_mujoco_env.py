import numpy as np
from typing import Dict, Optional, Tuple, Union
from gymnasium.envs.mujoco import MujocoEnv
from numpy.typing import NDArray
from gymnasium.spaces import Box
from pyarrow.lib import table_to_blocks
from werkzeug.exceptions import PreconditionRequired

from pid_controller import PIDController

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 0,
    "distance": 4.1225,
    "lookat": np.array((0.0, 0.0, 0.12250000000000005)),
}


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
        xml_file: str = "/home/simba/projects/thesis/mujoco_test/dynamics/point_mass.xml",
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

        # action = self.pid_controller.get_action(current_state, self.target_state)
        # print("ctrl: ", action)
        position_before = self.data.qpos.copy()
        self.do_simulation(action, self.frame_skip)
        position_after = self.data.qpos.copy()

        observation = self._get_obs()
        reward, reward_info = self._get_rew(position_before, position_after, action)
        distance_to_target = np.linalg.norm(self.target_state[0:3] - position_after)
        # print("action: ", action)
        # print("distance: ", distance_to_target)
        info = {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "z_position": self.data.qpos[2],
            "distance_to_target": distance_to_target,
            **reward_info,
        }
        if self.render_mode == "human":
            self.render()

        done = distance_to_target < 0.05 or distance_to_target > 56
        return observation, reward, done, False, info

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)

        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [self.data.qpos,
             self.data.qvel,
             self.target_state]
        ).ravel()

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def _get_rew(self, position_before, position_after, action):
        distance_before = np.linalg.norm(self.target_state[0:3] - position_before)
        distance_after = np.linalg.norm(self.target_state[0:3] - position_after)

        distance_reward = distance_before - distance_after

        ctrl_cost = self.control_cost(action)

        reward = distance_reward - ctrl_cost

        if distance_after < 0.1:
            reward += 10.0

        reward_info = {
            "distance_reward": distance_reward,
            "reward_ctrl": -ctrl_cost,
        }

        return reward, reward_info

    def get_dt(self):
        return self.dt