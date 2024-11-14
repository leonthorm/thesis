import numpy as np
from typing import Dict, Optional, Tuple, Union
from gymnasium.envs.mujoco import MujocoEnv
from numpy.typing import NDArray
from gymnasium.spaces import Box


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
        xml_file: str = "/home/simba/projects/thesis/mujoco_test/dynamics/point_mass.xml",
        frame_skip: int = 1,
        default_camera_config: Dict[str, Union[float, int]] = {},
        healthy_reward: float = 10.0,
        reset_noise_scale: float = 0.0,
        target_state: NDArray[np.float32] = np.array([0.5,0.25,0.5]),
        ** kwargs,
    ):
        observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64)


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

        self.kp = 1000.0
        self.ki = 10.0
        self.kd = 70.0
        self.integral_error = np.zeros(self.model.nu)
        self.previous_error = np.zeros(self.model.nu)
        self.target_state = target_state
        print(target_state)

    def step(self, action):
        current_state = self.data.xpos[self.model.body("point_mass").id]
        print('##################')
        print("state: ", current_state)

        ctrl = self._pid_controller(current_state)
        print("ctrl: ", ctrl)
        self.do_simulation(ctrl, self.frame_skip)

        observation = self._get_obs()

        #terminated = self._check_done()
        #truncated = self._check_truncated()

        if self.render_mode == "human":
            self.render()

        return observation, 0.0, False, False, {}

    def reset_model(self):
        self.set_state(self.init_qpos, self.init_qvel)

        return self._get_obs()


    def _pid_controller(self, current_state):

        error = self.target_state - current_state

        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        self.previous_error = error

        ctrl = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        return ctrl

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()