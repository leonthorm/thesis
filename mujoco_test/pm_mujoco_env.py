import numpy as np
from typing import Dict, Optional, Tuple, Union
from gymnasium.envs.mujoco import MujocoEnv
from numpy.typing import NDArray
from gymnasium import spaces

class PMMujocoEnv(MujocoEnv):
    def __init__(self, target_state):
        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        super().__init__(
            model_path="/home/simba/projects/thesis/mujoco_test/dynamics/point_mass.xml",
            frame_skip=5,
            observation_space=observation_space,
            render_mode="human"
        )
        self.kp = 30.0
        self.ki = 2.2
        self.kd = 2.5
        self.integral_error = np.zeros(self.model.nu)
        self.previous_error = np.zeros(self.model.nu)
        self.target_state = target_state

    def step(self, action: NDArray[np.float32]) -> Tuple[NDArray[np.float64], np.float64, bool, bool, Dict[str, np.float64]]:
        current_state = self.data.qpos[:self.model.nq]

        ctrl = self._pid_controller(current_state)
        self.do_simulation(ctrl, self.frame_skip)

        observation = self._get_obs()
        reward = 0.0
        terminated = self._check_done()
        truncated = self._check_truncated()

        return observation, reward, terminated, truncated, {}

    def reset_model(self) -> NDArray[np.float64]:
        qpos = self.init_qpos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)

        return self._get_observation()

    def _get_reset_info(self) -> Dict[str, float]:
        """Function that generates the `info` that is returned during a `reset()`."""
        return {}

    def _pid_controller(self, current_state: NDArray[np.float64]) -> NDArray[np.float64]:

        error = self.target_position - current_state

        self.integral_error += error * self.dt
        derivative_error = (error - self.previous_error) / self.dt
        self.previous_error = error

        ctrl = self.kp * error + self.ki * self.integral_error + self.kd * derivative_error

        self.previous_error = error

        return ctrl


import gym
from gym import spaces
import numpy as np
import mujoco
from gym.envs.registration import register


class PointMassMujocoEnv(gym.Env):
    def __init__(self):
        super(PointMassMujocoEnv, self).__init__()

        # Load your Mujoco model
        model_path = "/mujoco_test/dynamics/point_mass.xml"  # Update this path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Define the observation and action spaces
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        action_dim = self.model.nu  # Number of controls (actuators)
        self.action_space = spaces.Box(
            low=-10.0, high=10.0, shape=(action_dim,), dtype=np.float32
        )

    def reset(self):
        # Reset the simulation and return the initial observation
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        # Clip the action to ensure it's within valid bounds
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Apply the action and step the simulation
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        # Compute reward and check if the episode is done
        done = self._check_done()

        return self._get_obs(), done, {}

    def _get_obs(self):
        # Construct the observation from qpos and qvel
        return np.concatenate([self.data.qpos, self.data.qvel]).ravel()


    def _check_done(self):
        # Define your termination condition
        return False  # Placeholder, modify as needed

    def render(self, mode='human'):
        # Create a simple viewer if one doesn't exist yet
        if not hasattr(self, 'viewer'):
            self.viewer = mujoco.MjViewer(self.model, self.data)
        self.viewer.render()
