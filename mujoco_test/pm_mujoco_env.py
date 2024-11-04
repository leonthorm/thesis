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
