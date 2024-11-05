import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from pm_mujoco_env import PMMujocoEnv

env =PMMujocoEnv(np.array([1.0, 0.5, 1.0]))
observation, info = env.reset()