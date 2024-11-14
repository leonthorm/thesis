import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from pm_mujoco_env import PointMassEnv
from inverted_double_pendulum_v5 import InvertedDoublePendulumEnv

gym.envs.registration.register(
    id='PointMass-v0',
    entry_point='pm_mujoco_env:PointMassEnv',
)


env = gym.make("PointMass-v0")

observation = env.reset()

for _ in range(100000):
    env.render()
    action = env.action_space.sample()
    observation, _, terminated, _, _ = env.step(action)
    if terminated:
        observation = env.reset()

env.close()
