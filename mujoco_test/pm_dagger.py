import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np

from pm_mujoco_env import PointMassEnv
from inverted_double_pendulum_v5 import InvertedDoublePendulumEnv

gym.envs.registration.register(
    id='PointMass-v0',
    entry_point='pm_mujoco_env:PointMassEnv',
)

# Replace with the appropriate MuJoCo environment
env = gym.make("PointMass-v0")

# Correct method to reset the environment
observation = env.reset()

# Example loop to run the simulation
for _ in range(100000):
    env.render()  # Renders the simulation
    action = env.action_space.sample()  # Take random actions
    observation, _, terminated, _, _ = env.step(action)
    if terminated:
        observation = env.reset()  # Reset if the episode is done

env.close()
