import gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from imitation.algorithms import dagger
from gym.envs.registration import register

import pm_mujoco_env

register(
    id='PointMassMujocoEnv-v0',
    entry_point='pm_mujoco_env:PointMassMujocoEnv',
)

def make_env():
    return gym.make('PointMassMujocoEnv-v0')

num_envs = 4
vec_env = DummyVecEnv([make_env for _ in range(num_envs)])
vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)


