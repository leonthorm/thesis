import tempfile

import gymnasium as gym
import numpy as np
import torch
import logging
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3 import PPO
from pid_policy import PIDPolicy
from pm_mujoco_env import PointMassEnv

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env


rng = np.random.default_rng(0)
device = torch.device('cpu')
logging.getLogger().setLevel(logging.INFO)

target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])

gym.envs.registration.register(
    id='PointMass-v0',
    entry_point='pm_mujoco_env:PointMassEnv',
    # kwargs={'target_state': target_state},
)
if __name__ == '__main__':
    env_id = "PointMass-v0"
    env = make_vec_env(
        env_id,
        rng=rng,
        n_envs=4,
        parallel=True
    )

    expert = PIDPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space
    )



    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        device=device,
    )


    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(2_000)

    reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
    print("Reward:", reward)



# print(env.observation_space)
#
# observation = env.reset()
#
# for _ in range(100000):
#     env.render()
#     action = env.action_space.sample()
#     observation, reward, done, information = env.step(action)
#     if done:
#         observation = env.reset()
#
# env.close()
