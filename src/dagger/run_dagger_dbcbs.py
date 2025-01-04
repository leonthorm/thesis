import logging
import os
import tempfile
import shutil

import gymnasium as gym
import numpy as np
import torch
from pygame.draw import circle
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.policies import DbCbsPIDPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer, DAggerTrainer
from imitation.util.util import make_vec_env
from imitation.data import rollout, serialize, types
from dagger import dagger

dirname = os.path.dirname(__file__)
training_dir = dirname + "/../../training/dagger"
traj_dir = dirname + "/../../trajectories/target_trajectories/"

dynamics = dirname + "/../dynamics/"
two_double_integrator = dynamics + "2_double_integrator.xml"
swap1_double_integrator_3d = traj_dir + "db_cbs/swap1_double_integrator_3d_opt.yaml"
swap2_double_integrator_3d = traj_dir + "db_cbs/swap2_double_integrator_3d_opt.yaml"

rng = np.random.default_rng(0)
device = torch.device('cpu')

# logging.getLogger().setLevel(logging.INFO)

#target_state = np.concatenate([np.random.uniform(0, 0.5, 3), [0.0, 0.0, 0.0]]).flatten()
# target_state = np.array([0.5,0.25,0.5, 0, 0, 0])

gym.envs.registration.register(
    id='DbCbsEnv-v0',
    entry_point='mujoco_env_2_robot_dbcbs_traj:DbCbsEnv',
    kwargs={
        'dagger': 'dagger',
        'traj_file': swap2_double_integrator_3d,
        'n_robots': 2,
        'xml_file': two_double_integrator,
        # 'render_mode': 'human'
        'render_mode': 'rgb_array',
    },
)

beta = 0.2


if __name__ == '__main__':

    demo_dir = os.path.abspath(training_dir + "/demos")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)


    env_id = "DbCbsEnv-v0"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )

    trajs = [swap2_double_integrator_3d,
             ]

    for idx, env in enumerate(pm_venv.envs):
        attr = env.get_wrapper_attr('set_traj')
        attr(trajs[idx])

    bc_trainer = bc.BC(
        observation_space=pm_venv.observation_space,
        action_space=pm_venv.action_space,
        rng=rng,
        device=device,
    )
    total_timesteps = 4_000
    rollout_round_min_episodes = 3
    rollout_round_min_timesteps = 200

    dagger_trainer = dagger(venv=pm_venv,
                            iters=10,
                            scratch_dir=training_dir,
                            device=device,
                            observation_space=pm_venv.observation_space,
                            action_space=pm_venv.action_space,
                            rng=rng, expert_policy='DbCbsPIDPolicy', total_timesteps=total_timesteps, rollout_round_min_episodes=rollout_round_min_episodes,
                            rollout_round_min_timesteps=rollout_round_min_timesteps)

    reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)

    print(dagger_trainer.save_trainer())
    # print(bc_trainer.save)

    shutil.rmtree(training_dir + "/demos")
    print("Reward:", reward)
