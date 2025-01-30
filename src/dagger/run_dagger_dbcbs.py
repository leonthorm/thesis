import logging
import os
import tempfile
import shutil

import gymnasium as gym
import numpy as np
import torch
from gymnasium.spaces import Box
from pygame.draw import circle
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.policies import DbCbsPIDPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer, DAggerTrainer
from imitation.util.util import make_vec_env
from imitation.data import rollout, serialize, types
from dagger import dagger, dagger_multi_robot
from thrifty import thrifty

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



beta = 0.2


if __name__ == '__main__':

    n_robots = 2
    observation_space_size = 9 + (n_robots-1) * 3
    actions_space_size = 3
    n_envs = 3

    gym.envs.registration.register(
        id='DbCbsEnv-v0',
        entry_point='mujoco_env_2_robot_dbcbs_traj:DbCbsEnv',
        kwargs={
            'dagger': 'dagger',
            'traj_file': swap2_double_integrator_3d,
            'n_robots': n_robots,
            'xml_file': two_double_integrator,
            # 'render_mode': 'human'
            'render_mode': 'rgb_array',
        },
    )
    demo_dir = os.path.abspath(training_dir + "/demos")

    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)

    dagger_algo = True
    thrifty_algo = False

    env_id = "DbCbsEnv-v0"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=n_envs,
        parallel=False
    )

    trajs = [swap2_double_integrator_3d,swap2_double_integrator_3d,swap2_double_integrator_3d,swap2_double_integrator_3d,
             ]

    for idx, env in enumerate(pm_venv.envs):
        attr = env.get_wrapper_attr('set_traj')
        attr(trajs[idx])

    total_timesteps = 4_000
    rollout_round_min_episodes = 3
    rollout_round_min_timesteps = 200

    # todo: only 3 quaternions for payload?
    # x[i]: {position: [x,y,z], velocity: [vx,vy,vz], desired acc}
    x_desc = {"xp [m]", "yp [m]", "zp [m]", "qcx []",
              "qcy []", "qcz[]", "vpx [m/s]", "vpy [m/s]",
              "vpz [m/s]", "wcx [rad/s]", "wcy [rad/s]", "wcz [rad/s]",
              "qx []", "qy []", "qz []", "qw []",
              "wx [rad/s]", "wy [rad/s]", "wz [rad/s]"}


    # todo: other robot rotation or only position?
    other_robot_observation_len = 3 * (n_robots - 1)
    u_desc = {"f1 []", "f2 [], f3 [], f4 []"}
    print()
    # observation_space_single_quadrotor = Box(low=-np.inf, high=np.inf,
    #                                          shape=(n_robots, len(x_desc) + other_robot_observation_len,), dtype=np.float64)
    # action_space = Box(low=-5.0, high=5.0, shape=(len(u_desc),), dtype=np.float64)

    observation_space = Box(low=-np.inf, high=np.inf,
                            shape=(observation_space_size,), dtype=np.float64)
    action_space = Box(low=-10.0, high=10.0, shape=(actions_space_size,), dtype=np.float64)

    if dagger_algo:
        dagger_trainer = dagger_multi_robot(venv=pm_venv,
                                            iters=5,
                                            scratch_dir=training_dir,
                                            device=device,
                                            observation_space=observation_space,
                                            action_space=action_space,
                                            rng=rng, expert_policy='PIDPolicy', total_timesteps=total_timesteps, rollout_round_min_episodes=rollout_round_min_episodes,
                                            rollout_round_min_timesteps=rollout_round_min_timesteps, n_robots=n_robots, )
        #todo reward
        # reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
        print(dagger_trainer.save_trainer())

    if thrifty_algo:
        thrifty_trainer = thrifty(venv=pm_venv,
                                  iters=20,
                                  scratch_dir=training_dir,
                                  device=device,
                                  observation_space=observation_space,
                                  action_space=action_space,
                                  rng=rng, expert_policy='PIDPolicy', total_timesteps=total_timesteps,
                                  rollout_round_min_episodes=rollout_round_min_episodes,
                                  rollout_round_min_timesteps=rollout_round_min_timesteps)

        reward, _ = evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
        print(thrifty_trainer.save_trainer())

    if thrifty_algo and dagger_algo:
        evaluate_policy(thrifty_trainer.policy, pm_venv, 10)
        evaluate_policy(dagger_trainer.policy, pm_venv, 10)
    # print(bc_trainer.save)

    shutil.rmtree(training_dir + "/demos")
    print("Reward:", reward)
