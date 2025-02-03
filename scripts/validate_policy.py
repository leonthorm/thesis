import os

import gymnasium as gym
import numpy as np
import torch

from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from imitation.data import rollout, rollout_multi_robot


def validate_policy(traj_file, algo='dagger'):

    gym.envs.registration.register(
        id='PointMass-validate',
        entry_point='src.mujoco_envs.mujoco_env_pid:PointMassEnv',
        kwargs={
            'dagger': algo,
            'traj_file': traj_file
        },
    )
    env_id = "PointMass-validate"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )


    policy = bc.reconstruct_policy(policy_path=training_dir+"/policy-latest.pt",
                                   device=device,)
    rollout_round_min_timesteps = 200


    sample_until = rollout.make_sample_until(
        min_timesteps=rollout_round_min_timesteps,
        min_episodes=1,
    )

    trajectories = rollout.generate_trajectories(
        policy=policy,
        venv=pm_venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
    )

    print("done")


def validate_policy_multi_robot(traj_file, dynamics, n_robots, algo='dagger'):

    gym.envs.registration.register(
        id='PointMass-validate',
        entry_point='src.mujoco_envs.mujoco_env_2_robot_dbcbs_traj:DbCbsEnv',
        kwargs={
            'dagger': algo,
            'traj_file': traj_file,
            'n_robots': n_robots,
            'xml_file': dynamics,
            # 'render_mode': 'human'
        },
    )
    env_id = "PointMass-validate"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )


    policy = bc.reconstruct_policy(policy_path=training_dir+"/policy-latest.pt",
                                   device=device,)
    rollout_round_min_timesteps = 200


    sample_until = rollout.make_sample_until(
        min_timesteps=rollout_round_min_timesteps,
        min_episodes=1,
    )

    trajectories = rollout_multi_robot.generate_trajectories_multi_robot(
        policy=policy,
        venv=pm_venv,
        sample_until=sample_until,
        deterministic_policy=True,
        rng=rng,
        n_robots=n_robots
    )

    print("done")

if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    dynamics = dirname + "/../src/dynamics/"
    training_dir = dirname + "/../../training/dagger"
    traj_dir = dirname + "/../trajectories/target_trajectories/"


    circle_traj_file = traj_dir + "circle0.csv"
    figure8_traj_file = traj_dir + "figure8_0.csv"
    helix0_traj_file = traj_dir + "helix0.csv"
    lissajous0_traj_file = traj_dir + "lissajous0.csv"
    oscillation_traj_file = traj_dir + "radial_oscillation0.csv"
    wave_traj_file = traj_dir + "wave0.csv"

    # multi-robot_trajs
    two_double_integrator = dynamics + "2_double_integrator.xml"

    swap2_double_integrator_3d = traj_dir + "db_cbs/swap2_double_integrator_3d_opt.yaml"


    rng = np.random.default_rng(0)
    device = torch.device('cpu')
    beta = 0.2

    traj_file = traj_dir + "circle0.csv"


    multi_robot = True
    n_robots = 2
    algo = 'dagger'

    if multi_robot:
        validate_policy_multi_robot(traj_file=swap2_double_integrator_3d, dynamics=two_double_integrator,n_robots=n_robots, algo=algo)
    else:
        validate_policy(traj_file, algo=algo)