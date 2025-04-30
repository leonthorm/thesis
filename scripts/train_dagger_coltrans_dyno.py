import argparse
import logging
import shutil
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch as th
import wandb
from gymnasium.spaces import Box
from imitation.util.util import make_vec_env
from torch import nn

from deps.imitation.src.imitation.data import rollout, rollout_multi_robot
from deps.imitation.src.imitation.util.util import parse_path
from src.dagger.dagger import dagger_multi_robot, dagger, get_expert
from src.thrifty.algos import core
from src.thrifty.algos.thriftydagger_venv import generate_offline_data, thrifty
from src.thrifty.utils.run_utils import setup_logger_kwargs
# from src.thrifty_new.thrifty import thrifty

# from src.thrifty.utils.run_utils import setup_logger_kwargs
from src.util.helper import calculate_observation_space_size
from src.util.load_traj import load_model

# Set up logging configuration
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


def register_environment(model, model_path, reference_traj_path, num_robots, algorithm, validate=False):
    """
    Registers the custom gym environment for coltrans.

    Parameters:
        model: The model object.
        model_path (str): The path to the model.
        reference_traj_path (Path): The path to the reference trajectory.
        num_robots (int): Number of robots.
    """
    if validate:
        gym.envs.registration.register(
            id='dyno_coltrans-validate',
            entry_point='src.mujoco_envs.dyno_env_coltrans:DynoColtransEnv',
            kwargs={
                'model': model,
                'model_path': model_path,
                'reference_traj_path': str(reference_traj_path),
                'num_robots': num_robots,
                'algorithm': algorithm,
            },
        )
    else:
        gym.envs.registration.register(
            id='dyno_coltrans-v0',
            entry_point='src.mujoco_envs.dyno_env_coltrans:DynoColtransEnv',
            kwargs={
                'model': model,
                'model_path': model_path,
                'reference_traj_path': str(reference_traj_path),
                'num_robots': num_robots,
                'algorithm': algorithm,
            },
        )


def parse_arguments():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Robot Training using DAgger or Thrifty algorithms."
    )
    parser.add_argument(
        "--inp",
        nargs="+",
        type=str,
        help="YAML input reference trajectory paths.",
        default=None,
    )
    parser.add_argument(
        "--inp_dir",
        type=str,
        help="Directory containing input reference trajectory files.",
        default=None,
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment configuration file.",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--out",
        type=str,
        help="YAML output trajectory by policy.",
        default=None,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the model file.",
    )
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Flag to write outputs."
    )
    parser.add_argument("-a", "--compAcc", action="store_true", help="Flag for compAcc.")
    parser.add_argument("-noC", "--nocableTracking", action="store_true", help="Disable cable tracking.")
    parser.add_argument(
        "-alg",
        "--daggerAlgorithm",
        type=str,
        default="dagger",
        help="Algorithm to use: 'dagger' or 'thrifty'.",
    )
    parser.add_argument(
        "--validate",
        action="store_true"
    )

    parser.add_argument("-dc", "--decentralizedPolicy", action="store_true", help="Use decentralized policy.")
    args = parser.parse_args()

    if args.inp is None and args.inp_dir is None:
        parser.error("You must specify either --inp or --inp_dir")
    return args


def main():
    wandb.init(
        project="multi_robot_training",
        config={
            "total_timesteps": 1000,
            "rollout_round_min_episodes": 1,
            "rollout_round_min_timesteps": 400,
            "iters": 2,
            "layer_size": 32,
            "num_layers": 2,
            "activation_fn": "Tanh",
            "bc_episodes": 2,
            "num_nets": 3,
            "grad_steps": 500,
            "pi_lr": 1e-3,
            "bc_epochs": 2,
            "batch_size": 100,
            "obs_per_iter": 700,
            "target_rate": 0.01,
            "num_test_episodes": 1,
            "gamma": 0.9999,
            "retrain_policy": True,
            #ablation study
            "ablation": False,
            "cable_q": True,
            "cable_q_d": True,
            "cable_w": True,
            "cable_w_d": True,
            "robot_rot": True,
            "robot_rot_d": True,
            "robot_w": True,
            "robot_w_d": True,
            "other_cable_q": True,
            "other_robot_rot": True,
            "action_d_single_robot": True,
        },
        settings=wandb.Settings(
            console="off",
        )
    )
    config = wandb.config
    sweep_id = wandb.run._run_id
    args = parse_arguments()
    validate = args.validate
    base_dir = Path(__file__).parent.resolve()

    # Resolve reference trajectory paths
    if args.inp:
        reference_paths = [Path(p) for p in args.inp]
    else:
        reference_paths = list(Path(args.inp_dir).glob("trajectory_*.yaml"))[:32]
        if not reference_paths:
            logger.error("No trajectory files found in the specified directory: %s", args.inp_dir)
            sys.exit(1)

    # Load model and determine number of robots
    model, num_robots = load_model(args.model_path)
    logger.info("Loaded model from %s with %d robots.", args.model_path, num_robots)

    cable_lengths = [0.5] * num_robots
    algorithm = args.daggerAlgorithm.lower()
    decentralized = args.decentralizedPolicy

    # Define training and expert trajectory directories using pathlib

    training_dir = base_dir / ".." / "training" / algorithm / ("decentralized" if decentralized else "centralized")
    demo_dir = (training_dir / "demos").resolve()
    if demo_dir.exists():
        shutil.rmtree(str(demo_dir))
    expert_traj_dir = base_dir / ".." / "trajectories" / "expert_trajectories" / "coltrans_planning"

    # if sweep_id is not None:
    #     training_dir /= sweep_id
    # print("training_dir: ", training_dir)
    # print(sweep_id)
    # Register environment for the first trajectory (will be updated later for each env)
    register_environment(model, args.model_path, reference_paths[0], num_robots, algorithm)

    env_id = "dyno_coltrans-v0"
    rng = np.random.default_rng(0)
    venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=len(reference_paths),
        parallel=False
    )

    # Update each environment with its corresponding reference trajectory
    for idx, path in enumerate(reference_paths):
        # call the `set_reference_traj` method *inside* worker idx
        venv.env_method("set_reference_traj", str(path), indices=[idx])




    # Training parameters
    rollout_round_min_episodes = config.rollout_round_min_episodes
    rollout_round_min_timesteps = config.rollout_round_min_timesteps
    iters = config.iters
    layer_size = config.layer_size
    num_layers = config.num_layers
    net_arch = np.full(num_layers, layer_size).tolist()

    activation_fn = getattr(nn, config.activation_fn)

    # thrifty parameters
    bc_episodes = config.bc_episodes
    num_nets = config.num_nets
    # policy training
    grad_steps = config.grad_steps
    pi_lr = config.pi_lr
    bc_epochs = config.bc_epochs
    batch_size = config.batch_size
    ac_kwargs = dict(hidden_sizes=(layer_size,) * num_layers, activation=activation_fn)
    retrain_policy = config.retrain_policy
    # algorithm
    obs_per_iter = config.obs_per_iter
    target_rate = config.target_rate

    # q learning
    q_learning = True
    num_test_episodes = config.num_test_episodes
    gamma = config.gamma

    #ablation study
    ablation_kwargs = dict()
    if config.ablation:
        ablation_kwargs = dict(
            cable_q=config.cable_q,
            cable_q_d=config.cable_q_d,
            cable_w=config.cable_w,
            cable_w_d=config.cable_w_d,
            robot_rot=config.robot_rot,
            robot_rot_d=config.robot_rot_d,
            robot_w=config.robot_w,
            robot_w_d=config.robot_w_d,
            other_cable_q=config.other_cable_q,
            other_robot_rot=config.other_robot_rot,
        )

    # # thrifty parameters
    # NUM_BC_EPISODES = 7
    # num_nets = 5
    # # policy training
    # grad_steps = 500
    # pi_lr = 1e-3
    # bc_epochs = 5
    # batch_size = 100
    # ac_kwargs = dict(hidden_sizes=(layer_size,) * num_layers, activation=nn.ReLU)
    # # algorithm
    # obs_per_iter = 700
    #
    # # q learning
    # q_learning = True
    # num_test_episodes = 10
    # gamma = 0.9999

    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation_fn
    }
    total_timesteps = -1

    observation_dim = calculate_observation_space_size(num_robots)
    observation_space = Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float64)
    action_space = Box(low=0, high=1.5, shape=(4 * num_robots,), dtype=np.float64)

    if algorithm == 'dagger':
        demo_dir = (training_dir / "demos").resolve()
        if demo_dir.exists():
            shutil.rmtree(str(demo_dir))

        if decentralized:
            trainer = dagger_multi_robot(
                venv=venv,
                iters=iters,
                scratch_dir=str(training_dir),
                device=torch.device('cpu'),
                observation_space=observation_space,
                action_space=action_space,
                rng=rng,
                expert_policy='ColtransPolicy',
                total_timesteps=total_timesteps,
                rollout_round_min_episodes=rollout_round_min_episodes,
                rollout_round_min_timesteps=rollout_round_min_timesteps,
                num_robots=num_robots,
                cable_lengths=cable_lengths,
                **ablation_kwargs
            )
        else:
            trainer = dagger(
                venv=venv,
                iters=iters,
                scratch_dir=str(training_dir),
                device=torch.device('cpu'),
                observation_space=observation_space,
                action_space=action_space,
                rng=rng,
                expert_policy='ColtransPolicy',
                total_timesteps=total_timesteps,
                rollout_round_min_episodes=rollout_round_min_episodes,
                rollout_round_min_timesteps=rollout_round_min_timesteps,
                num_robots=num_robots,
                policy_kwargs=policy_kwargs

            )
        logger.info("Training with centralized DAgger...")
        policy_save_path = trainer.save_trainer()
        policy = trainer.policy
        logger.info("Trainer saved at: %s", policy_save_path)

    elif algorithm == 'thrifty':
        demo_dir = (training_dir / "demos").resolve()
        if demo_dir.exists():
            shutil.rmtree(str(demo_dir))

        if decentralized:
            raise NotImplementedError
            # trainer = thrifty_multi_robot(
            #     venv=venv,
            #     iters=iters,
            #     scratch_dir=str(training_dir_thrifty),
            #     device=torch.device('cpu'),
            #     observation_space=observation_space,
            #     action_space=action_space,
            #     rng=rng,
            #     expert_policy='ColtransPolicy',
            #     total_timesteps=total_timesteps,
            #     rollout_round_min_episodes=rollout_round_min_episodes,
            #     rollout_round_min_timesteps=rollout_round_min_timesteps,
            #     n_robots=num_robots,
            # )
            # logger.info("Training with decentralized Thrifty...")
            # policy_save_path = trainer.save_trainer()
            # logger.info("Trainer saved at: %s", policy_save_path)

        else:
            # trainer = thrifty(
            #     venv=venv,
            #     iters=iters,
            #     scratch_dir=str(training_dir_thrifty),
            #     device=torch.device('cpu'),
            #     observation_space=observation_space,
            #     action_space=action_space,
            #     rng=rng,
            #     expert_policy='ColtransPolicy',
            #     total_timesteps=total_timesteps,
            #     rollout_round_min_episodes=rollout_round_min_episodes,
            #     rollout_round_min_timesteps=rollout_round_min_timesteps,
            #     num_robots=num_robots,
            # )
            # policy_save_path = trainer.save_trainer()
            logger_kwargs = setup_logger_kwargs('ColtransPolicy', rng)
            expert = get_expert(action_space, 'ColtransPolicy', num_robots, observation_space, venv)

            generate_offline_data(venv, expert_policy=expert, action_space=action_space, num_episodes=bc_episodes)
            policy = core.Ensemble

            policy = thrifty(
                venv,
                iters=iters,
                actor_critic=policy,
                ac_kwargs=ac_kwargs,
                grad_steps=grad_steps,
                obs_per_iter=obs_per_iter,
                pi_lr=pi_lr,
                batch_size=batch_size,
                logger_kwargs=logger_kwargs,
                num_test_episodes=num_test_episodes,
                bc_epochs=bc_epochs,
                device_idx=-20,
                expert_policy=expert,
                num_nets=num_nets,
                target_rate=target_rate,
                gamma=gamma,
                input_file='data.pkl',
                q_learning=True,
                retrain_policy=retrain_policy
            )
            logger.info("Training with centralized Thrifty...")
            policy_save_path = training_dir / "thrifty_policy.pt"
            policy_save_path.parent.mkdir(parents=True, exist_ok=True)
            th.save(policy, parse_path(policy_save_path))
            logger.info("Trainer saved at: %s", policy_save_path)
    else:
        logger.error("Invalid algorithm selected: %s", algorithm)
        sys.exit(1)

    # validate
    if validate:
        if sweep_id is not None:
            policy_save_path = base_dir / ".." / "policies" / algorithm / (
                "decentralized" if decentralized else "centralized") / f"{sweep_id}_policy.pt"
            policy_save_path.parent.mkdir(parents=True, exist_ok=True)
            th.save(policy, parse_path(policy_save_path))

        reward, payload_pos_error = validate_policy(algorithm, args, model, num_robots, rng, policy, decentralized, **ablation_kwargs)
        logger.info("Validation reward: %f", reward)
        wandb.log({
            "reward": reward,
            "payload_pos_error": payload_pos_error,
        })
    else:
        # For runs without validation, you might also log final training metrics if available.
        wandb.log({
            "reward": 0,
            "payload_pos_error": 0,
        })

    wandb.finish()


def validate_policy(algorithm, args, model, num_robots, rng, policy, decentralized, **ablation_kwargs):
    validation_dir = parse_path(Path(args.inp_dir) / "test")
    validation_trajs = list(validation_dir.glob("trajectory_*.yaml"))[:32]
    register_environment(model, args.model_path, validation_trajs[0], num_robots, algorithm, validate=True)
    env_id = "dyno_coltrans-validate"
    venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=len(validation_trajs),
        parallel=True
    )

    for idx, path in enumerate(validation_trajs):
        venv.env_method("set_reference_traj", str(path), indices=[idx])


    sample_until = rollout.make_sample_until(
        min_episodes=1,
    )
    deterministic_policy = True
    if algorithm == 'thrifty':
        deterministic_policy = False
    if decentralized:
        trajectories = rollout_multi_robot.generate_trajectories_multi_robot(
            policy=policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=deterministic_policy,
            rng=rng,
            num_robots=num_robots,
            **ablation_kwargs
        )
    else:
        trajectories = rollout.generate_trajectories(
            policy=policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=deterministic_policy,
            rng=rng,
        )
    reward = sum(
        info["reward"]
        for traj in trajectories
        for info in traj.infos
    )

    payload_pos_error = sum(
        info["payload_pos_error"]
        for traj in trajectories
        for info in traj.infos
    )
    # reward = np.sum([traj.rews for traj in trajectories])
    # payload_pos_error = np.sum(np.sum(traj.infos["payload_pos_error"]) for traj in trajectories])
    return reward, payload_pos_error
