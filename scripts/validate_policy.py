import argparse
import os

import gymnasium as gym
import numpy as np
import torch as th

from imitation.algorithms import bc
from imitation.util.util import make_vec_env
from imitation.data import rollout, rollout_multi_robot

from scripts.analysis.visualize_payload import quad3dpayload_meshcatViewer
from src.util.load_traj import load_model, load_coltans_traj


def validate_policy(
    inp: str,
    ref_environment: str,
    out: str,
    model_path: str,
    policy_path: str,
    vis_out: str = None,
    dagger_algorithm: str = "dagger",
    decentralized: bool = False,
    vis: bool = False,
    write: bool = False,
    **ablation_kwargs
):
    """
    Run policy validation on a reference trajectory.

    Args:
        inp: Path to YAML input reference trajectory.
        ref_environment: Reference environment YAML path.
        out: Path to YAML output trajectory by policy.
        model_path: Path to saved model for loading.
        policy_path: Path to the policy file to validate.
        vis_out: Optional path to save HTML visualization.
        dagger_algorithm: 'dagger' or 'thrifty'.
        decentralized: Whether to use decentralized multi-robot rollout.
        vis: Whether to launch the meshcat visualizer.
        write: Flag for writing outputs (if supported).

    Returns:
        A dict with keys: reward, payload_tracking_error,
        avg_payload_tracking_error, avg_reward, traj_part_completed.
    """
    # Load model and trajectories
    model, num_robots = load_model(model_path)
    refresult = load_coltans_traj(inp)
    states_d = np.array(refresult['refstates'])
    print(f'len states_d = {len(states_d)}')

    # Register validation environment
    gym.envs.registration.register(
        id='dyno_coltrans-validate',
        entry_point='src.mujoco_envs.dyno_env_coltrans:DynoColtransEnv',
        kwargs={
            'model': model,
            'model_path': model_path,
            'reference_traj_path': inp,
            'num_robots': num_robots,
            'algorithm': dagger_algorithm,
            'validate': True,
            'validate_out': out,
        },
    )

    # Create vectorized env
    venv = make_vec_env(
        'dyno_coltrans-validate',
        rng=np.random.default_rng(0),
        n_envs=1,
        parallel=False,
    )

    # Sampling criterion
    sample_until = rollout.make_sample_until(min_episodes=1)

    # Load policy
    policy = th.load(policy_path, map_location=th.device('cpu'), weights_only=False).to(th.device('cpu'))
    deterministic_policy = (dagger_algorithm != 'thrifty')

    # Generate trajectories
    if decentralized:

        trajectories = rollout_multi_robot.generate_trajectories_multi_robot(
            policy=policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=deterministic_policy,
            rng=np.random.default_rng(0),
            num_robots=num_robots,
            **ablation_kwargs
        )
    else:
        trajectories = rollout.generate_trajectories(
            policy=policy,
            venv=venv,
            sample_until=sample_until,
            deterministic_policy=deterministic_policy,
            rng=np.random.default_rng(0),
        )

    # Compute metrics
    total_reward = sum(
        info['reward'] for traj in trajectories for info in traj.infos
    ) / num_robots
    total_payload_err = sum(
        info['payload_pos_error'] for traj in trajectories for info in traj.infos
    ) / num_robots

    sum_length = sum(len(traj) for traj in trajectories)
    avg_reward = total_reward / sum_length
    avg_payload_err = total_payload_err / sum_length

    traj_len = len(trajectories[0])
    ref_len = len(states_d) - 1
    completion = traj_len / ref_len if ref_len > 0 else 0.0

    # Print results
    print('######### RESULT METRICS #########')
    print(
        f'reward: {total_reward:.4f}, payload_err: {total_payload_err:.4f}, '  
        f'traj_len: {traj_len}, ref_len: {ref_len}, completion: {completion:.4f}, '  
        f'avg_payload_err: {avg_payload_err:.6f}, avg_reward: {avg_reward:.6f}'
    )

    # Visualization
    if vis and vis_out:
        quad3dpayload_meshcatViewer(ref_environment, out, vis_out, robot='point')

    return {
        'reward': total_reward,
        'payload_tracking_error': total_payload_err,
        'avg_payload_tracking_error': avg_payload_err,
        'avg_reward': avg_reward,
        'completion': completion,
    }


def parse_args():
    parser = argparse.ArgumentParser(description='Validate policy against a reference trajectory.')
    parser.add_argument('-i', '--inp', type=str, required=True, help='YAML input reference trajectory')
    parser.add_argument('-re', '--ref_environment', type=str, required=True, help='Reference environment YAML')
    parser.add_argument('--out', type=str, required=True, help='YAML output trajectory by policy')
    parser.add_argument('-vo', '--vis_out', type=str, help='Visualization output HTML')
    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to saved model')
    parser.add_argument('-p', '--policy_path', type=str, required=True, help='Policy file to validate')
    parser.add_argument('-alg', '--daggerAlgorithm', type=str, default='dagger', choices=['dagger','thrifty'], help='DAGGER variant')
    parser.add_argument('-dc', '--decentralizedPolicy', action='store_true', help='Enable decentralized multi-robot')
    parser.add_argument('-w', '--write', action='store_true', help='Enable writing outputs')
    parser.add_argument('-vis', action='store_true', help='Enable meshcat visualization')
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = validate_policy(
        inp=args.inp,
        ref_environment=args.ref_environment,
        out=args.out,
        model_path=args.model_path,
        policy_path=args.policy_path,
        vis_out=args.vis_out,
        dagger_algorithm=args.daggerAlgorithm,
        decentralized=args.decentralizedPolicy,
        vis=args.vis,
        write=args.write,
    )
    return metrics


if __name__ == '__main__':
    main()
