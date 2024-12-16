import tempfile
import wandb

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy

from src.dagger.pid_policy import PIDPolicy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env

rng = np.random.default_rng(0)
device = torch.device('cpu')
beta = 0.2

traj_file = "trajectories/target_trajectories/circle0.csv"

gym.envs.registration.register(
    id='PointMass-v2',
    entry_point='src.dagger.mujoco_env_pid:PointMassEnv',
    kwargs={
        'dagger': 'dagger',
        'traj_file': traj_file
    },
)

beta = 0.2

if __name__ == '__main__':

    # Initialize WandB
    wandb.init(
        project="dagger-training",  # Replace with your WandB project name
        name="pointmass_dagger_run",  # Custom name for the run
        config={
            "env_id": "PointMass-v2",
            "algorithm": "DAgger",
            "beta": beta,
        }
    )

    env_id = "PointMass-v2"
    pm_venv = make_vec_env(
        env_id,
        rng=rng,
        n_envs=1,
        parallel=False
    )

    expert = PIDPolicy(
        observation_space=pm_venv.observation_space,
        action_space=pm_venv.action_space,
    )

    bc_trainer = bc.BC(
        observation_space=pm_venv.observation_space,
        action_space=pm_venv.action_space,
        rng=rng,
        device=device,
    )

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        dagger_trainer = SimpleDAggerTrainer(
            venv=pm_venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )

        # Train and log progress
        training_steps = 4000
        for step in range(1, training_steps + 1):
            dagger_trainer.train(1)  # Train for 1 step at a time
            if step % 100 == 0:  # Log every 100 steps
                reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
                wandb.log({
                    "step": step,
                    "reward": reward
                })
                print(f"Step: {step}, Reward: {reward}")

    # Final Evaluation
    reward, _ = evaluate_policy(dagger_trainer.policy, pm_venv, 10)
    wandb.log({"final_reward": reward})
    print("Final Reward:", reward)

    # Save the trained policy
    training_dir = "../../mujoco_test/pid_controller_expert/training"
    policy_path = training_dir + "/policy.pth"
    dagger_trainer.policy.save(policy_path)

    # Log the policy file to WandB
    wandb.save(policy_path)

    wandb.finish()
