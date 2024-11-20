import tempfile
import torch
import numpy as np
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env

# Set the device
device = torch.device("cpu")

# Random number generator
rng = np.random.default_rng(0)

if __name__ == '__main__':
    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=rng,
    )

    # Load the expert policy and move it to the appropriate device
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    expert = expert.to(device)

    # Initialize BC trainer
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
        device=device  # Ensure BC trainer uses the correct device
    )

    # Temporary directory for DAgger training
    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        print(tmpdir)
        # Initialize DAgger trainer
        dagger_trainer = SimpleDAggerTrainer(
            venv=env,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        # Train the DAgger model
        dagger_trainer.train(8_000)

    # Evaluate the trained policy
    reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
    print("Reward:", reward)
