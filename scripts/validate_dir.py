#!/usr/bin/env python3
"""
batch_validate.py

Scan a directory for trajectory_*.yaml files, find their matching env_*.yaml files,
and run validate_policy on each pair, writing results and visualizations without console spam,
then print average metrics over all runs.
"""
import random
from pathlib import Path
import io
import contextlib

import numpy as np
import torch

# Import the validation function directly
from validate_policy import validate_policy

# --- CONFIG ---
INPUT_DIR = Path("training_data/test")
MODEL_FILE = Path("deps/dynobench/models/point_2.yaml")
POLICY_FILE = Path("analysis_policies/thrifty_norr.pt")
ALG = "thrifty"
OUTPUT_DIR = Path("results")
VIS_DIR = OUTPUT_DIR / "visualization"
VIS = True
DECENTRALIZED = True  # match original --dc flag
ablation_kwargs = dict(
    cable_q=True,
    cable_q_d=True,
    cable_w=True,
    cable_w_d=True,
    robot_rot=True,
    robot_rot_d=True,
    robot_w=True,
    robot_w_d=True,
    other_cable_q=True,
    other_robot_rot=False,
    payload_pos_e=True,
    payload_vel_e=True,
    action_d_single_robot=True
)


# ----------------

def main():
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    rng = np.random.default_rng(seed)


    all_metrics = []
    for traj_path in sorted(INPUT_DIR.glob("trajectory_*.yaml")):
        # Derive matching environment file name
        suffix = traj_path.name[len("trajectory_"):]
        env_path = INPUT_DIR / f"env_{suffix}"
        if not env_path.exists():
            print(f"[SKIP] no matching env file for {traj_path.name}")
            continue

        # Derive output file names
        base = suffix.rsplit(".yaml", 1)[0]
        result_path = OUTPUT_DIR / f"result_{base}.yaml"
        vis_path = VIS_DIR / f"result_vis_{base}.html"

        print(f"[RUN] validating {traj_path.name} with {env_path.name}")
        try:
            # Suppress validate_policy console output
            metrics = validate_policy(
                inp=str(traj_path),
                ref_environment=str(env_path),
                out=str(result_path),
                model_path=str(MODEL_FILE),
                policy_path=str(POLICY_FILE),
                vis_out=str(vis_path),
                dagger_algorithm=ALG,
                decentralized=DECENTRALIZED,
                vis=VIS,
                write=False,
                rng=rng,
                **ablation_kwargs
            )
            print(f"[DONE] {traj_path.name}: {metrics}")
            all_metrics.append(metrics)
        except Exception as e:
            print(f"[ERROR] validation failed on {traj_path.name}: {e}")

    # Compute and print average metrics if any runs succeeded
    if all_metrics:
        avg_metrics = {}
        count = len(all_metrics)
        # assume all dicts have the same keys
        keys = all_metrics[0].keys()
        for key in keys:
            if key == 'avg_payload_tracking_error':
                payload_errs = [m.get('avg_payload_tracking_error', 0.0) for m in all_metrics]

                avg_metrics['avg_payload_tracking_error'] = np.mean(payload_errs)
                avg_metrics['std_payload_tracking_error'] = np.std(payload_errs)
            else:
                total = sum(m.get(key, 0.0) for m in all_metrics)
                avg_metrics[key] = total / count

        print("\n=== AVERAGE METRICS OVER ALL RUNS ===")
        for key, value in avg_metrics.items():
            print(f"{key}: {value}")

        completion_count = sum(
            1
            for m in all_metrics
            if m.get('completion', 0.0) == 1.0
        )
        print(f"\n=== COMPLETION SUMMARY ===\nCompleted runs: {completion_count}/{len(all_metrics)}")
        print(f"policy: {POLICY_FILE}")

    else:
        print("No metrics to average.")


if __name__ == "__main__":
    main()
