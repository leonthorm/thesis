#!/usr/bin/env python3
"""
batch_validate.py

Scan a directory for trajectory_*.yaml files, find their matching env_*.yaml files,
and run validate-policy on each pair, writing results and visualizations.
"""

import subprocess
from pathlib import Path

# --- CONFIG ---
INPUT_DIR        = Path("training_data/validation")
MODEL_FILE       = Path("deps/dynobench/models/point_2.yaml")
POLICY_FILE      = Path("policies/thrifty/centralized/12aej4rc_policy.pt")
ALG              = "thrifty"
OUTPUT_DIR       = Path("results")
VIS_DIR          = OUTPUT_DIR / "visualization"
VALIDATE_BINARY  = "validate-policy"  # or full path to the script if not on $PATH
# ----------------

def main():
    # make sure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    for traj_path in sorted(INPUT_DIR.glob("trajectory_*.yaml")):
        # derive the matching env file name
        suffix = traj_path.name[len("trajectory_"):]   # e.g. "1742905679_random_2robots.yaml"
        env_path = INPUT_DIR / f"env_{suffix}"
        if not env_path.exists():
            print(f"[SKIP] no matching env file for {traj_path.name}")
            continue

        # derive output file names
        base = suffix.rsplit(".yaml", 1)[0]  # e.g. "1742905679_random_2robots"
        result_path = OUTPUT_DIR / f"result_{base}.yaml"
        vis_path    = VIS_DIR    / f"result_vis_{base}.html"

        # build the command
        cmd = [
            VALIDATE_BINARY,
            "-i", str(traj_path),
            "-re", str(env_path),
            "-m", str(MODEL_FILE),
            "-p", str(POLICY_FILE),
            "-alg", ALG,
            "--out", str(result_path),
            "-vo", str(vis_path),
            # "-dc",
        ]

        # run it
        print(f"[RUN] {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] validate-policy failed on {traj_path.name}: {e}")

if __name__ == "__main__":
    main()
