import yaml
import numpy as np
from pathlib import Path

# ─── CONFIG ────────────────────────────────────────────────────────────────
# Change this to wherever your trajectory_*.yaml files are located:
data_dir = Path("training_data/2robot/validation")
dt = 0.01  # timestep between rows, in seconds
# ────────────────────────────────────────────────────────────────────────────

energy_list = []       # one energy_per_file (Wh) per YAML
tracking_errors = []   # collect all per‐timestep Euclidean distances

for yaml_path in sorted(data_dir.glob("trajectory_*.yaml")):
    try:
        with open(yaml_path, "r") as f:
            info = yaml.load(f, Loader=yaml.CSafeLoader)
    except Exception as e:
        print(f"Warning: could not read {yaml_path.name}: {e}")
        energy_list.append(np.nan)
        continue

    # Make sure the "result" dict exists
    if "result" not in info or not isinstance(info["result"], dict):
        print(f"Warning: no 'result' section in {yaml_path.name}; skipping.")
        energy_list.append(np.nan)
        continue

    res = info["result"]

    # ─── ENERGY CALCULATION ────────────────────────────────────────────────
    if "actions" in res:
        actions_arr = np.array(res["actions"], dtype=float)  # shape = (T, N_actuators)
        # 1) sum over each row to get normalized_force[t]
        normalized_force = np.sum(actions_arr, axis=1)  # (T,)
        # 2) convert to grams: force_g = normalized_force * 34/4
        force_g = normalized_force * 34.0 / 4.0
        # 3) convert to watts: power_w = force_g / 4
        power_w = force_g / 4.0
        # 4) integrate: energy_Wh = sum(power_w)*dt / 3600
        energy_wh = np.sum(power_w) * dt / 3600.0
        energy_list.append(energy_wh)
    else:
        # If there is no "actions" key under result, record NaN
        print(f"Warning: 'actions' missing in result of {yaml_path.name}.")
        energy_list.append(np.nan)

    # ─── TRACKING‐ERROR CALCULATION ───────────────────────────────────────
    # We need both "refstates" and "states" under result, each with ≥3 columns
    if ("refstates" in res) and ("states" in res):
        ref_arr = np.array(res["refstates"], dtype=float)
        state_arr = np.array(res["states"], dtype=float)

        # Check they have the same number of timesteps and ≥3 columns
        if ref_arr.ndim == 2 and state_arr.ndim == 2 \
           and ref_arr.shape[0] == state_arr.shape[0] \
           and ref_arr.shape[1] >= 3 and state_arr.shape[1] >= 3:

            # compute ‖ref[:,0:3] - states[:,0:3]‖₂ per timestep
            diffs = ref_arr[:, 0:3] - state_arr[:, 0:3]  # shape = (T, 3)
            errs = np.linalg.norm(diffs, axis=1)         # shape = (T,)
            tracking_errors.extend(errs.tolist())
        else:
            print(f"Warning: shape mismatch in refstates/states of {yaml_path.name}; skipping tracking error.")
    else:
        # Either or both keys are missing; skip tracking-error for this file
        print(f"Warning: 'refstates' or 'states' missing in result of {yaml_path.name}.")

# ─── FINAL AGGREGATION ─────────────────────────────────────────────────────
energy_arr = np.array(energy_list, dtype=float)
tracking_arr = np.array(tracking_errors, dtype=float)

energy_mean = np.nanmean(energy_arr)
energy_std  = np.nanstd(energy_arr)

if tracking_arr.size > 0:
    tracking_error_mean = np.nanmean(tracking_arr)
    tracking_error_std  = np.nanstd(tracking_arr)
else:
    tracking_error_mean = np.nan
    tracking_error_std  = np.nan

print(f"Processed {len(energy_arr)} files matching 'trajectory_*.yaml'.")
print(f"  → energy_mean         = {energy_mean:.6f} Wh")
print(f"  → energy_std          = {energy_std:.6f} Wh")
print(f"  → tracking_error_mean = {tracking_error_mean:.6f}")
print(f"  → tracking_error_std  = {tracking_error_std:.6f}")
