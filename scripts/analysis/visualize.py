import argparse
import subprocess


def run_visualizer(filename_env, reference_traj, filename_result, filename_output):
    try:
        subprocess.run(["python3",
                        "../scripts/visualize_payload.py",
                        "--env", str(filename_env),
                        "--robot", "point",
                        "--ref", reference_traj,
                        "--result", str(filename_result),
                        "--output", str(filename_output)],
                       check=True)
    except Exception as e:
        print(e)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out",
        default=None,
        type=str,
        help="yaml output trajectory by policy",
        required=False,
    )
    result_folder = results_path / task.instance / "opt" / "{:03d}".format(task.trial)
    run_visualizer(result_folder / "env_inflated.yaml", result_folder / "output.trajopt.yaml",
                   result_folder / "trajectory_opt.yaml", result_folder / "trajectory_opt.html")
