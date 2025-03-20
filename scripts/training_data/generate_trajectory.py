import argparse
import os
import random
import time
from pathlib import Path
import numpy as np
import yaml


def generate_start_goal(env_min, env_max):
    """
    Generates a random start position in one of the 8 corners of the environment space,
    ensuring it is within a 0.8 buffer from the edges. The goal position is randomly selected
    in one of the other 7 corners, ensuring it is not the same as the start position.
    """
    buffer = 0.7
    corners = [
        [env_min[0] + buffer, env_min[1] + buffer, env_min[2] + 0.1],
        [env_min[0] + buffer, env_min[1] + buffer, env_max[2] - buffer],
        [env_min[0] + buffer, env_max[1] - buffer, env_min[2] + 0.1],
        [env_min[0] + buffer, env_max[1] - buffer, env_max[2] - buffer],
        [env_max[0] - buffer, env_min[1] + buffer, env_min[2] + 0.1],
        [env_max[0] - buffer, env_min[1] + buffer, env_max[2] - buffer],
        [env_max[0] - buffer, env_max[1] - buffer, env_min[2] + 0.1],
        [env_max[0] - buffer, env_max[1] - buffer, env_max[2] - buffer]
    ]

    start = random.choice(corners)
    corners.remove(start)  # Remove the start corner to ensure goal is different
    goal = random.choice(corners)

    return start, goal


def generate_random_obstacles(num_obstacles, env_min, env_max, start, goal):
    obstacles = []
    for _ in range(num_obstacles):
        while True:
            center = [
                random.uniform(env_min[0], env_max[0]),  # x-coordinate
                random.uniform(env_min[1], env_max[1]),  # y-coordinate
                0.0  # z-coordinate fixed
            ]
            # Ensure obstacle is not in start or goal position
            if center[:2] != start[:2] and center[:2] != goal[:2]:
                break
        obstacles.append({
            "center": center,
            "size": [0.05, 0.05, 3.0],
            "type": "box"
        })
    return obstacles


def generate_midpoint_obstacle(start, goal):
    """
    Generates an obstacle of size 1x1x1 that sits on the line between start and goal
    at a Euclidean distance of 2 from the start position.
    """
    start = np.array(start)
    goal = np.array(goal)
    direction = goal - start
    direction = direction / np.linalg.norm(direction)  # Normalize direction vector
    obstacle_center = start + 2 * direction  # Place the obstacle 2 units from start
    obstacle_center[2] = 0.5  # Set z position to center it at z=0.5

    return {
        "center": obstacle_center.tolist(),
        "size": [1.0, 1.0, 1.0],
        "type": "box"
    }


def generate_direction_vectors(start, num_robots):
    azimuth = random.uniform(0, 6.28)
    elevation = random.uniform(0, 1.57)

    directions = []
    azimuth_elevation = []
    spacing = 2 * np.pi / num_robots
    for i in range(num_robots):
        azimuth_n = i * spacing + azimuth

        unitvec = np.array([
            np.cos(azimuth_n) * np.cos(elevation),
            np.sin(azimuth_n) * np.cos(elevation),
            np.sin(elevation)
        ])

        directions.append(-unitvec)
        azimuth_elevation.append((azimuth_n, elevation))

    return np.array(directions), np.array(azimuth_elevation).flatten()


def create_dyno_yaml(cable_lengths, directions, environment, goal, num_robots, start):
    dyno_yaml = {"environment": environment}
    dyno_yaml["name"] = 'quad3d_payload_empty_0'
    r_vec_len = 6 + num_robots * 13
    r_start = np.zeros(r_vec_len)
    r_goal = np.zeros(r_vec_len)
    r_start[0:3] = start
    r_goal[0:3] = goal
    for n in range(num_robots):
        cable_start_idx = 6 + n * 6
        robot_start_idx = 6 + num_robots * 6 + n * 7

        r_start[cable_start_idx:cable_start_idx + 3] = directions[n]
        r_goal[cable_start_idx:cable_start_idx + 3] = directions[n]

        r_start[robot_start_idx + 3] = 1.
        r_goal[robot_start_idx + 3] = 1.
    robots = [
        {
            "goal": r_goal.tolist(),
            "l": cable_lengths.tolist(),
            "quadsNum": num_robots,
            "start": r_start.tolist(),
            "type": f'point_{num_robots}',
        }
    ]
    dyno_yaml["robots"] = robots

    return dyno_yaml


def create_col_yaml(azimuth_elevation, goal, num_robots, start, cable_lengths, environment):
    print(azimuth_elevation)
    col_yaml = {
        "plannerType": 'rrtstar',
        "timelimit": 300.0,
        "numofcables": num_robots,
        "interpolate": 1000,
    }
    col_yaml["environment"] = environment
    col_yaml["plannerType"] = 'rrtstar'
    col_yaml["timelimit"] = 300.0

    p_vec_len = 7 + num_robots * 2
    p_start = np.zeros(p_vec_len)
    p_goal = np.zeros(p_vec_len)
    p_start[0:3] = start
    p_goal[0:3] = goal
    p_start[6] = 1.
    p_goal[6] = 1.
    p_start[7:] = azimuth_elevation
    p_goal[7:] = azimuth_elevation
    print(start)
    shape = {
        "type": "sphere",
        "size": 0.01
    }
    payload = {
        "start": p_start.tolist(),
        "goal": p_goal.tolist(),
    }

    col_yaml["payload"] = payload
    col_yaml["payload"]["shape"] = {
        "type": "sphere",
        "size": 0.01
    }
    # col_yaml["payload"]["shape"]["size"] = 0.01

    # todo different cable lengts?
    max_values = [6.28, 1.57]
    cables = {
        "lengths": cable_lengths.tolist(),
        "attachmentpoints": np.zeros((1, 3 * num_robots)).tolist(),
        "min": np.zeros((1, 2 * num_robots)).tolist(),
        "max": np.tile(max_values, num_robots).tolist(),

    }
    col_yaml["cables"] = cables

    return col_yaml


def create_randomized_yaml(num_robots, num_obstacles=10):
    environment = {
        "max": [4.5, 1.0, 1],
        "min": [-1.5, -1.0, -0.1]
    }

    start, goal = generate_start_goal(environment["min"], environment["max"])
    obstacles = generate_random_obstacles(num_obstacles, environment["min"], environment["max"], start, goal)
    midpoint_obstacle = generate_midpoint_obstacle(start, goal)
    obstacles.append(midpoint_obstacle)

    directions, azimuth_elevation = generate_direction_vectors(start, num_robots)

    environment["obstacles"] = obstacles

    random_length = round(np.random.uniform(0.1, 0.5), 2)
    cable_lengths = np.full((1, num_robots), random_length)

    col_yaml = create_col_yaml(azimuth_elevation, goal, num_robots, start, cable_lengths, environment)

    dyno_yaml = create_dyno_yaml(cable_lengths, directions, environment, goal, num_robots, start)

    directory = 'random_envs'

    # Check if the directory exists, and if not, create it
    if not os.path.exists(directory):
        os.makedirs(Path(directory) / "col_env")
        os.makedirs(Path(directory) / "dyno_env")
        print(f"Directory '{directory}' created.")
    directory = Path(directory)
    unix_time = int(time.time())
    dyno_yaml_name = directory / "dyno_env"/ f"{unix_time}_random_{num_robots}robots.yaml"
    col_yaml_name = directory / "col_env" / f"{unix_time}_random_{num_robots}robots.yaml"

    with open(dyno_yaml_name, "w") as file:
        yaml.safe_dump(dyno_yaml, file, default_flow_style=None)
    with open(col_yaml_name, "w") as file:
        yaml.safe_dump(col_yaml, file, default_flow_style=None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-nr", "--num_robots",
        default=None,
        type=int,
        required=True,
    )
    parser.add_argument(
        "-nt", "--num_trajectories",
        default=1,
        type=int,
        required=False,
    )
    args = parser.parse_args()

    num_robots = args.num_robots
    num_trajectories = args.num_trajectories

    num_obstacles = int(random.uniform(0, 20))

    for _ in range(num_trajectories):
        time.sleep(1)
        create_randomized_yaml(num_robots, num_obstacles)


if __name__ == "__main__":
    main()
