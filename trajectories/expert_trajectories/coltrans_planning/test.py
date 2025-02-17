import os
import yaml

dirname = os.path.dirname(__file__)
traj_dir = dirname + "/../trajectories/expert_trajectories/coltrans_planning/"

dynamics = dirname + "/../src/dynamics/"

trajectory_opt = dirname + "/forest_4robots.yaml"

with open(trajectory_opt, 'r') as file:
    trajectories = yaml.safe_load(file)['result']

    trajectories
