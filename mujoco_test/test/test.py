import yaml
import numpy as np


trajectory_path = "../db_cbs_point_mass/trajectories/swap1_double_integrator_3d_opt.yaml"
with open(trajectory_path, 'r') as file:
    trajectory  = yaml.safe_load(file)['result'][0]

mass = 0.1
states =  np.array(trajectory['states'])
actions = np.array(trajectory['actions'])
print(type(states))

state = states[0]
state_idx = np.where(states == state)[0][0]

if state_idx >= trajectory['num_states']-1:
    control = np.zeros(3)
else:
    control = actions[state_idx] * mass


print(control)