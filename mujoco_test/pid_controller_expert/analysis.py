import numpy as np
from matplotlib import pyplot as plt

trajectory = np.loadtxt("thrifty/trajectories/trajectory_2024-12-09 18:56:15.412951.csv", delimiter=",")
# trajectory = np.loadtxt("trajectories/trajectory_2024-12-09 16:56:45.847941.csv", delimiter=",")
expert = np.loadtxt("trajectories/trajectory_2024-12-09 17:07:23.163250.csv", delimiter=",")


x1, y1, z1, x_vel, y_vel, z_vel = trajectory[:, 0], trajectory[:, 1], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5], trajectory[:, 6]
x_d, y_d, z_d, x_vel_d, y_vel_d, z_vel_d = trajectory[:, 6], trajectory[:, 7], trajectory[:, 8], trajectory[:, 9], trajectory[:, 10], trajectory[:, 11]
x_e, y_e, z_e, x_vel, y_vel, z_vel = expert[:, 0], expert[:, 1], expert[:, 2], expert[:, 4], expert[:, 5], expert[:, 6]

plt.figure(figsize=(10, 10))
plt.plot(x1, y1, color='blue', label='policy trajectory', alpha=0.7)
plt.plot(x_d, y_d, color='green', label='ideal trajectory', alpha=0.7)
# plt.plot(x_e, y_e, color='red', label='expert trajectory', alpha=0.7)
plt.plot(x_d[0],y_d[0], marker="^",color='black', label='start_pos')
plt.title('point mass trajectories')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()

pos_error = np.linalg.norm(trajectory[:, 0:3]-trajectory[:, 6:9], axis=1)
expert_error = np.linalg.norm(expert[:, 0:3]-expert[:, 6:9], axis=1)


# plt.figure(figsize=(10, 10))
# plt.title('state error')
# plt.plot(pos_error, label='policy error')
# # plt.plot(expert_error, color='red', label='expert error')
# plt.xlabel('step')
# plt.ylabel('state error')
# plt.grid(True)
# plt.legend()
# plt.show()

vel_error = np.linalg.norm(trajectory[:, 3:6]-trajectory[:, 9:12], axis=1)
expert_error = np.linalg.norm(expert[:, 3:6]-trajectory[:, 9:12], axis=1)

# plt.figure(figsize=(10, 10))
# plt.title('velocity error')
#
# plt.plot(vel_error, label='policy error')# plt.plot(expert_error, color='red', label='expert error')
# plt.xlabel('step')
# plt.ylabel('velocity error')
# plt.grid(True)
# plt.legend()
# plt.show()

pos_difference= np.linalg.norm(trajectory[:, 0:3]-expert[:, 0:3], axis=1)


# plt.figure(figsize=(10, 10))
# plt.title('state difference')
#
# plt.plot(pos_difference, label='policy expert difference')
# plt.xlabel('step')
# plt.ylabel('velocity error')
# plt.grid(True)
# plt.legend()
# plt.show()