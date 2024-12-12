import numpy as np
from matplotlib import pyplot as plt

from src.thrifty.algos.thriftydagger import thrifty

# expert = np.loadtxt("trajectories/dagger/trajectory_2024-12-11 23:19:01.307048.csv", delimiter=",")
# trajectory = np.loadtxt("../../src/thrifty/trajectories/trajectory_2024-12-09 18:56:15.412951.csv", delimiter=",")
# trajectory = np.loadtxt("trajectories/dagger/trajectory_2024-12-11 23:19:16.928623.csv", delimiter=",")


# x1, y1, z1, x_vel, y_vel, z_vel = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5]
# x_d, y_d, z_d, x_vel_d, y_vel_d, z_vel_d = trajectory[:, 6], trajectory[:, 7], trajectory[:, 8], trajectory[:, 9], trajectory[:, 10], trajectory[:, 11]
# x_e, y_e, z_e, x_vel, y_vel, z_vel = expert[:, 0], expert[:, 1], expert[:, 2], expert[:, 3], expert[:, 4], expert[:, 5]




def plot(to_plot, thrifty, traj_csv, thrifty_csv, expert_csv=None):
    if thrifty:
        # traj_csv = "trajectory_2024-12-11 23:19:16.928623.csv"
        trajectory = np.loadtxt("trajectories/dagger/"+thrifty_csv, delimiter=",")
    else:
        # expert_csv="trajectory_2024-12-11 23:19:01.307048.csv"
        # traj_csv ="trajectory_2024-12-11 23:19:16.928623.csv"
        expert = np.loadtxt("trajectories/dagger/"+expert_csv, delimiter=",")
        trajectory = np.loadtxt("trajectories/dagger/"+traj_csv, delimiter=",")
        x_e, y_e, z_e, x_vel, y_vel, z_vel = (expert[:, 0], expert[:, 1], expert[:, 2],
                                              expert[:, 3], expert[:, 4], expert[:, 5])

    x1, y1, z1, x_vel, y_vel, z_vel = (trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                                       trajectory[:, 3], trajectory[:, 4], trajectory[:, 5])
    x_d, y_d, z_d, x_vel_d, y_vel_d, z_vel_d = (trajectory[:, 6], trajectory[:, 7], trajectory[:, 8],
                                                trajectory[:, 9], trajectory[:, 10], trajectory[:, 11])
    plt.figure(figsize=(10, 10))
    if to_plot == 0:
        plt.plot(x1, y1, color='blue', label='policy trajectory', alpha=0.7)
        plt.plot(x_d, y_d, color='green', label='ideal trajectory', alpha=0.7)
        if not thrifty:
            plt.plot(x_e, y_e, color='red', label='expert trajectory', alpha=0.7)
        plt.plot(x_d[0],y_d[0], marker="^",color='black', label='start_pos')
        plt.title('point mass trajectories')
        plt.xlabel('X')
        plt.ylabel('Y')

    elif to_plot == 1:
        plt.plot(x1, z1, color='blue', label='policy trajectory', alpha=0.7)
        plt.plot(x_d, z_d, color='green', label='ideal trajectory', alpha=0.7)
        if not thrifty:
            plt.plot(x_e, z_e, color='red', label='expert trajectory', alpha=0.7)
        plt.plot(x_d[0],z_d[0], marker="^",color='black', label='start_pos')
        plt.title('point mass trajectories with z-axis')
        plt.xlabel('X')
        plt.ylabel('Z')

    elif to_plot == 2:
        pos_error = np.linalg.norm(trajectory[:, 0:3]-trajectory[:, 6:9], axis=1)

        plt.title('state error')
        plt.plot(pos_error, label='policy error')
        if not thrifty:
            expert_error = np.linalg.norm(expert[:, 0:3] - expert[:, 6:9], axis=1)
            plt.plot(expert_error, color='red', label='expert error')
        plt.xlabel('step')
        plt.ylabel('state error')

    elif to_plot == 3:
        vel_error = np.linalg.norm(trajectory[:, 3:6]-trajectory[:, 9:12], axis=1)

        plt.title('velocity error')
        plt.plot(vel_error, label='policy error')
        if not thrifty:
            expert_error = np.linalg.norm(expert[:, 3:6] - trajectory[:, 9:12], axis=1)
            plt.plot(expert_error, color='red', label='expert error')
        plt.xlabel('step')
        plt.ylabel('velocity error')

    elif to_plot == 4:

        pos_difference= np.linalg.norm(trajectory[:, 0:3]-expert[:, 0:3], axis=1)

        plt.title('state difference')
        plt.plot(pos_difference, label='policy expert difference')
        plt.xlabel('step')
        plt.ylabel('difference')


    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    to_plot = {
        0: "point mass trajectories",
        1: "point mass trajectories with z-axis",
        2: "state error",
        3: "velocity error",
        4: "state difference",
    }
    thrifty = True
    traj_csv = "trajectory_2024-12-11 23:52:30.303672.csv"
    thrifty_csv = "trajectory_2024-12-12 00:44:30.879624.csv"
    expert_csv = "trajectory_2024-12-11 23:52:07.198008.csv"
    plot(4, thrifty, traj_csv, thrifty_csv, expert_csv)