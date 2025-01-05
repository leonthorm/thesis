import numpy as np
from matplotlib import pyplot as plt

from src.thrifty.algos.thriftydagger import thrifty


# expert = np.loadtxt("trajectories/dagger/trajectory_2024-12-11 23:19:01.307048.csv", delimiter=",")
# trajectory = np.loadtxt("../../src/thrifty/trajectories/trajectory_2024-12-09 18:56:15.412951.csv", delimiter=",")
# trajectory = np.loadtxt("trajectories/dagger/trajectory_2024-12-11 23:19:16.928623.csv", delimiter=",")


# x1, y1, z1, x_vel, y_vel, z_vel = trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], trajectory[:, 3], trajectory[:, 4], trajectory[:, 5]
# x_d, y_d, z_d, x_vel_d, y_vel_d, z_vel_d = trajectory[:, 6], trajectory[:, 7], trajectory[:, 8], trajectory[:, 9], trajectory[:, 10], trajectory[:, 11]
# x_e, y_e, z_e, x_vel, y_vel, z_vel = expert[:, 0], expert[:, 1], expert[:, 2], expert[:, 3], expert[:, 4], expert[:, 5]


def plot(to_plot, thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert=None):
    (
        x_e_robot1, y_e_robot1, z_e_robot1,
        x_e_vel_robot1, y_e_vel_robot1, z_e_vel_robot1,
        x_e_acc_robot1, y_e_acc_robot1, z_e_acc_robot1,
        x_e_robot2, y_e_robot2, z_e_robot2,
        x_e_vel_robot2, y_e_vel_robot2, z_e_vel_robot2,
        x_e_acc_robot2, y_e_acc_robot2, z_e_acc_robot2
    ) = (
        expert[:, 0], expert[:, 1], expert[:, 2],
        expert[:, 3], expert[:, 4], expert[:, 5],
        expert[:, 6], expert[:, 7], expert[:, 8],
        expert[:, 18], expert[:, 19], expert[:, 20],
        expert[:, 21], expert[:, 22], expert[:, 23],
        expert[:, 24], expert[:, 25], expert[:, 26]
    )

    (
        x_robot1, y_robot1, z_robot1,
        x_vel_robot1, y_vel_robot1, z_vel_robot1,
        x_acc_robot1, y_acc_robot1, z_acc_robot1,
        x_robot2, y_robot2, z_robot2,
        x_vel_robot2, y_vel_robot2, z_vel_robot2,
        x_acc_robot2, y_acc_robot2, z_acc_robot2
    ) = (
        trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
        trajectory[:, 3], trajectory[:, 4], trajectory[:, 5],
        trajectory[:, 6], trajectory[:, 7], trajectory[:, 8],
        trajectory[:, 18], trajectory[:, 19], trajectory[:, 20],
        trajectory[:, 21], trajectory[:, 22], trajectory[:, 23],
        trajectory[:, 24], trajectory[:, 25], trajectory[:, 26]
    )

    (
        x_d_robot1, y_d_robot1, z_d_robot1,
        x_vel_d_robot1, y_vel_d_robot1, z_vel_d_robot1,
        x_acc_d_robot1, y_acc_d_robot1, z_acc_d_robot1,
        x_d_robot2, y_d_robot2, z_d_robot2,
        x_vel_d_robot2, y_vel_d_robot2, z_vel_d_robot2,
        x_acc_d_robot2, y_acc_d_robot2, z_acc_d_robot2
    ) = (
        expert[:, 9], expert[:, 10], expert[:, 11],
        expert[:, 12], expert[:, 13], expert[:, 14],
        expert[:, 15], expert[:, 16], expert[:, 17],
        expert[:, 27], expert[:, 28], expert[:, 29],
        expert[:, 30], expert[:, 31], expert[:, 32],
        expert[:, 33], expert[:, 34], expert[:, 35]
    )

    if to_plot == 1:
        plt.figure(figsize=(10, 10))
        plt.plot(x_robot1, y_robot1, color='blue', label='policy trajectory robot1', alpha=0.7)
        plt.plot(x_d_robot1, y_d_robot1, color='green', label='ideal trajectory robot1', alpha=0.7)
        plt.plot(x_e_robot1, y_e_robot1, color='red', label='expert trajectory robot1', alpha=0.7)
        plt.plot(x_d_robot1[0], y_d_robot1[0], marker="+", color='black', label='start_pos robot1')

        plt.plot(x_robot2, y_robot2, color='blue', linestyle='dotted', label='policy trajectory robot2', alpha=0.7)
        plt.plot(x_d_robot2, y_d_robot2, color='green', linestyle='dotted', label='ideal trajectory robot2', alpha=0.7)
        plt.plot(x_e_robot2, y_e_robot2, color='red', linestyle='dotted', label='expert trajectory robot2', alpha=0.7)
        plt.plot(x_d_robot2[0], y_d_robot2[0], marker="x", color='black', label='start_pos robot2')

        plt.title(f'{trajectorie_type_dict[trajectory_type]}')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif to_plot == 2:
        plt.figure(figsize=(10, 10))

        plt.plot(x_robot1, z_robot1, color='blue', label='policy trajectory robot1', alpha=0.7)
        plt.plot(x_d_robot1, z_d_robot1, color='green', label='ideal trajectory robot1', alpha=0.7)
        plt.plot(x_e_robot1, z_e_robot1, color='red', label='expert trajectory robot1', alpha=0.7)
        plt.plot(x_d_robot1[0], z_d_robot1[0], marker="+", color='black', label='start_pos robot1')

        plt.plot(x_robot2, z_robot2, color='blue', linestyle='dotted', label='policy trajectory robot2', alpha=0.7)
        plt.plot(x_d_robot2, z_d_robot2, color='green', linestyle='dotted', label='ideal trajectory robot2', alpha=0.7)
        plt.plot(x_e_robot2, z_e_robot2, color='red', linestyle='dotted', label='expert trajectory robot2', alpha=0.7)
        plt.plot(x_d_robot2[0], z_d_robot2[0], marker="x", color='black', label='start_pos robot2')

        plt.title(f'{trajectorie_type_dict[trajectory_type]} with z axis')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif to_plot == 3:
        pos_error_robot1 = np.array([x_d_robot1, y_d_robot1, z_d_robot1]) - np.array([x_robot1, y_robot1, z_robot1])
        pos_error_norm_robot1 = np.linalg.norm(pos_error_robot1, axis=0)

        pos_error_robot2 = np.array([x_d_robot2, y_d_robot2, z_d_robot2]) - np.array([x_robot2, y_robot2, z_robot1])
        pos_error_norm_robot2 = np.linalg.norm(pos_error_robot2, axis=0)

        pos_error_x_robot1 = pos_error_robot1[0, :]
        pos_error_y_robot1 = pos_error_robot1[1, :]
        pos_error_z_robot1 = pos_error_robot1[2, :]

        pos_error_x_robot2 = pos_error_robot2[0, :]
        pos_error_y_robot2 = pos_error_robot2[1, :]
        pos_error_z_robot2 = pos_error_robot2[2, :]

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].plot(pos_error_norm_robot1, color='blue', label='pos_error_norm_robot1')
        ax[0].plot(pos_error_norm_robot2, color='black', label='pos_error_norm_robot2')
        ax[0].set_title(f'{trajectorie_type_dict[trajectory_type]} Pos Error Norm')
        ax[0].legend()
        ax[0].set_ylabel('Pos Error Norm')
        ax[0].set_xlabel('Step')
        ax[0].grid(True)

        ax[1].plot(pos_error_x_robot1, color='red', label='Position error x_robot1')
        ax[1].plot(pos_error_y_robot1, color='blue', label='Position error y_robot1')
        ax[1].plot(pos_error_z_robot1, color='green', label='Position error z_robot1')

        ax[1].plot(pos_error_x_robot2, color='red', linestyle='dotted', label='Position error x_robot2')
        ax[1].plot(pos_error_y_robot2, color='blue', linestyle='dotted', label='Position error y_robot2')
        ax[1].plot(pos_error_z_robot2, color='green', linestyle='dotted', label='Position error z_robot2')

        ax[1].set_title(f'{trajectorie_type_dict[trajectory_type]} Position Error per Axis')
        ax[1].legend()
        ax[1].set_ylabel('Position Error')
        ax[1].set_xlabel('Step')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    elif to_plot == 4:
        vel_error_robot1 = np.array([x_vel_d_robot1, y_vel_d_robot1, z_vel_d_robot1]) - np.array(
            [x_vel_robot1, y_vel_robot1, z_vel_robot1])
        vel_error_norm_robot1 = np.linalg.norm(vel_error_robot1, axis=0)

        vel_error_robot2 = np.array([x_vel_d_robot2, y_vel_d_robot2, z_vel_d_robot2]) - np.array(
            [x_vel_robot2, y_vel_robot2, z_vel_robot2])
        vel_error_norm_robot2 = np.linalg.norm(vel_error_robot2, axis=0)

        vel_error_x_robot1 = vel_error_robot1[0, :]
        vel_error_y_robot1 = vel_error_robot1[1, :]
        vel_error_z_robot1 = vel_error_robot1[2, :]

        vel_error_x_robot2 = vel_error_robot2[0, :]
        vel_error_y_robot2 = vel_error_robot2[1, :]
        vel_error_z_robot2 = vel_error_robot2[2, :]

        fig, ax = plt.subplots(2, 1, figsize=(10, 10))

        ax[0].plot(vel_error_norm_robot1, color='blue', label='vel_error_norm_robot1')
        ax[0].plot(vel_error_norm_robot2, color='red', label='vel_error_norm_robot2')
        ax[0].set_title(f'{trajectorie_type_dict[trajectory_type]} Velocity Error Norm')
        ax[0].legend()
        ax[0].set_ylabel('Velocity Error Norm')
        ax[0].set_xlabel('Step')
        ax[0].grid(True)

        ax[1].plot(vel_error_x_robot1, color='red', label='velocity error x_robot1')
        ax[1].plot(vel_error_y_robot1, color='blue', label='velocity error y_robot1')
        ax[1].plot(vel_error_z_robot1, color='green', label='velocity error z_robot1')

        ax[1].plot(vel_error_x_robot2, color='red', linestyle='dotted', label='velocity error x_robot2')
        ax[1].plot(vel_error_y_robot2, color='blue', linestyle='dotted', label='velocity error y_robot2')
        ax[1].plot(vel_error_z_robot2, color='green', linestyle='dotted', label='velocity error z_robot2')

        ax[1].set_title('Velocity Error per Axis')
        ax[1].legend()
        ax[1].set_ylabel('Velocity Error')
        ax[1].set_xlabel('Step')
        ax[1].grid(True)

        plt.tight_layout()
        plt.show()

    elif to_plot == 5:
        plt.figure(figsize=(10, 10))
        pos_difference = np.linalg.norm(trajectory[:, 0:3] - expert[:, 0:3], axis=1)

        plt.title(f'{trajectorie_type_dict[trajectory_type]}  state difference')
        plt.plot(pos_difference, label='policy expert difference')
        plt.xlabel('step')
        plt.ylabel('difference')
        plt.legend()
        plt.grid(True)
        plt.show()

    elif to_plot == 6:
        fig, ax = plt.subplots(3, 1, figsize=(10, 15))
        ax[0].plot(x_d_robot1, color='red', linestyle='dotted', label='ideal x_robot1')
        ax[0].plot(y_d_robot1, color='blue', linestyle='dotted', label='ideal y_robot1')
        ax[0].plot(z_d_robot1, color='green', linestyle='dotted', label='ideal z_robot1')
        ax[0].plot(x_e_robot1, color='red', label='expert x_robot1')
        ax[0].plot(y_e_robot1, color='blue', label='expert y_robot1')
        ax[0].plot(z_e_robot1, color='green', label='expert z_robot1')
        ax[0].set_title(f'{trajectorie_type_dict[trajectory_type]}  Position per Axis')
        ax[0].legend()
        ax[0].set_ylabel('Position')
        ax[0].set_xlabel('Step')
        ax[0].grid(True)

        ax[1].plot(x_vel_d_robot1, color='red', linestyle='dotted', label='ideal x_vel_robot1')
        ax[1].plot(y_vel_d_robot1, color='blue', linestyle='dotted', label='ideal y_vel_robot1')
        ax[1].plot(z_vel_d_robot1, color='green', linestyle='dotted', label='ideal z_vel_robot1')
        ax[1].plot(x_e_vel_robot1, color='red', label='expert x_vel_robot1')
        ax[1].plot(y_e_vel_robot1, color='blue', label='expert y_vel_robot1')
        ax[1].plot(z_vel_d_robot1, color='green', label='expert z_vel_robot1')
        ax[1].set_title('Velocity per Axis')
        ax[1].legend()
        ax[1].set_ylabel('Velocity')
        ax[1].set_xlabel('Step')
        ax[1].grid(True)

        ax[2].plot(x_acc_d_robot1, color='red', linestyle='dotted', label='ideal x_robot1 acc')
        ax[2].plot(y_acc_d_robot1, color='blue', linestyle='dotted', label='ideal y_robot1 acc')
        ax[2].plot(z_acc_d_robot1, color='green', linestyle='dotted', label='ideal z_robot1 acc')
        ax[2].plot(x_e_acc_robot1, color='red', label='expert x_robot1 acc')
        ax[2].plot(y_e_acc_robot1, color='blue', label='expert y_robot1 acc')
        ax[2].plot(z_e_acc_robot1, color='green', label='expert z_robot1 acc')
        ax[2].set_title('Acceleration per Axis')
        ax[2].legend()
        ax[2].set_ylabel('Acceleration')
        ax[2].set_xlabel('Step')
        ax[2].grid(True)

        plt.tight_layout()
        plt.show()
    elif to_plot == 7:
        fig, ax = plt.subplots(1, 1, figsize=(10, 15))
        # ax.plot(x_d_robot1, color='red', linestyle='dotted', label='ideal x_robot1')
        # ax.plot(y_d_robot1, color='blue', linestyle='dotted', label='ideal y_robot1')
        ax.plot(z_d_robot1, color='green', linestyle='dotted', label='ideal z_robot1')
        # ax.plot(x_e_robot1, color='red', label='expert x_robot1')
        # ax.plot(y_e_robot1, color='blue', label='expert y_robot1')
        ax.plot(z_e_robot1, color='green', label='expert z_robot1')
        ax.set_title(f'{trajectorie_type_dict[trajectory_type]}  Position per Axis')
        ax.legend()
        ax.set_ylabel('Position')
        ax.set_xlabel('Step')
        ax.grid(True)

        plt.tight_layout()
        plt.show()


def get_metrics(thrifty=False, trajectory=None):
    if trajectory is None:
        trajectory = np.loadtxt("trajectories/dagger/trajectory_dagger_wave.csv", delimiter=",")

    if thrifty:
        print('thrifty')
    else:
        print('dagger')
    state_error = np.linalg.norm(
        np.hstack((trajectory[:, 0:3], trajectory[:, 18:21])) - np.hstack((trajectory[:, 9:12], trajectory[:, 27:30])),
        axis=1)
    vel_error = np.linalg.norm(
        np.hstack((trajectory[:, 3:6], trajectory[:, 21:24])) - np.hstack((trajectory[:, 12:15], trajectory[:, 30:33])),
        axis=1)
    state_error_mean = np.std(state_error)
    state_error_std = np.mean(state_error)
    vel_error_mean = np.std(vel_error)
    vel_error_std = np.mean(vel_error)

    print("state_error mean: ", np.mean(state_error))
    print("state_error std: ", np.std(state_error))
    print("vel_error mean: ", np.mean(vel_error))
    print("vel_error std: ", np.std(vel_error))
    print("x pos error: ", np.mean(np.absolute(trajectory[:, 0] - trajectory[:, 9])))
    print("y pos error: ", np.mean(np.absolute(trajectory[:, 1] - trajectory[:, 10])))
    print("z pos error: ", np.mean(np.absolute(trajectory[:, 2] - trajectory[:, 11])))

    return state_error_mean, state_error_std, vel_error_mean, vel_error_std


def plot_all(thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert):
    for i in range(1, 7):
        plot(i, thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert)


def load_trajectory(trajectories, trajectory_type, thrifty):
    dagger_csv = f"trajectory_dagger_{trajectories[trajectory_type]}.csv"
    thrifty_csv = f"trajectory_thrifty_{trajectories[trajectory_type]}.csv"
    # thrifty_csv = "thrifty/trajectory_thrifty.csv"
    expert_dagger_csv = f"trajectory_expert_dagger_{trajectories[trajectory_type]}.csv"
    expert_thrifty_csv = f"trajectory_expert_thrifty_{trajectories[trajectory_type]}.csv"
    if thrifty:
        trajectory = np.loadtxt("trajectories/thrifty/dbcbs/" + thrifty_csv, skiprows=1, delimiter=",")
        expert = np.loadtxt("trajectories/thrifty/dbcbs/" + expert_thrifty_csv, skiprows=1, delimiter=",")
    else:
        trajectory = np.loadtxt("trajectories/dagger/dbcbs/" + dagger_csv, skiprows=1, delimiter=",")
        expert = np.loadtxt("trajectories/dagger/dbcbs/" + expert_dagger_csv, skiprows=1, delimiter=",")

    return trajectory, expert


if __name__ == '__main__':
    to_plot = {
        1: "point mass trajectories",
        2: "point mass trajectories with z-axis",
        3: "state error",
        4: "velocity error",
        5: "state difference",
        6: "plot all components"
    }
    trajectorie_type_dict = {
        1: "swap2_double_integrator_3d",
    }
    thrifty = True
    trajectory_type = 1

    trajectory, expert = load_trajectory(trajectorie_type_dict, trajectory_type, thrifty)
    plot(4, thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert)
    get_metrics(thrifty, trajectory)
    # plot_all(thrifty, trajectorie_type_dict, trajectory_type, trajectory, expert)
