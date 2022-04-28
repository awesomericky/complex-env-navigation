import pdb

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from os.path import isdir
from os import makedirs

def check_saving_folder(folder_name):
    if not isdir(folder_name):
        makedirs(folder_name)

def plot_command_result(command_traj, folder_name, task_name, run_name, n_update, control_dt):
    """
    command_traj : (n_steps, 3)
    n_update : current epoch

    // 0: forward_vel, 1: lateral_vel, 2: yaw_rate
    """

    save_folder_name = f"{task_name}/{folder_name}/{run_name}"
    check_saving_folder(save_folder_name)
    x_value = np.arange(command_traj.shape[0]) * control_dt
    ylabels = ['Forward velocity [m/s]', 'Lateral velocity [m/s]', 'Yaw rate [rad/s]']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

    for i in range(command_traj.shape[-1]):
        ax[i].plot(x_value, command_traj[:, i])
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(ylabels[i])
    plt.savefig(f'{save_folder_name}/{n_update}.png')
    plt.clf()
    plt.close()

def plot_command_tracking_result(desird_result, actual_result, task_name, run_name, n_update, control_dt):
    """
    desired_result : user command (n_steps, 3)
    actual_result : (n_steps, 3)
    n_update : current epoch
    
    // 0: forward_vel, 1: lateral_vel, 2: yaw_rate
    """

    save_folder_name = f"command_tracking_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)
    x_value = np.arange(desird_result.shape[0]) * control_dt
    ylabels = ['Forward velocity [m/s]', 'Lateral velocity [m/s]', 'Yaw rate [rad/s]']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

    for i in range(desird_result.shape[-1]):
        ax[i].plot(x_value, desird_result[:, i], label='command')
        ax[i].plot(x_value, actual_result[:, i], label='real')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(ylabels[i])
    plt.legend()
    plt.savefig(f'{save_folder_name}/{n_update}.png')
    plt.clf()
    plt.close()

def plot_command_transform_result(before_user_command, after_user_command, actual_result, task_name, run_name, n_update, control_dt):
    """
    before_user_command : user command, before modification (n_steps, 3)
    after_user_command : user command, after modification (n_steps, 3)
    actual_result : (n_steps, 3)
    n_update : current epoch

    // 0: forward_vel, 1: lateral_vel, 2: yaw_rate
    """

    save_folder_name = f"command_tracking_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)
    x_value = np.arange(before_user_command.shape[0]) * control_dt
    ylabels = ['Forward velocity [m/s]', 'Lateral velocity [m/s]', 'Yaw rate [rad/s]']

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(25, 5))

    for i in range(before_user_command.shape[-1]):
        ax[i].plot(x_value, before_user_command[:, i], label='command (before)')
        ax[i].plot(x_value, after_user_command[:, i], label='command (after)')
        ax[i].plot(x_value, actual_result[:, i], label='real')
        ax[i].set_xlabel('Time [s]')
        ax[i].set_ylabel(ylabels[i])
    plt.legend()
    plt.savefig(f'{save_folder_name}/{n_update}.png')
    plt.clf()
    plt.close()

def plot_contact_result(contact_log, task_name, run_name, n_update, control_dt):
    save_folder_name = f"command_tracking_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)
    np.savez_compressed(f"{save_folder_name}/contact_{n_update}.npz", contact=contact_log)

    contact_log = np.log(contact_log + 1e-6)
    contact_log = contact_log - np.min(contact_log)
    contact_log = np.where(contact_log > 0, 1, 0)

    start = 100
    total_step = 200
    single_step = 50
    fig, ax = plt.subplots(1,1, figsize=(20, 5))
    img = ax.imshow(contact_log[:, start:start + total_step], aspect='auto', cmap='Blues')
    x_label_list = [i*control_dt for i in range(start + single_step, start + total_step + 1, single_step)]
    y_label_list = ['LF', 'RF', 'LH', 'RH']
    ax.set_xticks([i for i in range(single_step, total_step + 1, single_step)])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list, fontsize=18)
    fig.colorbar(img)
    ax.set_xlabel('Time [s]', fontsize=12)
    plt.savefig(f"{save_folder_name}/contact_{n_update}.png")
    plt.clf()
    plt.close()

def replot_contact_result(task_name, run_name, n_update, control_dt, start_n_step, total_step_size, single_step_size=50):
    save_folder_name = f"command_tracking_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)
    contact_log = np.load(f"{save_folder_name}/contact_{n_update}.npz")['contact']

    contact_log = contact_log - np.min(contact_log)
    contact_log = np.where(contact_log > 0, 1, 0)

    start = start_n_step
    total_step = total_step_size
    single_step = single_step_size
    fig, ax = plt.subplots(1,1, figsize=(20, 5))
    img = ax.imshow(contact_log[:, start:start + total_step], aspect='auto', cmap='Blues')
    x_label_list = [i*control_dt for i in range(start + single_step, start + total_step + 1, single_step)]
    y_label_list = ['LF', 'RF', 'LH', 'RH']
    ax.set_xticks([i for i in range(single_step, total_step + 1, single_step)])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xticklabels(x_label_list)
    ax.set_yticklabels(y_label_list, fontsize=18)
    fig.colorbar(img)
    ax.set_xlabel('Time [s]', fontsize=12)
    plt.savefig(f"{save_folder_name}/contact_{n_update}.png")
    plt.clf()
    plt.close()

def plot_torque_result(torque, task_name, run_name, n_update, control_dt):
    save_folder_name = f"command_tracking_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)
    n_step = torque.shape[-1]
    x_value = np.arange(n_step) * control_dt
    joint_name = ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
    torque_limit = 80 * 0.3
    torque_limit = np.ones(n_step) * torque_limit

    for i in range(12):
        plt.plot(x_value, torque[i], label=joint_name[i])
    plt.plot(x_value, torque_limit, label='Limit')
    plt.xlabel('Time [s]')
    plt.ylabel('Torque [Nm]')
    plt.legend()
    plt.savefig(f'{save_folder_name}/torque_{n_update}.png')
    plt.clf()
    plt.close()

def plot_joint_velocity_result(joint_velocity, task_name, run_name, n_update, control_dt):
    save_folder_name = f"command_tracking_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)
    n_step = joint_velocity.shape[-1]
    x_value = np.arange(n_step) * control_dt
    joint_name = ['LF_HAA', 'LF_HFE', 'LF_KFE', 'RF_HAA', 'RF_HFE', 'RF_KFE', 'LH_HAA', 'LH_HFE', 'LH_KFE', 'RH_HAA', 'RH_HFE', 'RH_KFE']
    velocity_limit = 7.5
    velocity_limit = np.ones(n_step) * velocity_limit

    for i in range(12):
        plt.plot(x_value, joint_velocity[i], label=joint_name[i])
    plt.plot(x_value, velocity_limit, label='Limit')
    plt.xlabel('Time [s]')
    plt.ylabel('Joint velocity [rad/s]')
    plt.legend()
    plt.savefig(f'{save_folder_name}/joint_velocity_{n_update}.png')
    plt.clf()
    plt.close()

def plot_trajectory_prediction_result(P_col_traj, coordinate_traj, predicted_P_col_traj, predicted_coordinate_traj,
                                        task_name, run_name, n_update, prediction_time):
    """

    Plot ground truth and predicted trajectory in the form of trajectory (orientation not included in the states)

    :param P_col_traj: (n_data, traj_len, 1)
    :param coordinate_traj: (n_data, traj_len, 2)
    :param predicted_P_col_traj: (n_data, traj_len, 1)
    :param predicted_coordinate_traj: (n_data, traj_len, 2)
    :param prediction_time: int
    :return:
    """
    save_folder_name = f"trajectory_prediction_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)

    n_data = P_col_traj.shape[0]
    traj_len = P_col_traj.shape[1]
    coordinate_traj = np.concatenate((np.zeros((n_data, 1, 2)), coordinate_traj), axis=1)
    predicted_coordinate_traj = np.concatenate((np.zeros((n_data, 1, 2)), predicted_coordinate_traj), axis=1)

    P_col_traj = np.squeeze(P_col_traj, axis=2)
    predicted_P_col_traj = np.squeeze(predicted_P_col_traj, axis=2)

    fig, ax = plt.subplots(ncols=n_data, figsize=(35, 5))

    for i in range(n_data):
        lines = []
        for j in range(traj_len):
            lines.append([tuple(predicted_coordinate_traj[i, j, :]), tuple(predicted_coordinate_traj[i, j + 1, :])])
        lines = LineCollection(lines, array=predicted_P_col_traj[i, :], cmap=plt.cm.Wistia, linewidths=2)
        ax[i].add_collection(lines)

        col_idxs = np.where(P_col_traj[i, :] == 1.)[0]
        if col_idxs != np.array([]):
            min_col_idx = np.min(col_idxs)
            ax[i].plot(coordinate_traj[i, :min_col_idx + 2, 0], coordinate_traj[i, :min_col_idx + 2, 1], color="black", linestyle='dotted')  # +2 because we concatenated zero vector and we want the trajectory to be continuous
            ax[i].scatter(coordinate_traj[i, min_col_idx + 1:, 0], coordinate_traj[i, min_col_idx + 1:, 1], s=10, marker='x', color="red")  # +1 because we concatenated zero vector
        else:
            ax[i].plot(coordinate_traj[i, :, 0], coordinate_traj[i, :, 1], color="black", linestyle='dotted')

        ax[i].set_xlim(-prediction_time, prediction_time)
        ax[i].set_ylim(-prediction_time, prediction_time)
    plt.suptitle("Dotted (real) / Solid (prediction)", fontsize=18)
    plt.savefig(f'{save_folder_name}/{n_update}.png')
    plt.clf()
    plt.close()

def plot_vector_trajectory_prediction_result(P_col_traj, coordinate_traj, predicted_P_col_traj, predicted_coordinate_traj,
                                             task_name, run_name, n_update):
    """

    Plot ground truth and predicted trajectory in the form of vector field (orientation included in the states)

    :param P_col_traj: (n_data, traj_len, 1)
    :param coordinate_traj: (n_data, traj_len, 3)
    :param predicted_P_col_traj: (n_data, traj_len, 1)
    :param predicted_coordinate_traj: (n_data, traj_len, 3)
    :return:
    """
    save_folder_name = f"trajectory_prediction_plot/{task_name}/{run_name}"
    check_saving_folder(save_folder_name)

    n_data = P_col_traj.shape[0]
    P_col_traj = np.concatenate((np.zeros((n_data, 1, 1)), P_col_traj), axis=1)
    coordinate_traj = np.concatenate((np.zeros((n_data, 1, 3)), coordinate_traj), axis=1)
    predicted_P_col_traj = np.concatenate((np.zeros((n_data, 1, 1)), predicted_P_col_traj), axis=1)
    predicted_coordinate_traj = np.concatenate((np.zeros((n_data, 1, 3)), predicted_coordinate_traj), axis=1)

    P_col_traj = np.squeeze(P_col_traj, axis=2)
    predicted_P_col_traj = np.squeeze(predicted_P_col_traj, axis=2)

    fig, ax = plt.subplots(ncols=n_data, figsize=(35, 5))

    for i in range(n_data):
        cm = matplotlib.cm.Wistia

        ax[i].quiver(predicted_coordinate_traj[i, :, 0], predicted_coordinate_traj[i, :, 1],
                     np.cos(predicted_coordinate_traj[i, :, 2]), np.sin(predicted_coordinate_traj[i, :, 2]),
                     predicted_P_col_traj[i], cmap=cm)
        ax[i].quiver(coordinate_traj[i, :, 0], coordinate_traj[i, :, 1],
                     np.cos(coordinate_traj[i, :, 2]), np.sin(coordinate_traj[i, :, 2]),
                     color='black')
        ax[i].set_xlim(-2, 2)
        ax[i].set_ylim(-2, 2)
    plt.suptitle("Black (real) / Color (prediction)", fontsize=18)
    plt.savefig(f'{save_folder_name}/{n_update}.png')
    plt.clf()
    plt.close()


