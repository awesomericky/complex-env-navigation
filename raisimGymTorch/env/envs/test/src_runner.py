from ruamel.yaml import YAML, dump, RoundTripDumper
from raisimGymTorch.env.bin import test
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
import os
import math
import time
import raisimGymTorch.algo.ppo.module as ppo_module
import torch.nn as nn
import numpy as np
import torch
import argparse
import pdb
from raisimGymTorch.env.envs.train.model import Forward_Dynamics_Model
from raisimGymTorch.env.envs.train.action import UserCommand, Stochastic_action_planner_normal
from raisimGymTorch.env.envs.train.storage import Buffer
import random

"""
Safety-Remote Control using random sampler

"""

def transform_coordinate_LW(w_init_coordinate, l_coordinate_traj):
    """
    Transform LOCAL frame coordinate trajectory to WORLD frame coordinate trajectory
    (LOCAL frame --> WORLD frame)

    :param w_init_coordinate: initial coordinate in WORLD frame (1, coordinate_dim)
    :param l_coordinate_traj: coordintate trajectory in LOCAL frame (n_step, coordinate_dim)
    :return:
    """
    transition_matrix = np.array([[np.cos(w_init_coordinate[0, 2]), np.sin(w_init_coordinate[0, 2])],
                                  [- np.sin(w_init_coordinate[0, 2]), np.cos(w_init_coordinate[0, 2])]], dtype=np.float32)
    w_coordinate_traj = np.matmul(l_coordinate_traj, transition_matrix)
    w_coordinate_traj += w_init_coordinate[:, :-1]
    return w_coordinate_traj

# task specification
task_name = "safety_remote_control"

evaluate_seed = 10
random.seed(evaluate_seed)
np.random.seed(evaluate_seed)
torch.manual_seed(evaluate_seed)

# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-fw', '--fdm_weight', help='Forward Dynamics Model weight path', type=str, default='')
parser.add_argument('-tw', '--tracking_weight', help='velocity command tracking policy weight path', type=str, default='')
parser.add_argument('-v', '--validate', help='validation or test', type=bool, default=False)
args = parser.parse_args()
FDM_weight_path = args.fdm_weight
command_tracking_weight_path = args.tracking_weight
validation = args.validate

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# complete path configuration
if FDM_weight_path == '':
    assert cfg["path"]["home"] != '' or cfg["path"]["FDM"] != '', "Weight path configuration not complete."
    FDM_weight_path = cfg["path"]["home"] + cfg["path"]["FDM"]
if command_tracking_weight_path == '':
    assert cfg["path"]["home"] != '' or cfg["path"]["command_tracking"] != '', "Weight path configuration not complete."
    command_tracking_weight_path = cfg["path"]["home"] + cfg["path"]["command_tracking"]

cfg["environment"]["test_initialize"]["point_goal"] = False
cfg["environment"]["test_initialize"]["safety_control"] = True
cfg["environment"]["visualize_path"] = False

# user command sampling
user_command = UserCommand(cfg, cfg['Naive']['planner']['number_of_sample'])

# create environment from the configuration file
cfg['environment']['num_envs'] = 1
cfg['environment']["harsh_collision"] = True

# create environment from the configuration file
env = VecEnv(test.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)

# shortcuts
user_command_dim = 3
proprioceptive_sensor_dim = 81
lidar_dim = 360
state_dim = cfg["environment_model"]["architecture"]["state_encoder"]["input"]
command_period_steps = math.floor(cfg['command_tracking']['command_period'] / cfg['environment']['control_dt'])
assert env.num_obs == proprioceptive_sensor_dim + lidar_dim, "Check configured sensor dimension"

# Use naive concatenation for encoding COM vel history
COM_feature_dim = cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["input"]
COM_history_time_step = cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["time_step"]
COM_history_update_period = int(cfg["environment_model"]["architecture"]["COM_encoder"]["naive"]["update_period"] / cfg["environment"]["control_dt"])
assert state_dim - lidar_dim == COM_feature_dim * COM_history_time_step, "Check COM_encoder output and state_encoder input in the cfg.yaml"

command_tracking_ob_dim = user_command_dim + proprioceptive_sensor_dim
command_tracking_act_dim = env.num_acts

COM_buffer = Buffer(env.num_envs, COM_history_time_step, COM_feature_dim)

# Load pre-trained command tracking policy weight
assert command_tracking_weight_path != '', "Pre-trained command tracking policy weight path should be determined."
command_tracking_policy = ppo_module.MLP(cfg['command_tracking']['architecture'], nn.LeakyReLU,
                                         command_tracking_ob_dim, command_tracking_act_dim)
command_tracking_policy.load_state_dict(torch.load(command_tracking_weight_path, map_location=device)['actor_architecture_state_dict'])
command_tracking_policy.to(device)
command_tracking_weight_dir = command_tracking_weight_path.rsplit('/', 1)[0] + '/'
iteration_number = command_tracking_weight_path.rsplit('/', 1)[1].split('_', 1)[1].rsplit('.', 1)[0]
env.load_scaling(command_tracking_weight_dir, int(iteration_number))
print("Loaded command tracking policy weight from {}\n".format(command_tracking_weight_path))

# Load learned environment model weight
loaded_environment_model = Forward_Dynamics_Model(state_encoding_config=cfg["environment_model"]["architecture"]["state_encoder"],
                                                  command_encoding_config=cfg["environment_model"]["architecture"]["command_encoder"],
                                                  recurrence_config=cfg["environment_model"]["architecture"]["recurrence"],
                                                  prediction_config=cfg["environment_model"]["architecture"]["traj_predictor"],
                                                  device=device)
loaded_environment_model.load_state_dict(torch.load(FDM_weight_path, map_location=device)['model_architecture_state_dict'])
loaded_environment_model.eval()
loaded_environment_model.to(device)
print("Loaded Forward Dynamics Model weight from {}\n".format(FDM_weight_path))

# Set action planner
n_prediction_step = int(cfg["Naive"]["planner"]["prediction_period"] / cfg['command_tracking']['command_period'])
evaluate_command_sampling_steps = int(cfg["Naive"]["planner"]["prediction_period"] / cfg['environment']['control_dt'])
action_planner = Stochastic_action_planner_normal(command_range=cfg["environment"]["command"],
                                                  n_sample=cfg["Naive"]["planner"]["number_of_sample"],
                                                  n_horizon=n_prediction_step,
                                                  sigma=cfg["Naive"]["planner"]["sigma"],
                                                  beta=cfg["Naive"]["planner"]["beta"],
                                                  gamma=cfg["Naive"]["planner"]["gamma"],
                                                  noise_sigma=0.1,
                                                  noise=False,
                                                  action_dim=user_command_dim)

# MUST safe period from collision
MUST_safety_period = 3.0
MUST_safety_period_n_steps = int(MUST_safety_period / cfg['command_tracking']['command_period'])

# Set constant
collision_threshold = 0.3
num_test_case_per_env = 10
if validation:
    init_seed = cfg["environment"]["seed"]["validate"]
    print("Validating ...")
else:
    init_seed = cfg["environment"]["seed"]["evaluate"]
    print("Evaluating ...")

# Make directory to save results
num_sample = cfg["Naive"]["planner"]["number_of_sample"]

cfg["environment"]["n_evaluate_envs"] = 5

print("<<-- Evaluating Safety Remote Control -->>")

pdb.set_trace()

for grid_size in [2.5, 3., 4., 5.]:
    print("===========================================")
    print(f"Grid_{str(grid_size)}:")
    print("===========================================")

    # Set obstacle grid size
    cfg["environment"]["test_obstacle_grid_size"] = grid_size

    # Set empty container to log result
    num_total_test_case = cfg["environment"]["n_evaluate_envs"] * num_test_case_per_env

    for env_id in range(cfg["environment"]["n_evaluate_envs"]):
        # Generate new environment with different seed (reset is automatically called)
        cfg["environment"]["seed"]["evaluate"] = env_id * 10 + init_seed
        env = VecEnv(test.RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper)), cfg['environment'], normalize_ob=False)
        env.load_scaling(command_tracking_weight_dir, int(iteration_number))

        # Initialize
        sample_user_command = np.zeros(3)

        for test_case_id in range(num_test_case_per_env):
            desired_command = user_command.uniform_sample_evaluate()[0, :]
            sample_user_command = desired_command.copy()

            true_collision = False

            # Reset
            env.initialize_n_step()
            env.reset()
            action_planner.reset()
            COM_buffer.reset()

            # Test without safety controller
            print("W/O SAFETY CONTROLLER")
            for step in range(evaluate_command_sampling_steps * 2):
                frame_start = time.time()
                new_action_time = step % command_period_steps == 0

                # observation before taking step
                obs, _ = env.observe(False)

                # update COM feature
                if step % COM_history_update_period == 0:
                    COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                    COM_buffer.update(COM_feature)

                if new_action_time:
                    # sample command sequences
                    action_candidates = action_planner.sample(desired_command)
                    action_candidates = np.swapaxes(action_candidates, 0, 1)
                    action_candidates = action_candidates.astype(np.float32)

                    # prepare state
                    init_coordinate_obs = env.coordinate_observe()
                    lidar_data = obs[0, proprioceptive_sensor_dim:]
                    COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
                    state = np.tile(np.concatenate((lidar_data, COM_history_feature)), (cfg["Naive"]["planner"]["number_of_sample"], 1))
                    state = state.astype(np.float32)

                    # simulate sampled command sequences
                    predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                       torch.from_numpy(action_candidates).to(device),
                                                                                       training=False)
                    desired_command_path = predicted_coordinates[:, 0, :]
                    predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                    reward = np.zeros(cfg["Naive"]["planner"]["number_of_sample"])

                    # visualize predicted desired command trajectory
                    w_coordinate_desired_command_path = transform_coordinate_LW(init_coordinate_obs, desired_command_path)
                    P_col_desired_command_path = predicted_P_cols[:, 0]
                    env.visualize_desired_command_traj(w_coordinate_desired_command_path,
                                                       P_col_desired_command_path,
                                                       collision_threshold)

                # Execute desired command
                tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
                tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
                tracking_obs = tracking_obs.astype(np.float32)

                with torch.no_grad():
                    tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))

                _, done = env.step(tracking_action.cpu().detach().numpy())

                # Check collision
                collision = env.single_env_collision_check()

                frame_end = time.time()

                if cfg["realistic"]:
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

                if collision:
                    true_collision = True
                    break

            # reset environment in same coordinate
            env.reset()
            action_planner.reset()
            COM_buffer.reset()
            

            modified_command_collision = False

            # Test with safety controller
            print("W/ SAFETY CONTROLLER")
            for step in range(evaluate_command_sampling_steps * 2):
                frame_start = time.time()
                new_action_time = step % command_period_steps == 0

                # observation before taking step
                obs, _ = env.observe(False)

                # update COM feature
                if step % COM_history_update_period == 0:
                    COM_feature = np.concatenate((obs[:, :3], obs[:, 15:21]), axis=1)
                    COM_buffer.update(COM_feature)

                if new_action_time:
                    # sample command sequences
                    action_candidates = action_planner.sample(desired_command)
                    action_candidates = np.swapaxes(action_candidates, 0, 1)
                    action_candidates = action_candidates.astype(np.float32)

                    # prepare state
                    init_coordinate_obs = env.coordinate_observe()
                    lidar_data = obs[0, proprioceptive_sensor_dim:]
                    COM_history_feature = COM_buffer.return_data(flatten=True)[0, :]
                    state = np.tile(np.concatenate((lidar_data, COM_history_feature)), (cfg["Naive"]["planner"]["number_of_sample"], 1))
                    state = state.astype(np.float32)

                    # simulate sampled command sequences
                    predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                       torch.from_numpy(action_candidates).to(device),
                                                                                       training=False)
                    desired_command_path = predicted_coordinates[:, 0, :]
                    predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                    reward = np.zeros(cfg["Naive"]["planner"]["number_of_sample"])

                    # Hard constraint for collision
                    for sample_id in range(num_sample):
                        current_done = np.where(predicted_P_cols[:, sample_id] > collision_threshold)[0]
                        if len(current_done) != 0:
                            done_idx = np.min(current_done)
                            predicted_coordinates[done_idx + 1:, sample_id, :] = predicted_coordinates[done_idx, sample_id, :]
                            predicted_P_cols[done_idx + 1:, sample_id] = predicted_P_cols[done_idx, sample_id]

                    # visualize predicted desired command trajectory
                    w_coordinate_desired_command_path = transform_coordinate_LW(init_coordinate_obs, desired_command_path)
                    P_col_desired_command_path = predicted_P_cols[:, 0]
                    env.visualize_desired_command_traj(w_coordinate_desired_command_path,
                                                       P_col_desired_command_path,
                                                       collision_threshold)

                    if len(np.where(predicted_P_cols[:MUST_safety_period_n_steps, 0] > collision_threshold)[0]) == 0:
                        # current desired command is safe
                        sample_user_command = action_planner.action(reward, safe=True)
                        action_planner.reset()
                    else:
                        # current desired command is not safe
                        safety_reward = 1 - predicted_P_cols
                        safety_reward = np.mean(safety_reward, axis=0)
                        safety_reward /= np.max(safety_reward) + 1e-5  # normalize reward
                        reward = safety_reward

                        # exclude trajectory that collides with obstacle
                        coll_idx = np.where(np.sum(np.where(predicted_P_cols[:MUST_safety_period_n_steps, :] > collision_threshold, 1, 0), axis=0) != 0)[0]
                        if len(coll_idx) != cfg["Naive"]["planner"]["number_of_sample"]:
                            reward[coll_idx] = 0

                        sample_user_command, sample_user_command_traj = action_planner.action(reward)

                        state = state[0, :][np.newaxis, :]
                        predicted_P_cols, predicted_coordinates = loaded_environment_model(torch.from_numpy(state).to(device),
                                                                                           torch.from_numpy(sample_user_command_traj[:, np.newaxis, :]).to(device),
                                                                                           training=False)

                        # Hard constraint for collision
                        predicted_P_cols = np.squeeze(predicted_P_cols, axis=-1)
                        current_done = np.where(predicted_P_cols[:, 0] > collision_threshold)[0]
                        if len(current_done) != 0:
                            done_idx = np.min(current_done)
                            predicted_coordinates[done_idx + 1:, 0, :] = predicted_coordinates[done_idx, 0, :]
                            predicted_P_cols[done_idx + 1:, 0] = predicted_P_cols[done_idx, 0]

                        # visualize predicted modified command trajectory
                        w_coordinate_modified_command_path = transform_coordinate_LW(init_coordinate_obs, predicted_coordinates[:, 0, :])
                        P_col_modified_command_path = predicted_P_cols[:, 0][:, np.newaxis]
                        env.visualize_modified_command_traj(w_coordinate_modified_command_path,
                                                            P_col_modified_command_path,
                                                            collision_threshold)

                # Execute desired command
                tracking_obs = np.concatenate((sample_user_command, obs[0, :proprioceptive_sensor_dim]))[np.newaxis, :]
                tracking_obs = env.force_normalize_observation(tracking_obs, type=1)
                tracking_obs = tracking_obs.astype(np.float32)

                with torch.no_grad():
                    tracking_action = command_tracking_policy.architecture(torch.from_numpy(tracking_obs).to(device))

                _, done = env.step(tracking_action.cpu().detach().numpy())

                # Check collision
                collision = env.single_env_collision_check()

                frame_end = time.time()

                if cfg["realistic"]:
                    wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                    if wait_time > 0.:
                        time.sleep(wait_time)

                if collision:
                    modified_command_collision = True
                    break

            if true_collision and modified_command_collision:
                print("collision fail")
            elif true_collision and not modified_command_collision:
                print("collision success")
            elif not true_collision and modified_command_collision:
                print("no collision fail")
            elif not true_collision and not modified_command_collision:
                print("no collision success")
            print('-----------------------------------')


env.turn_off_visualization()

