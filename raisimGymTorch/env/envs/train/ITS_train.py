import time
from ruamel.yaml import YAML
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from raisimGymTorch.env.envs.train.model import Informed_Trajectory_Sampler_training, Informed_Trajectory_Sampler_inference
from raisimGymTorch.env.envs.train.trainer import ITS_dataset, make_batch_all
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver
import wandb
import pdb
import os
import argparse
import numpy as np
import random

"""
Train Informed Trajectory Sampler (ITS)

[Objective]
- CVAE: average loss over latent samples
- BMS: minimum loss over latent samples
"""

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# task specification
task_name = "ITS_train"

parser = argparse.ArgumentParser()
parser.add_argument('-fw', '--fdm_weight', help='Forward Dynamics Model weight path', type=str, required=True)
args = parser.parse_args()

FDM_weight_path = args.fdm_weight

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# torch.backends.cudnn.benchmark = True

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# set numpy seed (crucial for creating train & validationm data)
seed = cfg["CVAE_training"]["command"]["seed"]
np.random.seed(seed)

task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../.."

# Create dataset
data_folder_path = f"{home_path}/analytic_planner_data"
data_file_list = os.listdir(data_folder_path)
n_data_files = len(data_file_list)
train_data_ratio = 0.9
n_train_data_files = int(n_data_files * train_data_ratio)
n_val_data_files = n_data_files - n_train_data_files
indices = np.random.permutation(n_data_files)
train_idx, validation_idx = indices[:n_train_data_files], indices[n_train_data_files:]
print("<--- Dataset size --->")
print(f"Train: {len(train_idx)} / Validation: {len(validation_idx)}")
print("----------------------")

required_date = {"observation": True, "goal_position": True, "command_traj": True, "waypoints": True}

training_set = ITS_dataset(
    data_file_list=data_file_list,
    file_idx_list=train_idx,
    folder_path=data_folder_path,
    required_data=required_date
)
training_generator = DataLoader(training_set,
                                batch_size=cfg["CVAE_training"]["command"]["batch_size"],
                                shuffle=cfg["CVAE_training"]["command"]["shuffle_batch"],
                                num_workers=cfg["CVAE_training"]["command"]["num_workers"],
                                drop_last=True,
                                collate_fn=make_batch_all)

validation_set = ITS_dataset(
    data_file_list=data_file_list,
    file_idx_list=validation_idx,
    folder_path=data_folder_path,
    required_data=required_date
)
validation_generator = DataLoader(validation_set,
                                  batch_size=cfg["CVAE_training"]["command"]["batch_size"],
                                  shuffle=cfg["CVAE_training"]["command"]["shuffle_batch"],
                                  num_workers=cfg["CVAE_training"]["command"]["num_workers"],
                                  drop_last=True,
                                  collate_fn=make_batch_all)

# Create CVAE training model
cvae_train_model = Informed_Trajectory_Sampler_training(
    state_encoding_config=cfg["CVAE_architecture"]["command"]["state_encoder"],
    waypoint_encoding_config=cfg["CVAE_architecture"]["command"]["waypoint_encoder"],
    command_encoding_config=cfg["CVAE_architecture"]["command"]["command_encoder"],
    waypoint_recurrence_encoding_config=cfg["CVAE_architecture"]["command"]["waypoint_recurrence_encoder"],
    command_recurrence_encoding_config=cfg["CVAE_architecture"]["command"]["command_recurrence_encoder"],
    latent_encoding_config=cfg["CVAE_architecture"]["command"]["latent_encoder"],
    latent_decoding_config=cfg["CVAE_architecture"]["command"]["latent_decoder"],
    recurrence_decoding_config=cfg["CVAE_architecture"]["command"]["recurrence_decoder"],
    command_decoding_config=cfg["CVAE_architecture"]["command"]["command_decoder"],
    device=device,
    pretrained_weight=FDM_weight_path,
    n_latent_sample=cfg["CVAE_training"]["command"]["n_latent_sample"]
)
cvae_train_model.to(device)
n_latent_sample = cfg["CVAE_training"]["command"]["n_latent_sample"]
n_prediction_step = int(cfg["data_collection"]["prediction_period"] / cfg["data_collection"]["command_period"])
loss_weight = {"reconstruction": cfg["CVAE_training"]["command"]["loss_weight"]["reconsturction"],
               "KL_posterior": cfg["CVAE_training"]["command"]["loss_weight"]["KL_posterior"]}

optimizer = optim.Adam(filter(lambda p: p.requires_grad, cvae_train_model.parameters()), lr=cfg["CVAE_training"]["command"]["learning_rate"])

saver = ConfigurationSaver(log_dir=home_path + "/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"])

if cfg["logging"]:
    wandb.init(name=task_name, project="Quadruped_navigation")
    # wandb.watch(cvae_train_model, log='all', log_freq=150)

print("Ready to start.")
pdb.set_trace()

for epoch in range(cfg["CVAE_training"]["command"]["num_epochs"]):
    if epoch % cfg["CVAE_training"]["command"]["evaluate_period"] == 0:
        print("Evaluating the current CVAE model")
        torch.save({
            'model_architecture_state_dict': cvae_train_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(epoch)+'.pt')

        saved_model_weight = saver.data_dir+"/full_"+str(epoch)+'.pt'

        # Create CVAE evaluating model (create new graph)
        cvae_evaluate_model = Informed_Trajectory_Sampler_training(
            state_encoding_config=cfg["CVAE_architecture"]["command"]["state_encoder"],
            waypoint_encoding_config=cfg["CVAE_architecture"]["command"]["waypoint_encoder"],
            command_encoding_config=cfg["CVAE_architecture"]["command"]["command_encoder"],
            waypoint_recurrence_encoding_config=cfg["CVAE_architecture"]["command"]["waypoint_recurrence_encoder"],
            command_recurrence_encoding_config=cfg["CVAE_architecture"]["command"]["command_recurrence_encoder"],
            latent_encoding_config=cfg["CVAE_architecture"]["command"]["latent_encoder"],
            latent_decoding_config=cfg["CVAE_architecture"]["command"]["latent_decoder"],
            recurrence_decoding_config=cfg["CVAE_architecture"]["command"]["recurrence_decoder"],
            command_decoding_config=cfg["CVAE_architecture"]["command"]["command_decoder"],
            device=device,
            pretrained_weight=FDM_weight_path,
            n_latent_sample=cfg["CVAE_training"]["command"]["n_latent_sample"]
        )
        cvae_evaluate_model.load_state_dict(torch.load(saved_model_weight, map_location=device)['model_architecture_state_dict'])
        cvae_evaluate_model.eval()
        cvae_evaluate_model.to(device)

        # Create CVAE inference model
        cvae_inference_model = Informed_Trajectory_Sampler_inference(
            latent_dim=cfg["CVAE_architecture"]["command"]["latent_encoder"]["output"],
            state_encoding_config=cfg["CVAE_architecture"]["command"]["state_encoder"],
            waypoint_encoding_config=cfg["CVAE_architecture"]["command"]["waypoint_encoder"],
            waypoint_recurrence_encoding_config=cfg["CVAE_architecture"]["command"]["waypoint_recurrence_encoder"],
            latent_decoding_config=cfg["CVAE_architecture"]["command"]["latent_decoder"],
            recurrence_decoding_config=cfg["CVAE_architecture"]["command"]["recurrence_decoder"],
            command_decoding_config=cfg["CVAE_architecture"]["command"]["command_decoder"],
            device=device,
            trained_weight=saved_model_weight,
            cfg_command=cfg["environment"]["command"]
        )
        cvae_inference_model.eval()
        cvae_inference_model.to(device)

        mean_loss = 0
        mean_reconstruction_loss = 0
        mean_KL_posterior_loss = 0
        mean_inference_reconstruction_loss = 0
        mean_min_inference_reconstruction_loss = 0
        n_update = 0

        for observation_batch, _, command_traj_batch, waypoints_batch, waypoints_length_batch in validation_generator:
            observation_batch = observation_batch.to(device)
            command_traj_batch = torch.swapaxes(command_traj_batch, 0, 1)
            command_traj_batch = command_traj_batch.to(device)  #(traj_len, b, single_command_dim)
            waypoints_batch = torch.swapaxes(waypoints_batch, 0, 1)
            waypoints_batch = waypoints_batch.to(device)  #(traj_len, batch_size, waypoint_dim)

            # Model forward computation
            with torch.no_grad():
                latent_mean, latent_log_var, sampled_command_traj = cvae_evaluate_model(
                    observation_batch,
                    waypoints_batch,
                    waypoints_length_batch,
                    command_traj_batch
                )
                inference_sampled_command_traj = cvae_inference_model(
                    observation_batch,
                    waypoints_batch,
                    cfg["CVAE_inference"]["n_sample"],
                    n_prediction_step,
                    True,  # return_torch
                    waypoints_length_batch
                )

            # Compute loss
            if n_latent_sample == 1:
                reconstruction_loss = torch.sum(torch.sum((sampled_command_traj - command_traj_batch).pow(2), dim=0), dim=-1)
            else:
                command_traj_batch_broadcast = torch.broadcast_to(command_traj_batch.unsqueeze(2),
                                                                  (command_traj_batch.shape[0], command_traj_batch.shape[1], n_latent_sample, command_traj_batch.shape[2]))
                if cfg["CVAE_training"]["command"]["objective_type"] == "CVAE":
                    reconstruction_loss = torch.mean(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)
                elif cfg["CVAE_training"]["command"]["objective_type"] == "BMS":
                    reconstruction_loss = torch.min(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)[0]  # log(n_latent_sample) can be ignored
                else:
                    raise ValueError("Unsupported loss type")

            reconstruction_loss = reconstruction_loss.mean()  # average over batch size
            KL_posterior_loss = 0.5 * (torch.sum(latent_mean.pow(2) + latent_log_var.exp() - latent_log_var - 1, dim=-1))
            KL_posterior_loss = KL_posterior_loss.mean()
            loss = reconstruction_loss * loss_weight["reconstruction"] + KL_posterior_loss * loss_weight["KL_posterior"]

            command_traj_batch_broadcast = torch.broadcast_to(command_traj_batch.unsqueeze(2),
                                                            (command_traj_batch.shape[0], command_traj_batch.shape[1], cfg["CVAE_inference"]["n_sample"], command_traj_batch.shape[2]))
            inference_reconstruction_loss = torch.mean(torch.sum(torch.sum((inference_sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1).mean()
            min_inference_reconstruction_loss = torch.min(torch.sum(torch.sum((inference_sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)[0].mean()

            mean_loss += loss.item()
            mean_reconstruction_loss += reconstruction_loss.item()
            mean_KL_posterior_loss += KL_posterior_loss.item()
            mean_inference_reconstruction_loss += inference_reconstruction_loss.item()
            mean_min_inference_reconstruction_loss += min_inference_reconstruction_loss.item()
            n_update += 1

        mean_loss /= n_update
        mean_reconstruction_loss /= n_update
        mean_KL_posterior_loss /= n_update
        mean_inference_reconstruction_loss /= n_update
        mean_min_inference_reconstruction_loss /= n_update

        if cfg["logging"]:
            # Log data
            logging_data = dict()
            logging_data['Evaluate/Total'] = mean_loss
            logging_data['Evaluate/Reconstruction'] = mean_reconstruction_loss
            logging_data['Evaluate/KL_posterior'] = mean_KL_posterior_loss
            logging_data['Evaluate/Inference_reconstruction'] = mean_inference_reconstruction_loss
            logging_data['Evaluate/Minimum_inference_reconstruction'] = mean_min_inference_reconstruction_loss
            wandb.log(logging_data)

        print('====================================================')
        print('{:>6}th evaluation'.format(epoch))
        print('{:<40} {:>6}'.format("total: ", '{:0.6f}'.format(mean_loss)))
        print('{:<40} {:>6}'.format("reconstruction: ", '{:0.6f}'.format(mean_reconstruction_loss)))
        print('{:<40} {:>6}'.format("kl posterior: ", '{:0.6f}'.format(mean_KL_posterior_loss)))
        print('{:<40} {:>6}'.format("inference reconstruction: ", '{:0.6f}'.format(mean_inference_reconstruction_loss)))
        print('{:<40} {:>6}'.format("minimum inference reconstruction: ", '{:0.6f}'.format(mean_min_inference_reconstruction_loss)))

        print('====================================================\n')

    epoch_start = time.time()

    mean_loss = 0
    mean_reconstruction_loss = 0
    mean_KL_posterior_loss = 0
    n_update = 0

    for observation_batch, _, command_traj_batch, waypoints_batch, waypoints_length_batch in training_generator:
        observation_batch = observation_batch.to(device)
        command_traj_batch = torch.swapaxes(command_traj_batch, 0, 1)
        command_traj_batch = command_traj_batch.to(device)  #(traj_len, b, single_command_dim)
        waypoints_batch = torch.swapaxes(waypoints_batch, 0, 1)
        waypoints_batch = waypoints_batch.to(device)  #(traj_len, batch_size, waypoint_dim)

        # Model forward computation
        latent_mean, latent_log_var, sampled_command_traj = cvae_train_model(
            observation_batch,
            waypoints_batch,
            waypoints_length_batch,
            command_traj_batch
        )

        # Compute loss
        if n_latent_sample == 1:
            reconstruction_loss = torch.sum(torch.sum((sampled_command_traj - command_traj_batch).pow(2), dim=0), dim=-1)
        else:
            command_traj_batch_broadcast = torch.broadcast_to(command_traj_batch.unsqueeze(2),
                                                              (command_traj_batch.shape[0], command_traj_batch.shape[1], n_latent_sample, command_traj_batch.shape[2]))
            if cfg["CVAE_training"]["command"]["objective_type"] == "CVAE":
                reconstruction_loss = torch.mean(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)
            elif cfg["CVAE_training"]["command"]["objective_type"] == "BMS":
                reconstruction_loss = torch.min(torch.sum(torch.sum((sampled_command_traj - command_traj_batch_broadcast).pow(2), dim=0), dim=-1), dim=1)[0]  # log(n_latent_sample) can be ignored
            else:
                raise ValueError("Unsupported loss type")
        reconstruction_loss = reconstruction_loss.mean()
        KL_posterior_loss = 0.5 * (torch.sum(latent_mean.pow(2) + latent_log_var.exp() - latent_log_var - 1, dim=-1))
        KL_posterior_loss = KL_posterior_loss.mean()
        loss = reconstruction_loss * loss_weight["reconstruction"] + KL_posterior_loss * loss_weight["KL_posterior"]

        # Gradient step
        optimizer.zero_grad()
        loss.backward()
        if cfg["CVAE_training"]["command"]["clip_gradient"]:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, cvae_train_model.parameters()), cfg["CVAE_training"]["command"]["max_gradient_norm"])
        optimizer.step()

        mean_loss += loss.item()
        mean_reconstruction_loss += reconstruction_loss.item()
        mean_KL_posterior_loss += KL_posterior_loss.item()
        n_update += 1

    mean_loss /= n_update
    mean_reconstruction_loss /= n_update
    mean_KL_posterior_loss /= n_update

    if cfg["logging"]:
        # Log data
        logging_data = dict()
        logging_data['Loss/Total'] = mean_loss
        logging_data['Loss/Reconstruction'] = mean_reconstruction_loss
        logging_data['Loss/KL_posterior'] = mean_KL_posterior_loss
        wandb.log(logging_data)

    epoch_end = time.time()
    elapse_time_seconds = epoch_end - epoch_start
    elaspe_time_minutes = int(elapse_time_seconds / 60)
    elapse_time_seconds -= (elaspe_time_minutes * 60)
    elapse_time_seconds = int(elapse_time_seconds)

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(epoch))
    print('{:<40} {:>6}'.format("total: ", '{:0.6f}'.format(mean_loss)))
    print('{:<40} {:>6}'.format("reconstruction: ", '{:0.6f}'.format(mean_reconstruction_loss)))
    print('{:<40} {:>6}'.format("kl posterior: ", '{:0.6f}'.format(mean_KL_posterior_loss)))
    print(f'Time: {elaspe_time_minutes}m {elapse_time_seconds}s')
    print('----------------------------------------------------\n')




