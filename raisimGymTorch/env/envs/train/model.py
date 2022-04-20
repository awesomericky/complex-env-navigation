import torch.nn as nn
import numpy as np
import torch


class Forward_Dynamics_Model(nn.Module):
    def __init__(self,
                 state_encoding_config,
                 command_encoding_config,
                 recurrence_config,
                 prediction_config,
                 device,
                 cvae_retrain=False):
        super(Forward_Dynamics_Model, self).__init__()
        
        self.state_encoding_config = state_encoding_config
        self.command_encoding_config = command_encoding_config
        self.recurrence_config = recurrence_config
        self.prediction_config = prediction_config
        self.device = device
        self.cvae_retrain = cvae_retrain
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}

        assert self.state_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.prediction_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        self.set_module()

    def set_module(self):
        self.state_encoder = MLP(self.state_encoding_config["shape"],
                                 self.activation_map[self.state_encoding_config["activation"]],
                                 self.state_encoding_config["input"],
                                 self.state_encoding_config["output"],
                                 dropout=self.state_encoding_config["dropout"],
                                 batchnorm=self.state_encoding_config["batchnorm"])
        self.command_encoder = MLP(self.command_encoding_config["shape"],
                                   self.activation_map[self.command_encoding_config["activation"]],
                                   self.command_encoding_config["input"],
                                   self.command_encoding_config["output"],
                                   dropout=self.command_encoding_config["dropout"],
                                   batchnorm=self.command_encoding_config["batchnorm"])
        self.recurrence = torch.nn.LSTM(self.recurrence_config["input"],
                                        self.recurrence_config["hidden"],
                                        self.recurrence_config["layer"],
                                        dropout=self.recurrence_config["dropout"])
        self.Pcol_prediction = MLP(self.prediction_config["shape"],
                                   self.activation_map[self.prediction_config["activation"]],
                                   self.prediction_config["input"],
                                   self.prediction_config["collision"]["output"],
                                   dropout=self.prediction_config["dropout"],
                                   batchnorm=self.prediction_config["batchnorm"])
        self.coordinate_prediction = MLP(self.prediction_config["shape"],
                                         self.activation_map[self.prediction_config["activation"]],
                                         self.prediction_config["input"],
                                         self.prediction_config["coordinate"]["output"],
                                         dropout=self.prediction_config["dropout"],
                                         batchnorm=self.prediction_config["batchnorm"])
        self.sigmoid = nn.Sigmoid()

    def forward(self, *args, training=False):
        """

        :return:
            p_col: (traj_len, n_sample, 1)
            coordinate: (traj_len, n_sample, 2)
        """

        """
        :param state: (n_sample, state_dim)
        :param command_traj: (traj_len, n_sample, single_command_dim)
        """
        state, command_traj = args

        state = state.contiguous()
        command_traj = command_traj.contiguous()

        if self.cvae_retrain:
            encoded_state = self.state_encoder.architecture(state).detach()
        else:
            encoded_state = self.state_encoder.architecture(state)
        initial_cell_state = torch.broadcast_to(encoded_state, (self.recurrence_config["layer"], *encoded_state.shape)).contiguous()
        initial_hidden_state = torch.zeros_like(initial_cell_state).to(self.device)

        traj_len, n_sample, single_command_dim = command_traj.shape
        command_traj = command_traj.view(-1, single_command_dim)
        if self.cvae_retrain:
            encoded_command = self.command_encoder.architecture(command_traj).view(traj_len, n_sample, -1).detach()
        else:
            encoded_command = self.command_encoder.architecture(command_traj).view(traj_len, n_sample, -1)

        encoded_prediction, (_, _) = self.recurrence(encoded_command, (initial_hidden_state, initial_cell_state))
        traj_len, n_sample, encoded_prediction_dim = encoded_prediction.shape
        encoded_prediction = encoded_prediction.view(-1, encoded_prediction_dim)
        collision_prob_traj = self.sigmoid(self.Pcol_prediction.architecture(encoded_prediction))
        collision_prob_traj = collision_prob_traj.view(traj_len, n_sample, self.prediction_config["collision"]["output"])
         
        # coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        # coordinate_traj = coordinate_traj.view(traj_len, n_sample, self.prediction_config["coordinate"]["output"])

        delata_coordinate_traj = self.coordinate_prediction.architecture(encoded_prediction)
        delata_coordinate_traj = delata_coordinate_traj.view(traj_len, n_sample, self.prediction_config["coordinate"]["output"])

        coordinate_traj = torch.zeros(traj_len, n_sample, self.prediction_config["coordinate"]["output"]).to(self.device)
        for i in range(traj_len):
            if i == 0:
                coordinate_traj[i, :, :] = delata_coordinate_traj[i, :, :]
            else:
                coordinate_traj[i, :, :] = coordinate_traj[i - 1, :, :] + delata_coordinate_traj[i, :, :]

        if training:
            # return "device" torch tensor
            return collision_prob_traj, coordinate_traj
        else:
            # return "cpu" numpy tensor
            return collision_prob_traj.cpu().detach().numpy(), coordinate_traj.cpu().detach().numpy()

#######################################################################################################################

class Informed_Trajectory_Sampler_training(nn.Module):
    """
    P(Y | X)

    X: observation, waypoints
    Y: command_traj
    """
    def __init__(self,
                 state_encoding_config,
                 waypoint_encoding_config,
                 command_encoding_config,
                 waypoint_recurrence_encoding_config,
                 command_recurrence_encoding_config,
                 latent_encoding_config,
                 latent_decoding_config,
                 recurrence_decoding_config,
                 command_decoding_config,
                 device,
                 pretrained_weight,  # state_encoder, command_encoder
                 waypoint_encoder_pretrained_weight=None,  # waypoint_encoder
                 n_latent_sample=1):
        super(Informed_Trajectory_Sampler_training, self).__init__()

        self.state_encoding_config = state_encoding_config
        self.waypoint_encoding_config = waypoint_encoding_config
        self.command_encoding_config = command_encoding_config
        self.waypoint_recurrence_encoding_config = waypoint_recurrence_encoding_config
        self.command_recurrence_encoding_config = command_recurrence_encoding_config
        self.latent_encoding_config = latent_encoding_config
        self.latent_decoding_config = latent_decoding_config
        self.recurrence_decoding_config = recurrence_decoding_config
        self.command_decoding_config = command_decoding_config
        self.device = device
        self.pretrained_weight = torch.load(pretrained_weight, map_location=self.device)["model_architecture_state_dict"]
        if waypoint_encoder_pretrained_weight is not None:
            self.waypoint_encoder_pretrained_weight = torch.load(waypoint_encoder_pretrained_weight, map_location=self.device)["model_architecture_state_dict"]
        else:
            self.waypoint_encoder_pretrained_weight = None
        self.n_latent_sample = n_latent_sample
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}
        self.latent_dim = self.latent_encoding_config["output"]

        assert self.state_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.waypoint_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.latent_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.latent_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        assert self.recurrence_decoding_config["input"] == self.recurrence_decoding_config["hidden"], "Input and output size of GRU should be equal."

        self.set_module()

    def set_module(self):
        self.state_encoder = MLP(self.state_encoding_config["shape"],
                                 self.activation_map[self.state_encoding_config["activation"]],
                                 self.state_encoding_config["input"],
                                 self.state_encoding_config["output"],
                                 dropout=self.state_encoding_config["dropout"],
                                 batchnorm=self.state_encoding_config["batchnorm"])
        self.waypoint_encoder = MLP(self.waypoint_encoding_config["shape"],
                                    self.activation_map[self.waypoint_encoding_config["activation"]],
                                    self.waypoint_encoding_config["input"],
                                    self.waypoint_encoding_config["output"],
                                    dropout=self.waypoint_encoding_config["dropout"],
                                    batchnorm=self.waypoint_encoding_config["batchnorm"])
        self.command_encoder = MLP(self.command_encoding_config["shape"],
                                   self.activation_map[self.command_encoding_config["activation"]],
                                   self.command_encoding_config["input"],
                                   self.command_encoding_config["output"],
                                   dropout=self.command_encoding_config["dropout"],
                                   batchnorm=self.command_encoding_config["batchnorm"])
        self.waypoint_recurrence_encoder = torch.nn.GRU(self.waypoint_recurrence_encoding_config["input"],
                                                        self.waypoint_recurrence_encoding_config["hidden"],
                                                        self.waypoint_recurrence_encoding_config["layer"],
                                                        dropout=self.waypoint_recurrence_encoding_config["dropout"])
        self.command_recurrence_encoder = torch.nn.GRU(self.command_recurrence_encoding_config["input"],
                                                       self.command_recurrence_encoding_config["hidden"],
                                                       self.command_recurrence_encoding_config["layer"],
                                                       dropout=self.command_recurrence_encoding_config["dropout"])
        self.latent_mean_encoder = MLP(self.latent_encoding_config["shape"],
                                       self.activation_map[self.latent_encoding_config["activation"]],
                                       self.latent_encoding_config["input"],
                                       self.latent_encoding_config["output"],
                                       dropout=self.latent_encoding_config["dropout"],
                                       batchnorm=self.latent_encoding_config["batchnorm"])
        self.latent_log_var_encoder = MLP(self.latent_encoding_config["shape"],
                                          self.activation_map[self.latent_encoding_config["activation"]],
                                          self.latent_encoding_config["input"],
                                          self.latent_encoding_config["output"],
                                          dropout=self.latent_encoding_config["dropout"],
                                          batchnorm=self.latent_encoding_config["batchnorm"])
        self.latent_decoder = MLP(self.latent_decoding_config["shape"],
                                  self.activation_map[self.latent_decoding_config["activation"]],
                                  self.latent_decoding_config["input"],
                                  self.latent_decoding_config["output"],
                                  dropout=self.latent_decoding_config["dropout"],
                                  batchnorm=self.latent_decoding_config["batchnorm"])
        self.recurrence_decoder = torch.nn.GRU(self.recurrence_decoding_config["input"],
                                               self.recurrence_decoding_config["hidden"],
                                               self.recurrence_decoding_config["layer"],
                                               dropout=self.recurrence_decoding_config["dropout"])
        self.command_decoder = MLP(self.command_decoding_config["shape"],
                                   self.activation_map[self.command_decoding_config["activation"]],
                                   self.command_decoding_config["input"],
                                   self.command_decoding_config["output"],
                                   dropout=self.command_decoding_config["dropout"],
                                   batchnorm=self.command_decoding_config["batchnorm"])

        # Prepare weights to be loaded
        pretrained_state_encoder_state_dict = dict()
        pretrained_command_encoder_state_dict = dict()
        for k, v in self.pretrained_weight.items():
            if k.split('.', 1)[0] == "state_encoder":
                pretrained_state_encoder_state_dict[k.split('.', 1)[1]] = v
            elif k.split('.', 1)[0] == "command_encoder":
                pretrained_command_encoder_state_dict[k.split('.', 1)[1]] = v
        assert len(pretrained_state_encoder_state_dict.keys()) != 0, "Error when loading weights"
        assert len(pretrained_command_encoder_state_dict.keys()) != 0, "Error when loading weights"

        if self.waypoint_encoder_pretrained_weight is not None:
            pretrained_waypoint_encoder_state_dict = dict()
            pretrained_waypoint_recurrence_encoder_state_dict = dict()
            for k, v in self.waypoint_encoder_pretrained_weight.items():
                if k.split('.', 1)[0] == "waypoint_encoder":
                    pretrained_waypoint_encoder_state_dict[k.split('.', 1)[1]] = v
                elif k.split('.', 1)[0] == "recurrence_encoder":
                    pretrained_waypoint_recurrence_encoder_state_dict[k.split('.', 1)[1]] = v
            assert len(pretrained_waypoint_encoder_state_dict.keys()) != 0, "Error when loading weights"
            assert len(pretrained_waypoint_recurrence_encoder_state_dict.keys()) != 0, "Error when loading weights"

        # Load pretrained weight and set mode for each part
        # load pretrained state encoder
        state_encoder_state_dict = self.state_encoder.state_dict()
        state_encoder_state_dict.update(pretrained_state_encoder_state_dict)
        self.state_encoder.load_state_dict(state_encoder_state_dict)
        self.state_encoder.eval()
        for param in self.state_encoder.parameters():
            param.requires_grad = False

        # load pretrained command encoder
        command_encoder_state_dict = self.command_encoder.state_dict()
        command_encoder_state_dict.update(pretrained_command_encoder_state_dict)
        self.command_encoder.load_state_dict(command_encoder_state_dict)
        self.command_encoder.eval()
        for param in self.command_encoder.parameters():
            param.requires_grad = False

        if self.waypoint_encoder_pretrained_weight is not None:
            # load pretrained waypoint encoder
            waypoint_encoder_state_dict = self.waypoint_encoder.state_dict()
            waypoint_encoder_state_dict.update(pretrained_waypoint_encoder_state_dict)
            self.waypoint_encoder.load_state_dict(waypoint_encoder_state_dict)
            self.waypoint_encoder.eval()
            for param in self.waypoint_encoder.parameters():
                param.requires_grad = False

            # load pretrained waypoint recurrence encoder
            waypoint_recurrence_encoder_state_dict = self.waypoint_recurrence_encoder.state_dict()
            waypoint_recurrence_encoder_state_dict.update(pretrained_waypoint_recurrence_encoder_state_dict)
            self.waypoint_recurrence_encoder.load_state_dict(waypoint_recurrence_encoder_state_dict)
            self.waypoint_recurrence_encoder.eval()
            for param in self.waypoint_recurrence_encoder.parameters():
                param.requires_grad = False
        else:
            self.waypoint_encoder.train()
            self.waypoint_recurrence_encoder.train()

        self.command_recurrence_encoder.train()
        self.latent_mean_encoder.train()
        self.latent_log_var_encoder.train()
        self.latent_decoder.train()
        self.recurrence_decoder.train()
        self.command_decoder.train()

    def forward(self, state, waypoints, waypoints_length, command_traj, return_torch=True):
        """

        :param state: (batch_size, state_dim)
        :param waypoints: (traj_len, batch_size, waypoint_dim)
        :param waypoints_length: (batch_size,)
        :param command_traj: (traj_len, batch_size, command_dim)

        :return:
        latent_mean: (batch_size, latent_dim)
        latent_log_var: (batch_size, latent_dim)

        if self.n_latent_sample == 1:
            sampled_command_traj: (traj_len, batch_size, command_dim)
        else:
            sampled_command_traj: (traj_len, batch_size, self.n_latent_sample, command_dim)
        """
        state = state.contiguous()
        waypoints = waypoints.contiguous()
        command_traj = command_traj.contiguous()

        # state encoding
        with torch.no_grad():
            encoded_state = self.state_encoder.architecture(state)

        # waypoint encoding
        waypoint_traj_len, n_sample, waypoint_dim = waypoints.shape
        waypoints = waypoints.view(-1, waypoint_dim)
        if self.waypoint_encoder_pretrained_weight is not None:
            with torch.no_grad():
                encoded_waypoints = self.waypoint_encoder.architecture(waypoints).view(waypoint_traj_len, n_sample, -1)
        else:
            encoded_waypoints = self.waypoint_encoder.architecture(waypoints).view(waypoint_traj_len, n_sample, -1)

        # command encoding
        traj_len, n_sample, single_command_dim = command_traj.shape
        command_traj = command_traj.view(-1, single_command_dim)
        with torch.no_grad():
            encoded_command = self.command_encoder.architecture(command_traj).view(traj_len, n_sample, -1)

        # waypoints encoding
        encoded_waypoints = nn.utils.rnn.pack_padded_sequence(encoded_waypoints, lengths=waypoints_length, batch_first=False, enforce_sorted=True)
        if self.waypoint_encoder_pretrained_weight is not None:
            with torch.no_grad():
                _, encoded_total_waypoints = self.waypoint_recurrence_encoder(encoded_waypoints)
        else:
            _, encoded_total_waypoints = self.waypoint_recurrence_encoder(encoded_waypoints)
        encoded_total_waypoints = encoded_total_waypoints.squeeze(0)

        # command trajectory encoding
        _, encoded_command_traj = self.command_recurrence_encoder(encoded_command)
        encoded_command_traj = encoded_command_traj.squeeze(0)

        # predict posterior distribution in latent space
        total_encoded_result = torch.cat((encoded_state, encoded_total_waypoints, encoded_command_traj), dim=1)  # (n_sample, encoded_dim)
        latent_mean = self.latent_mean_encoder.architecture(total_encoded_result)
        latent_log_var = self.latent_log_var_encoder.architecture(total_encoded_result)

        if self.n_latent_sample == 1:
            # sample with reparameterization trick
            latent_std = torch.exp(0.5 * latent_log_var)
            eps = torch.rand_like(latent_std)
            sample = latent_mean + (eps * latent_std)

            # decode command trajectory
            total_decoded_result = torch.cat((encoded_state, encoded_total_waypoints, sample), dim=1)
            hidden_state = self.latent_decoder.architecture(total_decoded_result).unsqueeze(0)
            decoded_traj = torch.zeros(traj_len, n_sample, self.recurrence_decoding_config["hidden"]).to(self.device)
            input_state = torch.zeros(1, n_sample, self.recurrence_decoding_config["input"]).to(self.device)

            for i in range(traj_len):
                output, hidden_state = self.recurrence_decoder(input_state, hidden_state)
                decoded_traj[i] = output.squeeze(0)
                input_state = output

            decoded_traj = decoded_traj.view(-1, self.recurrence_decoding_config["hidden"])
            sampled_command_traj = self.command_decoder.architecture(decoded_traj)
            sampled_command_traj = sampled_command_traj.view(traj_len, n_sample, -1)

            if return_torch:
                return latent_mean, latent_log_var, sampled_command_traj
            else:
                return latent_mean.cpu().detach().numpy(), latent_log_var.cpu().detach().numpy(), sampled_command_traj.cpu().detach().numpy()
        else:
            # sample with reparameterization trick
            latent_std = torch.exp(0.5 * latent_log_var)
            eps = torch.rand((n_sample, self.n_latent_sample, self.latent_dim)).to(self.device)
            sample = latent_mean.unsqueeze(1) + (eps * latent_std.unsqueeze(1))   # (n_sample, self.n_latent_sample, latent_dim)

            # decode command trajectory
            encoded_state = torch.broadcast_to(encoded_state.unsqueeze(1), (n_sample, self.n_latent_sample, encoded_state.shape[-1])).contiguous()
            encoded_total_waypoints = torch.broadcast_to(encoded_total_waypoints.unsqueeze(1), (n_sample, self.n_latent_sample, encoded_total_waypoints.shape[-1])).contiguous()
            total_decoded_result = torch.cat((encoded_state, encoded_total_waypoints, sample), dim=-1)
            total_decoded_result = total_decoded_result.view(n_sample * self.n_latent_sample, -1)
            hidden_state = self.latent_decoder.architecture(total_decoded_result).unsqueeze(0)
            decoded_traj = torch.zeros(traj_len, n_sample * self.n_latent_sample, self.recurrence_decoding_config["hidden"]).to(self.device)
            input_state = torch.zeros(1, n_sample * self.n_latent_sample, self.recurrence_decoding_config["input"]).to(self.device)

            for i in range(traj_len):
                output, hidden_state = self.recurrence_decoder(input_state, hidden_state)
                decoded_traj[i] = output.squeeze(0)
                input_state = output

            decoded_traj = decoded_traj.view(-1, self.recurrence_decoding_config["hidden"])
            sampled_command_traj = self.command_decoder.architecture(decoded_traj)
            sampled_command_traj = sampled_command_traj.view(traj_len, n_sample, self.n_latent_sample, -1)

            if return_torch:
                return latent_mean, latent_log_var, sampled_command_traj
            else:
                return latent_mean.cpu().detach().numpy(), latent_log_var.cpu().detach().numpy(), sampled_command_traj.cpu().detach().numpy()


class Informed_Trajectory_Sampler_inference(nn.Module):
    """
    P(Y | X)

    X: observation, waypoints
    Y: command_traj
    """
    def __init__(self,
                 latent_dim,
                 state_encoding_config,
                 waypoint_encoding_config,
                 waypoint_recurrence_encoding_config,
                 latent_decoding_config,
                 recurrence_decoding_config,
                 command_decoding_config,
                 device,
                 trained_weight,
                 cfg_command):
        super(Informed_Trajectory_Sampler_inference, self).__init__()

        self.state_encoding_config = state_encoding_config
        self.waypoint_encoding_config = waypoint_encoding_config
        self.waypoint_recurrence_encoding_config = waypoint_recurrence_encoding_config
        self.latent_decoding_config = latent_decoding_config
        self.recurrence_decoding_config = recurrence_decoding_config
        self.command_decoding_config = command_decoding_config
        self.device = device
        self.trained_weight = torch.load(trained_weight, map_location=self.device)["model_architecture_state_dict"]
        self.activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "leakyrelu": nn.LeakyReLU}
        self.latent_dim = latent_dim
        self.cfg_command = cfg_command

        assert self.state_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.waypoint_encoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.latent_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."
        assert self.command_decoding_config["activation"] in list(self.activation_map.keys()), "Unavailable activation."

        assert self.recurrence_decoding_config["input"] == self.recurrence_decoding_config["hidden"], "Input and output size of GRU should be equal."

        self.set_module()

    def set_module(self):
        self.state_encoder = MLP(self.state_encoding_config["shape"],
                                 self.activation_map[self.state_encoding_config["activation"]],
                                 self.state_encoding_config["input"],
                                 self.state_encoding_config["output"],
                                 dropout=self.state_encoding_config["dropout"],
                                 batchnorm=self.state_encoding_config["batchnorm"])
        self.waypoint_encoder = MLP(self.waypoint_encoding_config["shape"],
                                    self.activation_map[self.waypoint_encoding_config["activation"]],
                                    self.waypoint_encoding_config["input"],
                                    self.waypoint_encoding_config["output"],
                                    dropout=self.waypoint_encoding_config["dropout"],
                                    batchnorm=self.waypoint_encoding_config["batchnorm"])
        self.waypoint_recurrence_encoder = torch.nn.GRU(self.waypoint_recurrence_encoding_config["input"],
                                                        self.waypoint_recurrence_encoding_config["hidden"],
                                                        self.waypoint_recurrence_encoding_config["layer"],
                                                        dropout=self.waypoint_recurrence_encoding_config["dropout"])
        self.latent_decoder = MLP(self.latent_decoding_config["shape"],
                                  self.activation_map[self.latent_decoding_config["activation"]],
                                  self.latent_decoding_config["input"],
                                  self.latent_decoding_config["output"],
                                  dropout=self.latent_decoding_config["dropout"],
                                  batchnorm=self.latent_decoding_config["batchnorm"])
        self.recurrence_decoder = torch.nn.GRU(self.recurrence_decoding_config["input"],
                                               self.recurrence_decoding_config["hidden"],
                                               self.recurrence_decoding_config["layer"],
                                               dropout=self.recurrence_decoding_config["dropout"])
        self.command_decoder = MLP(self.command_decoding_config["shape"],
                                   self.activation_map[self.command_decoding_config["activation"]],
                                   self.command_decoding_config["input"],
                                   self.command_decoding_config["output"],
                                   dropout=self.command_decoding_config["dropout"],
                                   batchnorm=self.command_decoding_config["batchnorm"])

        # Prepare weights to be loaded
        trained_state_encoder_state_dict = dict()
        trained_waypoint_encoder_state_dict = dict()
        trained_waypoint_recurrence_encoder_state_dict = dict()
        trained_latent_decoder_state_dict = dict()
        trained_recurrence_decoder_state_dict = dict()
        trained_command_decoder_state_dict = dict()
        for k, v in self.trained_weight.items():
            if k.split('.', 1)[0] == "state_encoder":
                trained_state_encoder_state_dict[k.split('.', 1)[1]] = v
            elif k.split('.', 1)[0] == "waypoint_encoder":
                trained_waypoint_encoder_state_dict[k.split('.', 1)[1]] = v
            elif k.split('.', 1)[0] == "waypoint_recurrence_encoder":
                trained_waypoint_recurrence_encoder_state_dict[k.split('.', 1)[1]] = v
            elif k.split('.')[0] == "latent_decoder":
                trained_latent_decoder_state_dict[k.split('.', 1)[1]] = v
            elif k.split('.')[0] == "recurrence_decoder":
                trained_recurrence_decoder_state_dict[k.split('.', 1)[1]] = v
            elif k.split('.')[0] == "command_decoder":
                trained_command_decoder_state_dict[k.split('.', 1)[1]] = v
        assert len(trained_state_encoder_state_dict.keys()) != 0, "Error when loading weights"
        assert len(trained_waypoint_encoder_state_dict.keys()) != 0, "Error when loading weights"
        assert len(trained_waypoint_recurrence_encoder_state_dict.keys()) != 0, "Error when loading weights"
        assert len(trained_latent_decoder_state_dict.keys()) != 0, "Error when loading weights"
        assert len(trained_recurrence_decoder_state_dict.keys()) != 0, "Error when loading weights"
        assert len(trained_command_decoder_state_dict.keys()) != 0, "Error when loading weights"

        # load weight
        state_encoder_state_dict = self.state_encoder.state_dict()
        state_encoder_state_dict.update(trained_state_encoder_state_dict)
        self.state_encoder.load_state_dict(state_encoder_state_dict)
        self.state_encoder.eval()

        waypoint_encoder_state_dict = self.waypoint_encoder.state_dict()
        waypoint_encoder_state_dict.update(trained_waypoint_encoder_state_dict)
        self.waypoint_encoder.load_state_dict(waypoint_encoder_state_dict)
        self.waypoint_encoder.eval()

        waypoint_recurrence_encoder_state_dict = self.waypoint_recurrence_encoder.state_dict()
        waypoint_recurrence_encoder_state_dict.update(trained_waypoint_recurrence_encoder_state_dict)
        self.waypoint_recurrence_encoder.load_state_dict(waypoint_recurrence_encoder_state_dict)
        self.waypoint_recurrence_encoder.eval()

        latent_decoder_state_dict = self.latent_decoder.state_dict()
        latent_decoder_state_dict.update(trained_latent_decoder_state_dict)
        self.latent_decoder.load_state_dict(latent_decoder_state_dict)
        self.latent_decoder.eval()

        recurrence_decoder_state_dict = self.recurrence_decoder.state_dict()
        recurrence_decoder_state_dict.update(trained_recurrence_decoder_state_dict)
        self.recurrence_decoder.load_state_dict(recurrence_decoder_state_dict)
        self.recurrence_decoder.eval()

        command_decoder_state_dict = self.command_decoder.state_dict()
        command_decoder_state_dict.update(trained_command_decoder_state_dict)
        self.command_decoder.load_state_dict(command_decoder_state_dict)
        self.command_decoder.eval()

    def forward(self, state, waypoints, n_sample, traj_len, return_torch=False, waypoints_length=None):
        """
        :param state: (batch_size, state_dim)
        :param waypoints: (traj_len, batch_size, waypoint_dim)  # if batch_size != 1, it should be padded and descending sorted result
        :param n_sample: int
        :param traj_len: int
        :param waypoints_length: (batch_size,)

        :return:
        if batch_size == 1:
            sampled_command_traj: (traj_len, n_sample, command_dim)
        else:
            sampled_command_traj: (traj_len, batch_size, n_sample, command_dim)
        """
        state = state.contiguous()
        waypoints = waypoints.contiguous()

        batch_size = state.shape[0]

        with torch.no_grad():
            # state encoding
            encoded_state = self.state_encoder.architecture(state)

            # waypoint encoding
            waypoint_traj_len, _, waypoint_dim = waypoints.shape
            waypoints = waypoints.view(-1, waypoint_dim)
            encoded_waypoints = self.waypoint_encoder.architecture(waypoints).view(waypoint_traj_len, batch_size, -1)

            # sample
            sample = torch.rand((batch_size, n_sample, self.latent_dim)).to(self.device)

            # waypoints encoding
            if batch_size == 1:
                _, encoded_total_waypoints = self.waypoint_recurrence_encoder(encoded_waypoints)
            else:
                assert waypoints_length is not None, "Waypoint lengths are not given"
                encoded_waypoints = nn.utils.rnn.pack_padded_sequence(encoded_waypoints, lengths=waypoints_length, batch_first=False, enforce_sorted=True)
                _, encoded_total_waypoints = self.waypoint_recurrence_encoder(encoded_waypoints)
            encoded_total_waypoints = encoded_total_waypoints.squeeze(0)

            # decode command trajectory
            encoded_state = torch.broadcast_to(encoded_state.unsqueeze(1), (batch_size, n_sample, encoded_state.shape[-1])).contiguous()
            encoded_total_waypoints = torch.broadcast_to(encoded_total_waypoints.unsqueeze(1), (batch_size, n_sample, encoded_total_waypoints.shape[-1])).contiguous()

            total_decoded_result = torch.cat((encoded_state, encoded_total_waypoints, sample), dim=-1)
            total_decoded_result = total_decoded_result.view(batch_size * n_sample, -1)
            hidden_state = self.latent_decoder.architecture(total_decoded_result).unsqueeze(0)
            decoded_traj = torch.zeros(traj_len, batch_size * n_sample, self.recurrence_decoding_config["hidden"]).to(self.device)
            input_state = torch.zeros(1, batch_size * n_sample, self.recurrence_decoding_config["input"]).to(self.device)

            for i in range(traj_len):
                output, hidden_state = self.recurrence_decoder(input_state, hidden_state)
                decoded_traj[i] = output.squeeze(0)
                input_state = output

            decoded_traj = decoded_traj.view(-1, self.recurrence_decoding_config["hidden"])
            sampled_command_traj = self.command_decoder.architecture(decoded_traj)
            sampled_command_traj = sampled_command_traj.view(traj_len, batch_size, n_sample, -1)

            if batch_size == 1:
                sampled_command_traj = sampled_command_traj.squeeze(1)  # (traj_len, n_sample, command_dim)

            if return_torch:
                # pytorch clamping function with min, max as Tensor just works > 1.10.0
                sampled_command_traj = torch.clamp(sampled_command_traj,
                                                   torch.tensor([self.cfg_command["forward_vel"]["min"], self.cfg_command["lateral_vel"]["min"], self.cfg_command["yaw_rate"]["min"]], device=self.device),
                                                   torch.tensor([self.cfg_command["forward_vel"]["max"], self.cfg_command["lateral_vel"]["max"], self.cfg_command["yaw_rate"]["max"]], device=self.device))
            else:
                sampled_command_traj = sampled_command_traj.cpu().detach().numpy()
                sampled_command_traj = np.clip(sampled_command_traj,
                                               [self.cfg_command["forward_vel"]["min"], self.cfg_command["lateral_vel"]["min"], self.cfg_command["yaw_rate"]["min"]],
                                               [self.cfg_command["forward_vel"]["max"], self.cfg_command["lateral_vel"]["max"], self.cfg_command["yaw_rate"]["max"]])
                sampled_command_traj = sampled_command_traj.astype(np.float32)

            return sampled_command_traj

#######################################################################################################################

class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size, dropout=0.0, batchnorm=False):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            if batchnorm:
                modules.append(nn.BatchNorm1d(shape[idx+1]))
            modules.append(self.activation_fn())
            if dropout != 0.0:
                modules.append(nn.Dropout(dropout))
            scale.append(np.sqrt(2))

        modules.append(nn.Linear(shape[-1], output_size))
        self.architecture = nn.Sequential(*modules)
        scale.append(np.sqrt(2))

        self.init_weights(self.architecture, scale)
        self.input_shape = [input_size]
        self.output_shape = [output_size]

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]
