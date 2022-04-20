import torch
import numpy as np

class UserCommand:
    def __init__(self, cfg, n_envs):
        self.min_forward_vel = cfg['environment']['command']['forward_vel']['min']
        self.max_forward_vel = cfg['environment']['command']['forward_vel']['max']
        self.min_lateral_vel = cfg['environment']['command']['lateral_vel']['min']
        self.max_lateral_vel = cfg['environment']['command']['lateral_vel']['max']
        self.min_yaw_rate = cfg['environment']['command']['yaw_rate']['min']
        self.max_yaw_rate = cfg['environment']['command']['yaw_rate']['max']
        self.n_envs = n_envs

    def uniform_sample_train(self):
        forward_vel = np.random.uniform(low=self.min_forward_vel, high=self.max_forward_vel, size=self.n_envs)
        lateral_vel = np.random.uniform(low=self.min_lateral_vel, high=self.max_lateral_vel, size=self.n_envs)
        yaw_rate = np.random.uniform(low=self.min_yaw_rate, high=self.max_yaw_rate, size=self.n_envs)
        command = np.stack((forward_vel, lateral_vel, yaw_rate), axis=1)
        return np.ascontiguousarray(command).astype(np.float32)

    def uniform_sample_evaluate(self):
        forward_vel = np.random.uniform(low=self.min_forward_vel, high=self.max_forward_vel, size=1)
        lateral_vel = np.random.uniform(low=self.min_lateral_vel, high=self.max_lateral_vel, size=1)
        yaw_rate = np.random.uniform(low=self.min_yaw_rate, high=self.max_yaw_rate, size=1)
        command = np.stack((forward_vel, lateral_vel, yaw_rate), axis=1)
        return np.ascontiguousarray(np.broadcast_to(command, (self.n_envs, 3))).astype(np.float32)


class Constant_command_sampler:
    """
    Sample constant command:

    """
    def __init__(self, random_command_sampler):
        self.new_command = None
        self.random_command_sampler = random_command_sampler

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        if isinstance(self.new_command, type(None)):
            self.new_command = self.random_sample()
        return self.new_command

    def reset(self):
        self.new_command = None


class Linear_time_correlated_command_sampler:
    """
    Sample time correlated command (using soft update rule):

    Time coorelation factor is controlled with 'beta'.
    
    Larger 'beta' == Bigger time correlation
    """
    def __init__(self, random_command_sampler, beta=0.7):
        self.old_command = None
        self.new_command = None
        self.random_command_sampler = random_command_sampler
        self.max_beta = beta

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        self.new_command = self.random_sample()
        if isinstance(self.old_command, type(None)):
            modified_command = self.new_command
        else:
            opposite_beta = np.random.uniform(0, 1 - self.max_beta, (self.random_command_sampler.n_envs, 3))
            modified_command = self.old_command * (1- opposite_beta) + self.new_command * opposite_beta
        self.old_command = modified_command
        return np.ascontiguousarray(modified_command).astype(np.float32)

    def reset(self):
        self.old_command = None
        self.new_command = None


class Normal_time_correlated_command_sampler:
    """
    Sample time correlated command (using normal distribution):

    Time coorelation factor is controlled with 'sigma'.
    """
    def __init__(self, random_command_sampler, cfg_command, sigma=0.3, std_scale_fixed=False):
        self.old_command = None
        self.random_command_sampler = random_command_sampler
        self.cfg_command = cfg_command
        self.max_sigma = 0.5 * np.array([cfg_command["forward_vel"]["max"] - cfg_command["forward_vel"]["min"],
                                         cfg_command["lateral_vel"]["max"] - cfg_command["lateral_vel"]["min"],
                                         cfg_command["yaw_rate"]["max"] - cfg_command["yaw_rate"]["min"]])
        self.max_sigma_scale = sigma
        self.std_scale_fixed = std_scale_fixed

    def random_sample(self, training=True):
        if training:
            random_command = self.random_command_sampler.uniform_sample_train()
        else:
            random_command = self.random_command_sampler.uniform_sample_evaluate()
        return random_command

    def sample(self):
        if isinstance(self.old_command, type(None)):
            modified_command = self.random_sample()
        else:
            if self.std_scale_fixed:
                sigma_scale = self.max_sigma_scale
            else:
                sigma_scale = np.random.uniform(0, self.max_sigma_scale, (self.random_command_sampler.n_envs, 3))  # sample command std scale (uniform distribution)
            sigma = self.max_sigma * sigma_scale
            modified_command = np.random.normal(self.old_command, sigma)  # sample command (normal distribution)
            modified_command = np.clip(modified_command,
                                       [self.cfg_command["forward_vel"]["min"], self.cfg_command["lateral_vel"]["min"], self.cfg_command["yaw_rate"]["min"]],
                                       [self.cfg_command["forward_vel"]["max"], self.cfg_command["lateral_vel"]["max"], self.cfg_command["yaw_rate"]["max"]])
        self.old_command = modified_command
        return np.ascontiguousarray(modified_command).astype(np.float32)

    def reset(self):
        self.old_command = None


############################################################################################################

class Stochastic_action_planner_uniform_bin:
    def __init__(self, command_range, n_sample, n_horizon, n_bin, beta, gamma, sigma,
                 random_command_sampler, time_correlation_beta=0.1,
                 noise_sigma=0.1, noise=False, action_dim=3):
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.forward_vel_bin_size = (self.max_forward_vel - self.min_forward_vel) / n_bin
        self.forward_vel_bin_array = np.arange(self.min_forward_vel, self.max_forward_vel, self.forward_vel_bin_size)
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.lateral_vel_bin_size = (self.max_lateral_vel - self.min_lateral_vel) / n_bin
        self.lateral_vel_bin_array = np.arange(self.min_lateral_vel, self.max_lateral_vel, self.lateral_vel_bin_size)
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.yaw_rate_bin_size = (self.max_yaw_rate - self.min_yaw_rate) / n_bin
        self.yaw_rate_bin_array = np.arange(self.min_yaw_rate, self.max_yaw_rate, self.yaw_rate_bin_size)
        self.noise = noise

        self.max_sigma = 0.5 * np.array([self.max_forward_vel - self.min_forward_vel,
                                         self.max_lateral_vel - self.min_lateral_vel,
                                         self.max_yaw_rate - self.min_yaw_rate])

        self.n_sample = n_sample
        self.n_bin = n_bin
        self.beta = beta
        self.gamma = gamma
        self.max_sigma_scale = sigma
        self.n_horizon = n_horizon
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

        self.action_bottom_margin = np.zeros((self.n_sample, self.action_dim))
        for i in range(self.n_sample):
            forward_vel_idx = (i // (self.n_bin ** 2)) % self.n_bin
            lateral_vel_idx = (i // self.n_bin) % self.n_bin
            yaw_rate_idx = i % self.n_bin
            self.action_bottom_margin[i] = [self.forward_vel_bin_array[forward_vel_idx],
                                            self.lateral_vel_bin_array[lateral_vel_idx],
                                            self.yaw_rate_bin_array[yaw_rate_idx]]

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

        self.random_command_sampler = random_command_sampler
        self.time_correlation_beta = time_correlation_beta

    def sample(self, current_command=None):
        """

        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        self.a_tilda = np.random.uniform(0.0, 1.0, size=(self.n_sample, self.action_dim))
        self.a_tilda[:, 0] *= self.forward_vel_bin_size
        self.a_tilda[:, 1] *= self.lateral_vel_bin_size
        self.a_tilda[:, 2] *= self.yaw_rate_bin_size
        self.a_tilda += self.action_bottom_margin
        self.a_tilda = self.a_tilda[:, np.newaxis, :]

        self.a_tilda = np.broadcast_to(self.a_tilda, (self.n_sample, self.n_horizon, self.action_dim)).copy()

        if current_command is None:
            for i in range(1, self.n_horizon):
                # sigma_scale = np.random.uniform(0, self.max_sigma_scale, (self.random_command_sampler.n_envs, 3))  # sample command std scale (uniform distribution)
                sigma_scale = self.max_sigma_scale
                sigma = self.max_sigma * sigma_scale
                self.a_tilda[:, i, :] = np.random.normal(self.a_tilda[:, i-1, :], sigma)  # sample command (normal distribution)
                self.a_tilda[:, i, :] = np.clip(self.a_tilda[:, i, :],
                                                a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                                                a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])
        else:
            self.a_tilda[:, 0, :] = current_command

            for i in range(2, self.n_horizon):
                # sigma_scale = np.random.uniform(0, self.max_sigma_scale, (self.random_command_sampler.n_envs, 3))  # sample command std scale (uniform distribution)
                sigma_scale = self.max_sigma_scale
                sigma = self.max_sigma * sigma_scale
                self.a_tilda[:, i, :] = np.random.normal(self.a_tilda[:, i-1, :], sigma)  # sample command (normal distribution)
                self.a_tilda[:, i, :] = np.clip(self.a_tilda[:, i, :],
                                                a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                                                a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda = self.a_tilda * (1 - self.beta) + self.a_hat[np.newaxis, :, :] * self.beta

        # add noise
        if self.noise:
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            self.a_tilda[:, 1:, :] += noise_epsil

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def update(self, rewards):
        # self.a_hat = self.a_tilda[np.argmax(rewards)]
        safe_idx = np.where(rewards != 0)[0]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda[safe_idx, :, :] * probs[:, np.newaxis, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= np.sum(probs) + 1e-10
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        self.update(rewards)
        return self.a_hat[0], self.a_hat.astype(np.float32)


class Stochastic_action_planner_w_ITS:
    def __init__(self, wo_cvae_sampler, w_cvae_sampler, wo_cvae_n_sample, w_cvae_n_sample, n_prediction_step, gamma, beta):
        self.wo_cvae_sampler = wo_cvae_sampler
        self.w_cvae_sampler = w_cvae_sampler
        self.wo_cvae_n_sample = wo_cvae_n_sample
        self.w_cvae_n_sample = w_cvae_n_sample
        self.n_prediction_step = n_prediction_step
        self.gamma = gamma
        self.beta = beta
        self.sampled_command_traj = None
        self.optimized_command_traj = None
        self.first = True

    def sample(self, observation=None, goal_position=None, waypoints=None, current_command=None):
        """
        observation: (1, observation_dim)  type: torch.tensor
        goal_position: (1, goal_position_dim)  type: torch.tensor
        cf) current_command: (command_dim,)

        return:
            sampled_command_traj: (traj_len, self.wo_cvae_n_sample + self.w_cvae_n_sample, command_dim)  type: numpy.tensor

        * sampled_command_traj stacked order: (1) wo_cvae_sampling (2) w_cvae_sampling
        """
        # sample using w/o cvae sampler
        wo_cvae_sampled_command_traj = self.wo_cvae_sampler.sample(current_command)
        wo_cvae_sampled_command_traj = np.swapaxes(wo_cvae_sampled_command_traj, 0, 1)

        if self.w_cvae_n_sample != 0:
            # sample using w/ cvae sampler
            cvae_sampled_command_traj = self.w_cvae_sampler(observation, waypoints, self.w_cvae_n_sample, self.n_prediction_step, return_torch=False)

            if current_command is not None:
                cvae_sampled_command_traj[0, :, :] = current_command

            if self.first:
                self.first = False
            else:
                cvae_sampled_command_traj = cvae_sampled_command_traj * (1 - self.beta) + self.optimized_command_traj[:, np.newaxis, :] * self.beta
            self.sampled_command_traj = np.concatenate((wo_cvae_sampled_command_traj, cvae_sampled_command_traj), axis=1)

        else:
            self.sampled_command_traj = wo_cvae_sampled_command_traj

        return self.sampled_command_traj.astype(np.float32).copy()

    def reset(self):
        self.wo_cvae_sampler.reset()
        self.sampled_command_traj = None
        self.optimized_command_traj = None
        self.first = True

    def seperate_update(self, rewards):
        wo_cvae_rewards = rewards[:self.wo_cvae_n_sample]
        wo_cvae_safe_idx = np.where(wo_cvae_rewards != 0)[0]
        wo_cvae_sampled_command_traj = self.sampled_command_traj[:, :self.wo_cvae_n_sample, :]
        w_cvae_rewards = rewards[self.wo_cvae_n_sample:]
        w_cvae_safe_idx = np.where(w_cvae_rewards != 0)[0]
        w_cvae_sampled_command_traj = self.sampled_command_traj[:, self.wo_cvae_n_sample:, :]
        
        # optimize w/o CVAE command trajectory
        if len(wo_cvae_safe_idx) != 0:
            probs = np.exp(self.gamma * wo_cvae_rewards[wo_cvae_safe_idx])
            probs /= (np.sum(probs) + 1e-10)
            wo_cvae_optimized_command_traj = np.sum(wo_cvae_sampled_command_traj[:, wo_cvae_safe_idx, :] * probs[np.newaxis, :, np.newaxis], axis=1)
        else:
            probs = np.exp(self.gamma * wo_cvae_rewards)
            probs /= (np.sum(probs) + 1e-10)
            wo_cvae_optimized_command_traj = np.sum(wo_cvae_sampled_command_traj * probs[np.newaxis, :, np.newaxis], axis=1)

        # optimize w/ CVAE command trajectory
        if len(w_cvae_safe_idx) != 0:
            probs = np.exp(self.gamma * w_cvae_rewards[w_cvae_safe_idx])
            probs /= (np.sum(probs) + 1e-10)
            w_cvae_optimized_command_traj = np.sum(w_cvae_sampled_command_traj[:, w_cvae_safe_idx, :] * probs[np.newaxis, :, np.newaxis], axis=1)
        else:
            probs = np.exp(self.gamma * w_cvae_rewards)
            probs /= (np.sum(probs) + 1e-10)
            w_cvae_optimized_command_traj = np.sum(w_cvae_sampled_command_traj * probs[np.newaxis, :, np.newaxis], axis=1)

        return wo_cvae_optimized_command_traj.astype(np.float32), w_cvae_optimized_command_traj.astype(np.float32)

    def set_optimized_result(self, optimized_command_traj):
        self.wo_cvae_sampler.a_hat = optimized_command_traj   # update a_hat in wo_cvae_sampler
        self.optimized_command_traj = optimized_command_traj   # update a_hat in cvae_sampler

    def update(self, rewards):
        safe_idx = np.where(rewards != 0)[0]
        if len(safe_idx) != 0:
            # safe_a_tilde = self.sampled_command_traj[:, safe_idx, :]
            # self.optimized_command_traj = safe_a_tilde[:, np.argmax(rewards[safe_idx]), :]
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= (np.sum(probs) + 1e-10)
            self.optimized_command_traj = np.sum(self.sampled_command_traj[:, safe_idx, :] * probs[np.newaxis, :, np.newaxis], axis=1)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= (np.sum(probs) + 1e-10)
            self.optimized_command_traj = np.sum(self.sampled_command_traj * probs[np.newaxis, :, np.newaxis], axis=1)

    def action(self, rewards):
        """
        rewards: (self.wo_cvae_n_sample + self.w_cvae_n_sample,)
  
        return:
            optimized_current_command: (command_dim,)
            optimized_command_traj: (traj_len, command_dim)
        """

        self.update(rewards)
        self.wo_cvae_sampler.a_hat = self.optimized_command_traj   # update a_hat in wo_cvae_sampler
        return self.optimized_command_traj[0], self.optimized_command_traj.astype(np.float32)


class Stochastic_action_planner_normal:
    """
    Sample commands from normal distribution, where the mean value is user command
    """
    def __init__(self, command_range, n_sample, n_horizon, sigma, beta, gamma, noise_sigma=0.1, noise=True, action_dim=3):
        # Available command limit
        self.min_forward_vel = command_range["forward_vel"]["min"]
        self.max_forward_vel = command_range["forward_vel"]["max"]
        self.min_lateral_vel = command_range["lateral_vel"]["min"]
        self.max_lateral_vel = command_range["lateral_vel"]["max"]
        self.min_yaw_rate = command_range["yaw_rate"]["min"]
        self.max_yaw_rate = command_range["yaw_rate"]["max"]
        self.delta = 0.5 * np.array([self.max_forward_vel - self.min_forward_vel, self.max_lateral_vel - self.min_lateral_vel, self.max_yaw_rate - self.min_yaw_rate])
        self.noise = noise

        self.n_sample = n_sample
        self.n_horizon = n_horizon
        self.sigma = sigma
        self.gamma = gamma
        self.noise_sigma = noise_sigma
        self.action_dim = action_dim

        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.a_tilda = np.zeros((self.n_sample, self.n_horizon, self.action_dim))
        self.first = True
        self.beta = beta

    def sample(self, user_command):
        """

        :param user_command: (self.action_dim, )  type: numpy tensor
        :return: (self.n_sample, self.n_horizon, self.action_dim)  type: numpy tensor
        """
        epsil = np.random.normal(0.0, self.sigma, size=(self.n_sample - 1, self.action_dim))
        epsil = self.delta * epsil
        epsil = np.broadcast_to(epsil[:, np.newaxis, :], (self.n_sample - 1, self.n_horizon, self.action_dim)).copy()
        epsil = np.concatenate((np.zeros((1, self.n_horizon, self.action_dim)), epsil), axis=0)  # (self.n_sample, self.n_horizon, self.action_dim)

        if self.noise:
            # add extra noise along the command trajectory
            noise_epsil = np.random.normal(0.0, self.noise_sigma, size=(self.n_sample, self.n_horizon - 1, self.action_dim))
            epsil[:, 1:, :] += noise_epsil

        self.a_tilda = epsil + user_command

        # time correlated sampling
        if self.first:
            self.first = False
        else:
            self.a_tilda[1:, :, :] = self.a_tilda[1:, :, :] * (1 - self.beta) + self.a_hat[np.newaxis, :, :] * self.beta

        self.a_tilda = np.clip(self.a_tilda,
                               a_min=[self.min_forward_vel, self.min_lateral_vel, self.min_yaw_rate],
                               a_max=[self.max_forward_vel, self.max_lateral_vel, self.max_yaw_rate])

        return self.a_tilda.astype(np.float32).copy()

    def reset(self):
        self.a_hat = np.zeros((self.n_horizon, self.action_dim))
        self.first = True

    def update(self, rewards):
        safe_idx = np.where(rewards != 0)[0]
        # safe_idx = np.argpartition(rewards, -500)[-500:]
        if len(safe_idx) != 0:
            probs = np.exp(self.gamma * rewards[safe_idx])
            probs /= (np.sum(probs) + 1e-10)
            self.a_hat = np.sum(self.a_tilda[safe_idx, :, :] * probs[:, np.newaxis, np.newaxis], axis=0)
        else:
            probs = np.exp(self.gamma * rewards)
            probs /= (np.sum(probs) + 1e-10)
            self.a_hat = np.sum(self.a_tilda * probs[:, np.newaxis, np.newaxis], axis=0)

    def action(self, rewards, safe=False):
        """

        :param rewards: (self.n_sample,)  type: numpy
        :return:
        """
        if safe:
            return self.a_tilda[0, 0, :]
        self.update(rewards)
        if self.noise:
            noise_epsil = np.concatenate((np.zeros((1, self.action_dim)), np.random.normal(0.0, self.noise_sigma, size=(self.n_horizon - 1, self.action_dim))), axis=0)
            a_hat_traj = self.a_hat[0, :] + noise_epsil
        else:
            a_hat_traj = np.broadcast_to(self.a_hat[0], (self.n_horizon, self.action_dim)).copy()
        return self.a_hat[0], a_hat_traj.astype(np.float32)










