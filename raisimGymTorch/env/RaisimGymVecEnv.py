# //----------------------------//
# // This file is part of RaiSim//
# // Copyright 2020, RaiSim Tech//
# //----------------------------//
import pdb

import numpy as np
import platform
import os


class RaisimGymVecEnv:

    def __init__(self, impl, cfg, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        if platform.system() == "Darwin":
            os.environ['KMP_DUPLICATE_LIB_OK']='True'
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self.wrapper = impl
        self.num_obs = self.wrapper.getObDim()
        self.num_acts = self.wrapper.getActionDim()
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.coordinate_observation = np.zeros([self.num_envs, 3], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        self.obs_rms_second = None
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self._goal = np.zeros(2, dtype=np.float32)
        self._parallel_goal = np.zeros((self.num_envs, 2), dtype=np.float32)
        self._parallel_collision = np.zeros(self.num_envs, dtype=np.bool)
        # self.rewards = [[] for _ in range(self.num_envs)]

        self.hSize = self.wrapper.getDepthImageHSize()
        self.vSize = self.wrapper.getDepthImageVSize()
        self._depthImage_vectorized = np.zeros([self.num_envs, self.hSize*self.vSize], dtype=np.float32)

        try:
            self.reward_log = np.zeros([self.num_envs, cfg["n_rewards"] + 1], dtype=np.float32)
            self.reward_w_cpeff_log = np.zeros([self.num_envs, cfg["n_rewards"] + 1], dtype=np.float32)
        except:
            self.reward_log = None
            self.reward_w_cpeff_log = None
        self.contact_log = np.zeros([self.num_envs, 4], dtype=np.float32)
        self.torque_and_velocity_log = np.zeros([self.num_envs, 24], dtype=np.float32)

        self.potential_computed_heading_direction = np.zeros(2, dtype=np.float32)

    def seed(self, seed=None):
        self.wrapper.setSeed(seed)

    def turn_on_visualization(self):
        self.wrapper.turnOnVisualization()

    def turn_off_visualization(self):
        self.wrapper.turnOffVisualization()

    def start_video_recording(self, file_name):
        self.wrapper.startRecordingVideo(file_name)

    def stop_video_recording(self):
        self.wrapper.stopRecordingVideo()

    def step(self, action):
        self.wrapper.step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def partial_step(self, action):
        self.wrapper.partial_step(action, self._reward, self._done)
        return self._reward.copy(), self._done.copy()

    def load_scaling(self, dir_name, iteration, count=1e5, type=None):
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"

        if self.obs_rms_second == None:
            self.obs_rms.count = count
            self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
            self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)
        else:
            assert type in [1, 2], "Unavailable scaling type."
            if type == 1:
                # Collision avoidance
                self.obs_rms.count = count
                self.obs_rms.mean = np.loadtxt(mean_file_name, dtype=np.float32)
                self.obs_rms.var = np.loadtxt(var_file_name, dtype=np.float32)
            else:
                # Command tracking
                self.obs_rms_second.count = count
                self.obs_rms_second.mean = np.loadtxt(mean_file_name, dtype=np.float32)
                self.obs_rms_second.var = np.loadtxt(var_file_name, dtype=np.float32)

    def get_running_mean_var_explicit(self, type=None):
        assert type in [1, 2], "Unavailable scaling type."
        if type == 1:
            return self.obs_rms.mean, self.obs_rms.var, self.obs_rms.count
        else:
            return self.obs_rms_second.mean, self.obs_rms_second.var, self.obs_rms_second.count

    def set_running_mean_var_explicit(self, mean, var, count, type=None):
        assert type in [1, 2], "Unavailable scaling type."
        if type == 1:
            self.obs_rms.mean = mean
            self.obs_rms.var = var
            self.obs_rms.count = count
        else:
            self.obs_rms_second.mean = mean
            self.obs_rms_second.var = var
            self.obs_rms_second.count = count

    def save_scaling(self, dir_name, iteration, type=None):
        mean_file_name = dir_name + "/mean" + iteration + ".csv"
        var_file_name = dir_name + "/var" + iteration + ".csv"

        if self.obs_rms_second == None:
            np.savetxt(mean_file_name, self.obs_rms.mean)
            np.savetxt(var_file_name, self.obs_rms.var)
        else:
            assert type in [1, 2], "Unavailable scaling type."
            if type == 1:
                np.savetxt(mean_file_name, self.obs_rms.mean)
                np.savetxt(var_file_name, self.obs_rms.var)
            else:
                np.savetxt(mean_file_name, self.obs_rms_second.mean)
                np.savetxt(var_file_name, self.obs_rms_second.var)

    def set_running_mean_var(self, first_type_dim, second_type_dim):
        self.obs_rms = RunningMeanStd(shape=first_type_dim)
        self.obs_rms_second = RunningMeanStd(shape=second_type_dim)

    def coordinate_observe(self):
        self.wrapper.coordinate_observe(self.coordinate_observation)
        return self.coordinate_observation.copy()

    def observe(self, update_mean=True):
        self.wrapper.observe(self._observation)
        not_normalized_obs = self._observation.copy()

        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)

            return self._normalize_observation(self._observation), not_normalized_obs.copy()
        else:
            return self._observation.copy(), not_normalized_obs.copy()  # two are same

    def reset(self):
        self._done = np.zeros(self.num_envs, dtype=np.bool)
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self.wrapper.reset()

    def partial_reset(self, agentIDs):
        if len(agentIDs) != 0:
            needed_reset = np.zeros(self.num_envs, dtype=np.bool)
            needed_reset[agentIDs] = True
            self._done = np.zeros(self.num_envs, dtype=np.bool)
            self._reward = np.zeros(self.num_envs, dtype=np.float32)
            self.wrapper.partial_reset(needed_reset)

    def _normalize_observation(self, obs, force_normalize=False, type=None):
        if self.normalize_ob:
            return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8), -self.clip_obs,
                           self.clip_obs)
        elif force_normalize:
            if type == 1:
                # loaded obs_rms shape could be different due to different n_envs
                return np.clip((obs - self.obs_rms.mean[0]) / np.sqrt(self.obs_rms.var[0] + 1e-8), -self.clip_obs,
                               self.clip_obs)
            else:
                # loaded obs_rms shape could be different due to different n_envs
                return np.clip((obs - self.obs_rms_second.mean[0]) / np.sqrt(self.obs_rms_second.var[0] + 1e-8), -self.clip_obs,
                               self.clip_obs)
        else:
            return obs

    def force_normalize_observation(self, obs, type=None):
        assert type in [1, 2], "Unavailable scaling type."
        return self._normalize_observation(obs, force_normalize=True, type=type)

    def force_update_ob_rms(self, obs, type=None):
        assert type in [1, 2], "Unavailable scaling type."
        if type == 1:
            self.obs_rms.update(obs)
        else:
            self.obs_rms_second.update(obs)

    def close(self):
        self.wrapper.close()

    def curriculum_callback(self):
        self.wrapper.curriculumUpdate()

    def visualize_desired_command_traj(self, coordinate_desired_command_traj, P_col_desired_command, collision_threshold):
        """

        :param coordinate_desired_command_traj: (n_prediction_step, 2)
        :param P_col_desired_command: (n_prediction_step,)
        :return:
        """
        coordinate_desired_command_traj = np.ascontiguousarray(coordinate_desired_command_traj)
        P_col_desired_command = np.ascontiguousarray(P_col_desired_command)
        self.wrapper.visualize_desired_command_traj(coordinate_desired_command_traj, P_col_desired_command, collision_threshold)

    def visualize_modified_command_traj(self, coordinate_modified_command, P_col_modified_command, collision_threshold):
        coordinate_modified_command = np.ascontiguousarray(coordinate_modified_command)
        P_col_modified_command = np.ascontiguousarray(P_col_modified_command)
        self.wrapper.visualize_modified_command_traj(coordinate_modified_command, P_col_modified_command, collision_threshold)

    def set_goal(self):
        self.wrapper.set_goal(self._goal)
        return self._goal.copy()

    def parallel_set_goal(self):
        self.wrapper.parallel_set_goal(self._parallel_goal)
        return self._parallel_goal.copy()

    def observe_potential_heading_direction(self):
        self.wrapper.computed_heading_direction(self.potential_computed_heading_direction)
        return self.potential_computed_heading_direction.copy()

    @property
    def num_envs(self):
        return self.wrapper.getNumOfEnvs()
    
    def set_user_command(self, command):
        self.wrapper.set_user_command(command)
    
    def reward_logging(self, n_reward):
        self.wrapper.reward_logging(self.reward_log, self.reward_w_cpeff_log, n_reward)

    def contact_logging(self):
        self.wrapper.contact_logging(self.contact_log)

    def torque_and_velocity_logging(self):
        self.wrapper.torque_and_velocity_logging(self.torque_and_velocity_log)
    
    def get_reward_Info(self):
        return self.wrapper.rewardInfo()

    def initialize_n_step(self):
        self.wrapper.initialize_n_step()

    def single_env_collision_check(self):
        # check collision for env0
        return self.wrapper.single_env_collision_check()

    def parallel_env_collision_check(self):
        # check collision for multiple environments
        self.wrapper.parallel_env_collision_check(self._parallel_collision)
        return self._parallel_collision.copy()

    def analytic_planner_collision_check(self, x, y):
        return self.wrapper.analytic_planner_collision_check(x, y)

    def visualize_analytic_planner_path(self, path):
        """

        :param path: (N, 2)  N: path length
        :return:
        """
        self.wrapper.visualize_analytic_planner(path)

    def visualize_waypoints(self, waypoints):
        """

        :param waypoints: (N_waypoints, 2)  N_waypoints: number of waypoints
        :return:
        """
        self.wrapper.visualize_waypoints(waypoints)

    def get_map_size(self, map_size):
        self.wrapper.getMapSize(map_size)

    def get_map_bound(self, map_bound):
        self.wrapper.getMapBound(map_bound)

    def getDepthImage(self):
        self.wrapper.getDepthImage(self._depthImage_vectorized)
        return np.reshape(self._depthImage_vectorized,(self.num_envs, self.vSize, self.hSize))


class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

