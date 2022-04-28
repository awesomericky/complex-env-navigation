import torch.nn as nn
import numpy as np
import torch
from torch.distributions import Normal
import pdb

class Actor:
    def __init__(self, architecture, distribution, device='cpu'):
        super(Actor, self).__init__()

        self.architecture = architecture
        self.distribution = distribution
        self.architecture.to(device)
        self.distribution.to(device)
        self.device = device

    def sample(self, obs):
        logits = self.architecture.architecture(obs)
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs)
        return self.distribution.evaluate(obs, action_mean, actions)

    def parameters(self):
        return [*self.architecture.parameters(), *self.distribution.parameters()]

    def noiseless_action(self, obs):
        return self.architecture.architecture(torch.from_numpy(obs).to(self.device))

    def save_deterministic_graph(self, file_name, example_input, device='cpu'):
        transferred_graph = torch.jit.trace(self.architecture.architecture.to(device), example_input)
        torch.jit.save(transferred_graph, file_name)
        self.architecture.architecture.to(self.device)

    def deterministic_parameters(self):
        return self.architecture.parameters()

    @property
    def obs_shape(self):
        return self.architecture.input_shape

    @property
    def action_shape(self):
        return self.architecture.output_shape

class Actor_two_side_clip(Actor):
    def __init__(self, architecture, distribution, clipping_range, device='cpu'):
        """
        :param clipping_range: (N, 2) (numpy tensor) ==> [:, 0]: min, [:, 1]: max, N: number of value types (= architecture output dim)
        """
        super(Actor_two_side_clip, self).__init__(architecture, distribution, device)
        assert self.action_shape[0] == clipping_range.shape[0], "Clipping range dimension does not match with output dimension"
        self.min_clip = torch.from_numpy(clipping_range[:, 0].astype(np.float32)).to(device)
        self.max_clip = torch.from_numpy(clipping_range[:, 1].astype(np.float32)).to(device)
        self.final_activation_fn = nn.Tanh()

    def sample(self, obs):
        logits = self.architecture.architecture(obs)
        logits = self.final_activation_fn(logits) * self.max_clip
        actions, log_prob = self.distribution.sample(logits)
        return actions.cpu().detach(), log_prob.cpu().detach()

    def evaluate(self, obs, actions):
        action_mean = self.architecture.architecture(obs)
        action_mean = self.final_activation_fn(action_mean) * self.max_clip
        return self.distribution.evaluate(obs, action_mean, actions)

class Critic:
    def __init__(self, architecture, device='cpu'):
        super(Critic, self).__init__()
        self.architecture = architecture
        self.architecture.to(device)

    def predict(self, obs):
        return self.architecture.architecture(obs).detach()

    def evaluate(self, obs):
        return self.architecture.architecture(obs)

    def parameters(self):
        return [*self.architecture.parameters()]

    @property
    def obs_shape(self):
        return self.architecture.input_shape


class MLP(nn.Module):
    def __init__(self, shape, actionvation_fn, input_size, output_size):
        super(MLP, self).__init__()
        self.activation_fn = actionvation_fn

        modules = [nn.Linear(input_size, shape[0]), self.activation_fn()]
        scale = [np.sqrt(2)]

        for idx in range(len(shape)-1):
            modules.append(nn.Linear(shape[idx], shape[idx+1]))
            modules.append(self.activation_fn())
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


class MultivariateGaussianDiagonalCovariance(nn.Module):
    def __init__(self, dim, init_std):
        super(MultivariateGaussianDiagonalCovariance, self).__init__()
        self.dim = dim
        if type(init_std) == float:
            self.std = nn.Parameter(init_std * torch.ones(dim))
        elif type(init_std) == list:
            self.std = nn.Parameter(torch.tensor(init_std))
        else:
            raise ValueError(f"Unsupported init_std type: {type(init_std)}")

        self.distribution = None

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob

    def evaluate(self, inputs, logits, outputs):
        distribution = Normal(logits, self.std.reshape(self.dim))

        actions_log_prob = distribution.log_prob(outputs).sum(dim=1)
        entropy = distribution.entropy().sum(dim=1)

        return actions_log_prob, entropy

    def entropy(self):
        return self.distribution.entropy()

    def enforce_minimum_std(self, min_std):
        current_std = self.std.detach()
        new_std = torch.maximum(current_std, min_std.detach()).detach()
        self.std.data = new_std

class MultivariateGaussianDiagonalCovariance_two_side_clip(MultivariateGaussianDiagonalCovariance):
    def __init__(self, dim, init_std, clipping_range, device):
        """

        :param clipping_range: (N, 2) (numpy tensor) ==> [:, 0]: min, [:, 1]: max, N: number of value types (= dim)
        """
        assert dim == clipping_range.shape[0], "Clipping range dimension does not match with action dimension"
        super(MultivariateGaussianDiagonalCovariance_two_side_clip, self).__init__(dim, init_std)
        self.min_clip = torch.from_numpy(clipping_range[:, 0].astype(np.float32)).to(device)
        self.max_clip = torch.from_numpy(clipping_range[:, 1].astype(np.float32)).to(device)

    def sample(self, logits):
        self.distribution = Normal(logits, self.std.reshape(self.dim))

        samples = self.distribution.sample()
        samples = torch.clamp(samples, min=self.min_clip, max=self.max_clip)
        log_prob = self.distribution.log_prob(samples).sum(dim=1)

        return samples, log_prob
