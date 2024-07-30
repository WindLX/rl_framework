import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Normal,
    Categorical,
    Independent,
    MixtureSameFamily,
)


import numpy as np


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x


class RewardScaling:
    def __init__(self, shape, gamma: float):
        self.shape = shape  # reward shape = 1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):
        # When an episode is done, we should reset 'self.R'
        self.R = np.zeros(self.shape)


def normalize(data: Tensor):
    """#### Normalize function"""
    return (data - data.mean()) / (data.std() + 1e-8)


def joint_log_probs(pi: Distribution, actions: Tensor) -> Tensor:
    """#### Calculate log probabilities of actions"""
    log_probs = pi.log_prob(actions)

    # joint log probs
    joint_log_probs = log_probs.sum(axis=-1) if len(log_probs.shape) > 1 else log_probs
    return joint_log_probs


# Extend MixtureSameFamily to include entropy method
class MixtureSameFamilyWithEntropy(MixtureSameFamily):
    def entropy(self):
        """Calculate the entropy of the GMM distribution."""
        # Get the component distributions
        component_distributions = self.component_distribution
        # Compute entropy for each component
        entropies = component_distributions.entropy()
        # Weighted sum of component entropies
        weighted_entropies = torch.sum(
            entropies * self.mixture_distribution.probs, dim=-1
        )
        return weighted_entropies


def gaussian_mixture_model(weights: Tensor, loc: Tensor, scale: Tensor) -> Distribution:
    """#### Create a Gaussian Mixture Model (GMM) distribution

    Args:
        weights (Tensor): logits_weights of the mixture components (batch_size, num_mixtures)
        loc (Tensor): (batch_size, num_mixtures, action_space)
        scale (Tensor): (batch_size, num_mixtures, action_space)

    Returns:
        Distribution: gmm
    """
    mix = Categorical(logits=weights)
    comp = Independent(Normal(loc, scale), 1)
    gmm = MixtureSameFamilyWithEntropy(mix, comp)
    return gmm
