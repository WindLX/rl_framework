import torch
from torch import Tensor
from torch.distributions import (
    Distribution,
    Beta,
    Normal,
    Categorical,
    Independent,
    MixtureSameFamily,
)


def orthogonal_init(layer, gain: float = 1.0):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain=gain)
        torch.nn.init.constant_(layer.bias, 0)


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


def beta_mixture_model(weights: Tensor, alpha: Tensor, beta: Tensor) -> Distribution:
    """#### Create a Beta Mixture Model (BMM) distribution

    Args:
        weights (Tensor): logits_weights of the mixture components (batch_size, num_mixtures)
        alpha (Tensor): (batch_size, num_mixtures, action_space)
        beta (Tensor): (batch_size, num_mixtures, action_space)

    Returns:
        Distribution: bmm
    """
    mix = Categorical(logits=weights)
    comp = Independent(Beta(alpha, beta), 1)
    bmm = MixtureSameFamilyWithEntropy(mix, comp)
    return bmm
