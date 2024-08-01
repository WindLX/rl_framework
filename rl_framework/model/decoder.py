import torch
import torch.nn as nn

from ..utils.math import beta_mixture_model, gaussian_mixture_model


class BetaActionDecoder(nn.Module):
    def __init__(self, in_features: int, action_space: int, num_mixtures: int = 1):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.action_space = action_space
        self.weights = nn.Linear(in_features=in_features, out_features=num_mixtures)
        self.alphas = nn.Linear(
            in_features=in_features, out_features=num_mixtures * action_space
        )
        self.betas = nn.Linear(
            in_features=in_features, out_features=num_mixtures * action_space
        )
        self.softplus = nn.Softplus()

    def forward(self, h: torch.Tensor):
        weights = self.weights(h)
        alphas = (
            self.softplus(self.alphas(h).view(-1, self.num_mixtures, self.action_space))
            + 1.0
        )
        betas = (
            self.softplus(self.betas(h).view(-1, self.num_mixtures, self.action_space))
            + 1.0
        )
        pi = beta_mixture_model(weights, alphas, betas)
        return pi


class GaussianActionDecoder(nn.Module):
    def __init__(self, in_features: int, action_space: int, num_mixtures: int = 1):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.action_space = action_space
        self.weights = nn.Linear(in_features=in_features, out_features=num_mixtures)
        self.locs = nn.Linear(
            in_features=in_features, out_features=num_mixtures * action_space
        )
        self.scales = nn.Linear(
            in_features=in_features, out_features=num_mixtures * action_space
        )
        self.softplus = nn.Softplus()

    def forward(self, h: torch.Tensor):
        weights = self.weights(h)
        locs = self.locs(h).view(-1, self.num_mixtures, self.action_space)
        scales = self.softplus(
            self.scales(h).view(-1, self.num_mixtures, self.action_space)
        )

        pi = gaussian_mixture_model(weights, locs, scales)
        return pi
