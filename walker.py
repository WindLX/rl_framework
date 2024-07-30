from os import path

import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium.wrappers.transform_reward import TransformReward

from labml import experiment
from labml.internal.configs.dynamic_hyperparam import (
    FloatDynamicHyperParam,
    IntDynamicHyperParam,
)

from rl_framework.agent.ppo import PPOAgent, PPOConfigs, PPOEvalConfigs, PPOEvalAgent
from rl_framework.utils.env import RlEnvWrapper
from rl_framework.utils.math import gaussian_mixture_model


class ActorModel(nn.Module):
    """
    ## Model
    """

    def __init__(self, num_mixtures=6):
        super().__init__()

        self.num_mixtures = num_mixtures

        self.lin = nn.Linear(in_features=24, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=128)

        # Output layers for the GMM
        self.weights = nn.Linear(in_features=128, out_features=num_mixtures)
        self.means = nn.Linear(in_features=128, out_features=num_mixtures * 4)
        self.log_stds = nn.Linear(in_features=128, out_features=num_mixtures * 4)

        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        # Predict the weights, means, and log standard deviations for the GMM
        weights = self.weights(h)  # (batch_size, num_mixtures)
        means = self.means(h).view(
            -1, self.num_mixtures, 4
        )  # (batch_size, num_mixtures, action_space)
        log_stds = self.log_stds(h).view(
            -1, self.num_mixtures, 4
        )  # (batch_size, num_mixtures, action_space)
        stds = torch.exp(log_stds)  # (batch_size, num_mixtures, action_space)

        # Create the GMM distribution
        pi = gaussian_mixture_model(weights, means, stds)

        return pi


class CriticModel(nn.Module):
    """
    ## Model
    """

    def __init__(self, num_mixtures=6):
        super().__init__()

        self.num_mixtures = num_mixtures

        self.lin = nn.Linear(in_features=24, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=128)

        self.value = nn.Linear(in_features=128, out_features=1)

        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        value = self.value(h).reshape(-1)

        return value


def main():
    base_env = gym.make("BipedalWalker-v3")
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: RlEnvWrapper(TransformReward(base_env, lambda r: 1 * r)),
        ]
        * 8
    )

    conf = PPOConfigs()
    conf.is_model_shared = False
    conf.model = (ActorModel(), CriticModel())
    conf.envs = envs
    conf.learning_rate = FloatDynamicHyperParam(1e-5, (0, 1e-3))
    conf.env_steps = 1600
    conf.updates = 300
    conf.epochs = IntDynamicHyperParam(16, (1, 100))
    conf.batches = 4
    conf.value_loss_coef = FloatDynamicHyperParam(0.2, (0, 1))
    conf.entropy_bonus_coef = FloatDynamicHyperParam(0.01, (0, 1))
    conf.clip_range = FloatDynamicHyperParam(0.1, (0, 1))
    conf.gae_gamma = 0.99
    conf.gae_lambda = 0.7
    conf.save_path = "BipedalWalker2.pth"

    experiment.create(name="BipedalWalker")
    experiment.configs(conf)

    global_step = 0
    if path.exists(conf.save_path):
        checkpoint = torch.load(conf.save_path, weights_only=True)
        # global_step = checkpoint["global_step"]

    with PPOAgent(conf) as m:
        with experiment.start(global_step=global_step):
            m.run_training_loop(global_step)


def eval():
    base_env = gym.make("BipedalWalker-v3", render_mode="human")
    env = RlEnvWrapper(base_env)

    conf = PPOEvalConfigs()
    conf.env = env
    conf.model = ActorModel()
    conf.is_model_shared = False
    conf.save_path = "BipedalWalker2.pth"

    experiment.create(name="BipedalWalker_eval")

    with PPOEvalAgent(conf) as m:
        with experiment.start():
            m.run_eval_loop(10)


if __name__ == "__main__":
    # main()
    eval()
