from os import path

import torch
import torch.nn as nn
from torch.nn import Module
from torch.distributions import Categorical

import gymnasium as gym

from labml import experiment
from labml.internal.configs.dynamic_hyperparam import (
    FloatDynamicHyperParam,
    IntDynamicHyperParam,
)

from rl_framework.agent.ppo import PPOAgent, PPOConfigs, PPOEvalConfigs, PPOEvalAgent
from rl_framework.utils.env import RlEnvWrapper


class ActorModel(Module):
    """
    ## Model
    """

    def __init__(self):
        super().__init__()

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        self.lin = nn.Linear(in_features=4, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=128)

        # A fully connected layer to get logits for $\pi$
        self.pi_logits = nn.Linear(in_features=128, out_features=2)

        #
        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        pi = Categorical(logits=self.pi_logits(h))
        return pi


class CriticModel(Module):
    """
    ## Model
    """

    def __init__(self):
        super().__init__()

        # A fully connected layer takes the flattened
        # frame from third convolution layer, and outputs
        self.lin = nn.Linear(in_features=4, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=128)

        # A fully connected layer to get value function
        self.value = nn.Linear(in_features=128, out_features=1)

        #
        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        value = self.value(h).reshape(-1)

        return value


def main():
    base_env = gym.make("CartPole-v1")
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: RlEnvWrapper(base_env),
        ]
        * 16
    )

    conf = PPOConfigs()
    conf.is_model_shared = False
    conf.model = (ActorModel(), CriticModel())
    conf.envs = envs
    conf.learning_rate = FloatDynamicHyperParam(1e-6, (0, 1e-3))
    conf.env_steps = 500
    conf.updates = 2500
    conf.epochs = IntDynamicHyperParam(32, (1, 100))
    conf.batches = 8
    conf.value_loss_coef = FloatDynamicHyperParam(0.5, (0, 1))
    conf.entropy_bonus_coef = FloatDynamicHyperParam(0.05, (0, 1))
    conf.clip_range = FloatDynamicHyperParam(0.1, (0, 1))
    conf.gae_gamma = 0.99
    conf.gae_lambda = 0.95
    conf.save_path = "ppo.pth"

    experiment.create(name="ppo")
    experiment.configs(conf)

    global_step = 0
    if path.exists("ppo.pth"):
        checkpoint = torch.load("ppo.pth", weights_only=True)
        global_step = checkpoint["global_step"]

    with PPOAgent(conf) as m:
        with experiment.start(global_step=global_step):
            m.run_training_loop(global_step)


def eval():
    base_env = gym.make("CartPole-v1", render_mode="human")
    env = RlEnvWrapper(base_env)

    conf = PPOEvalConfigs()
    conf.env = env
    conf.model = ActorModel()
    conf.is_model_shared = False
    conf.save_path = "ppo.pth"

    experiment.create(name="ppo_eval")

    with PPOEvalAgent(conf) as m:
        with experiment.start():
            m.run_eval_loop(10)


if __name__ == "__main__":
    # main()
    eval()
