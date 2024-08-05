import functools

import torch
import torch.nn as nn
from torch.optim import Adam

import gymnasium as gym
from gymnasium.wrappers.rescale_action import RescaleAction
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

from rl_framework.model.decoder import BetaActionDecoder
from rl_framework.agent.ac import ACEvalConfig
from rl_framework.agent.ppo import PPOConfig, AdvantageNormalizeOptions
from rl_framework.env.wrapper import RewardSumWrapper, SyncWrapper
from rl_framework.env.worker import WorkerSet
from rl_framework.utils.math import (
    orthogonal_init,
)

from tests.test_ppo import test


class Encoder(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.out_features = out_features
        self.lin1 = nn.Linear(in_features=in_features, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=out_features)
        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        x = obs
        h1 = self.activation(self.lin1(x))
        h2 = self.activation(self.lin2(h1))
        return h2


class ActorModel(nn.Module):
    def __init__(self, num_mixtures=4):
        super().__init__()

        self.encoder = Encoder(3, 128)
        self.decoder = BetaActionDecoder(
            action_space=1,
            num_mixtures=num_mixtures,
            in_features=self.encoder.out_features,
        )

    def forward(self, obs: torch.Tensor):
        latent = self.encoder(obs)
        pi = self.decoder(latent)
        return pi


class CriticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder(3, 128)
        self.value = nn.Linear(in_features=self.encoder.out_features, out_features=1)

    def forward(self, obs: torch.Tensor):
        latent = self.encoder(obs)
        value = self.value(latent).reshape(-1)

        return value


if __name__ == "__main__":
    env_name = "Pendulum-v1"

    gamma = 0.99
    base_env = gym.make(env_name, g=9.81)
    envs = WorkerSet(
        RescaleAction(
            NormalizeObservation(
                NormalizeReward(
                    SyncWrapper(RewardSumWrapper(base_env)),
                    gamma=gamma,
                )
            ),
            min_action=0.0,
            max_action=1.0,
        ),
        4,
    )

    ac_config = {
        "is_model_shared": False,
        "save_path": f"{env_name}.pth",
        "clip_grad_norm": {
            "actor": 0.5,
            "critic": 0.5,
        },
    }
    ppo_config = {
        "updates": 500,
        "epochs": 10,
        "batch_size": 2048,
        "mini_batch_size": 128,
        "value_loss_coef": None,
        "entropy_bonus_coef": 0.01,
        "clip_range": 0.1,
        "gamma": gamma,
        "gae_lambda": 0.95,
        "advantage_normalize_option": AdvantageNormalizeOptions.batch,
    }

    conf = PPOConfig(**ac_config, **ppo_config)

    actor = ActorModel()
    critic = CriticModel()
    init_fn = functools.partial(orthogonal_init, gain=0.01)
    actor.apply(init_fn)
    critic.apply(init_fn)
    model = {
        "actor": actor,
        "critic": critic,
    }
    optimizer = {
        "actor": Adam(model["actor"].parameters(), lr=3e-4),
        "critic": Adam(model["critic"].parameters(), lr=3e-4),
    }
    lr_scheduler = {
        # "actor": None,
        # "critic": None,
        "actor": torch.optim.lr_scheduler.ExponentialLR(
            optimizer["actor"], gamma=0.9999
        ),
        "critic": torch.optim.lr_scheduler.ExponentialLR(
            optimizer["critic"], gamma=0.9999
        ),
    }

    # eval
    eval_base_env = gym.make(env_name, render_mode="human", g=9.81)
    eval_env = RewardSumWrapper(
        RescaleAction(
            NormalizeObservation(eval_base_env),
            min_action=0.0,
            max_action=1.0,
        )
    )

    ac_eval_conf = {
        "is_model_shared": False,
        "save_path": f"{env_name}.pth",
        "is_render": True,
    }

    eval_conf = ACEvalConfig(**ac_eval_conf)
    actor = ActorModel()

    test(
        env_name, conf, eval_conf, envs, eval_env, model, actor, optimizer, lr_scheduler
    )
