import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

from rl_framework.agent.ac import ACEvalConfig
from rl_framework.agent.ppo import PPOConfig, AdvantageNormalizeOptions
from rl_framework.env.wrapper import RewardSumWrapper, SyncWrapper
from rl_framework.env.worker import WorkerSet

from tests.test_ppo import test


class ActorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(in_features=8, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=64)

        self.pi_logits = nn.Linear(in_features=64, out_features=4)

        self.activation = nn.LeakyReLU()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        pi = Categorical(logits=self.pi_logits(h))
        return pi


class CriticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(in_features=8, out_features=64)
        self.lin2 = nn.Linear(in_features=64, out_features=64)

        self.value = nn.Linear(in_features=64, out_features=1)

        self.activation = nn.LeakyReLU()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        value = self.value(h).reshape(-1)

        return value


if __name__ == "__main__":
    env_name = "LunarLander-v2"

    # train
    base_env = gym.make(env_name)
    envs = WorkerSet(
        NormalizeObservation(NormalizeReward(SyncWrapper(RewardSumWrapper(base_env)))),
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
        "updates": 3000,
        "epochs": 10,
        "batch_size": 2048,
        "mini_batch_size": 128,
        "value_loss_coef": None,
        "entropy_bonus_coef": 0.01,
        "clip_range": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "advantage_normalize_option": AdvantageNormalizeOptions.batch,
        "use_clip_value_loss": True,
    }

    conf = PPOConfig(**ac_config, **ppo_config)
    model = {
        "actor": ActorModel(),
        "critic": CriticModel(),
    }
    optimizer = {
        "actor": Adam(model["actor"].parameters(), lr=3e-4),
        "critic": Adam(model["critic"].parameters(), lr=3e-4),
    }
    lr_scheduler = {
        "actor": torch.optim.lr_scheduler.ExponentialLR(
            optimizer["actor"], gamma=0.9999
        ),
        "critic": torch.optim.lr_scheduler.ExponentialLR(
            optimizer["critic"], gamma=0.9999
        ),
    }

    # eval
    eval_base_env = gym.make(env_name, render_mode="human")
    eval_env = NormalizeObservation(RewardSumWrapper(eval_base_env))

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
