import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium.wrappers.autoreset import AutoResetWrapper
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward

from rl_framework.agent.ac import ACEvalConfig
from rl_framework.agent.ppo import PPOConfig, AdvantageNormalizeOptions
from rl_framework.utils.env import RewardSumWrapper

from tests.test_ppo import test


class ActorModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(in_features=4, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=128)

        self.pi_logits = nn.Linear(in_features=128, out_features=2)

        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        pi = Categorical(logits=self.pi_logits(h))
        return pi


class CriticModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.lin = nn.Linear(in_features=4, out_features=256)
        self.lin2 = nn.Linear(in_features=256, out_features=128)

        self.value = nn.Linear(in_features=128, out_features=1)

        self.activation = nn.Tanh()

    def forward(self, obs: torch.Tensor):
        h = self.activation(self.lin(obs))
        h = self.activation(self.lin2(h))

        value = self.value(h).reshape(-1)

        return value


if __name__ == "__main__":
    env_name = "CartPole-v1"

    # train
    base_env = gym.make(env_name)
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: NormalizeReward(
                NormalizeObservation(RewardSumWrapper(AutoResetWrapper(base_env))),
                gamma=0.99,
            ),
        ]
        * 16
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
        "epochs": 8,
        "env_steps": 500,
        "batches": 4,
        "value_loss_coef": None,
        "entropy_bonus_coef": 0.01,
        "clip_range": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "advantage_normalize_option": AdvantageNormalizeOptions.batch,
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
        "actor": None,
        "critic": None,
        # "actor": torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer["actor"], gamma=0.999
        # ),
        # "critic": torch.optim.lr_scheduler.ExponentialLR(
        #     optimizer["critic"], gamma=0.999
        # ),
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
