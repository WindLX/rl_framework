from typing import Union, Optional

from torch.nn import Module

from labml.configs import BaseConfigs
from labml.internal.configs.dynamic_hyperparam import (
    FloatDynamicHyperParam,
    IntDynamicHyperParam,
)

from gymnasium import Env
from gymnasium.vector import VectorEnv


class PPOConfigs(BaseConfigs):
    """PPO Configurations"""

    # model
    model: Union[Module, tuple[Module, Module]]
    # is actor and critic share the same model
    is_model_shared: bool = True
    # envs
    envs: VectorEnv
    # Number of updates
    updates: int
    # ⚙️ Number of epochs to train the model with sampled data.
    # You can change this while the experiment is running.
    epochs: IntDynamicHyperParam = IntDynamicHyperParam(8)
    # Number of steps to run on each process for a single update
    env_steps: int = 128
    # Number of mini batches
    batches: int = 4
    # ⚙️ Value loss coefficient.
    # You can change this while the experiment is running.
    value_loss_coef: FloatDynamicHyperParam = FloatDynamicHyperParam(0.5)
    # ⚙️ Entropy bonus coefficient.
    # You can change this while the experiment is running.
    entropy_bonus_coef: FloatDynamicHyperParam = FloatDynamicHyperParam(0.01)
    # ⚙️ Clip range.
    clip_range: FloatDynamicHyperParam = FloatDynamicHyperParam(0.1)
    # You can change this while the experiment is running.
    # ⚙️ Learning rate.
    learning_rate: FloatDynamicHyperParam = FloatDynamicHyperParam(1e-3, (0, 1e-3))
    # gae_gamma
    gae_gamma: float = 0.99
    # gae lambda
    gae_lambda: float = 0.95
    # save_path
    save_path: Optional[str] = None


class PPOEvalConfigs(BaseConfigs):
    """Evaluation Configurations"""

    # model
    model: Module
    # is actor and critic share the same model
    is_model_shared: bool = True
    # envs
    env: Env
    # save path
    save_path: Optional[str] = None
    # is_render
    is_render: bool = True
