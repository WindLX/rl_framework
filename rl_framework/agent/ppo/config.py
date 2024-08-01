from typing import Optional
from enum import Enum

from pydantic import model_validator, field_validator, ValidationInfo

from ..ac import ACConfig
from ...utils.error import SharedACError


class AdvantageNormalizeOptions(Enum):
    minibatch = 0
    batch = 1
    disable = 2


class PPOConfig(ACConfig):
    """PPO Configurations"""

    # Number of updates
    updates: int
    # ️Number of epochs to train the model with sampled data.
    epochs: int = 4
    # Number of steps to run on each process for a single update
    env_steps: int
    # Number of mini batches
    batches: int = 4
    # Value loss coefficient.
    value_loss_coef: Optional[float] = 0.5
    # Entropy bonus coefficient.
    entropy_bonus_coef: float = 0.01
    # Clip range.
    clip_range: float = 0.1
    # gamma discount factor
    gamma: float = 0.99
    # gae lambda
    gae_lambda: float = 0.95
    # norm_advantage
    advantage_normalize_option: AdvantageNormalizeOptions = (
        AdvantageNormalizeOptions.minibatch
    )

    @model_validator(mode="after")
    def check_value_loss_coef(self):
        if self.value_loss_coef is None and self.is_model_shared:
            raise SharedACError(
                "`value_loss_coef` mustn't be None if `is_model_shared` is True"
            )
        return self

    @field_validator("clip_range", "gamma", "gae_lambda", mode="after")
    @classmethod
    def check_value_range(cls, v: float, info: ValidationInfo):
        if v < 0 or v > 1:
            raise ValueError(f"`{info.field_name}` should be in range [0, 1]")
        return v
