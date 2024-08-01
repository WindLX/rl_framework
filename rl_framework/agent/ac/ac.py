from typing import Union, Optional
from abc import abstractmethod, ABCMeta
from os import path

from torch import Tensor, load, save, tensor, float32
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from gymnasium import Env
from gymnasium.vector import VectorEnv
from labml import tracker, logger
from labml_helpers.device import DeviceConfigs

from .config import ACConfig, ACEvalConfig
from ...utils.optim import get_lr
from ...utils.error import SharedACError


class ACAgent(metaclass=ABCMeta):
    """
    Abstract class for an Actor-Critic agent.
    """

    def __init__(
        self,
        config: ACConfig,
        envs: VectorEnv,
        model: Union[Module, dict[str, Module]],
        optimizer: Union[Optimizer, dict[str, Optimizer]],
        lr_scheduler: Union[Optimizer, dict[str, Optional[Optimizer]], None],
    ) -> None:
        self.is_model_shared = config.is_model_shared
        self.save_path = config.save_path
        self.clip_grad_norm = config.clip_grad_norm
        self._data_validation(self.is_model_shared, model, optimizer, lr_scheduler)

        self.device = DeviceConfigs().device

        # initialize tensors for observations
        self.envs = envs
        self.obs, info = self.envs.reset()

        # model
        if self.is_model_shared:
            self.model = model.to(self.device)
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler

        else:
            self.actor_model = model["actor"].to(self.device)
            self.critic_model = model["critic"].to(self.device)
            self.actor_optimizer = optimizer["actor"]
            self.critic_optimizer = optimizer["critic"]

            if lr_scheduler is not None:
                self.actor_lr_scheduler = lr_scheduler["actor"]
                self.critic_lr_scheduler = lr_scheduler["critic"]
            else:
                self.actor_lr_scheduler = None
                self.critic_lr_scheduler = None

        self.load_model()

    @staticmethod
    def _data_validation(
        is_model_shared: bool,
        model: Union[Module, dict[str, Module]],
        optimizer: Union[Optimizer, dict[str, Optimizer]],
        lr_scheduler: Union[Optimizer, dict[str, Optional[Optimizer]], None],
    ):
        if is_model_shared:
            if type(model) != Module:
                raise SharedACError(
                    "Model should be single model for shared actor-critic model"
                )
            if type(optimizer) != Optimizer:
                raise SharedACError(
                    "Learning rate should be single value for shared actor-critic model"
                )
            if lr_scheduler is not None and type(lr_scheduler) != Optimizer:
                raise SharedACError(
                    "Learning rate scheduler should be single value for shared actor-critic model"
                )
        else:
            if (
                type(model) != dict
                and "actor" not in model.keys()
                and "critic" not in model.keys()
            ):
                raise SharedACError(
                    "Model should be tuple for separate actor-critic model"
                )
            if (
                type(optimizer) != dict
                and "actor" not in optimizer.keys()
                and "critic" not in optimizer.keys()
            ):
                raise SharedACError(
                    "Learning rate should be tuple for separate actor-critic model"
                )
            if (
                lr_scheduler is not None
                and type(lr_scheduler) != dict
                and "actor" not in lr_scheduler.keys()
                and "critic" not in lr_scheduler.keys()
            ):
                raise SharedACError(
                    "Learning rate scheduler should be tuple for separate actor-critic model"
                )

    @property
    def models_dict(self):
        return (
            {"shared_model": self.model.state_dict()}
            if self.is_model_shared
            else {
                "actor_model": self.actor_model.state_dict(),
                "critic_model": self.critic_model.state_dict(),
            }
        )

    def save_model(self):
        if self.save_path:
            save(self.models_dict, self.save_path)

    def load_model(self):
        if self.save_path and path.exists(self.save_path):
            model_dict = load(self.save_path, weights_only=True)
            if self.is_model_shared:
                self.model.load_state_dict(model_dict["shared_model"])
            else:
                self.actor_model.load_state_dict(model_dict["actor_model"])
                self.critic_model.load_state_dict(model_dict["critic_model"])

    def destroy(self):
        self.envs.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    @abstractmethod
    def sample(self) -> dict[str, Tensor]:
        raise NotImplementedError("Method not implemented")

    @abstractmethod
    def calc_loss(
        self, samples: dict[str, Tensor]
    ) -> Union[Tensor, tuple[Tensor, Tensor]]:
        raise NotImplementedError("Method not implemented")

    def train_step(self, samples: dict[str, Tensor]):
        if self.is_model_shared:
            loss = self.calc_loss(samples)

            # Zero out the previously calculated gradients
            self.optimizer.zero_grad()

            # Calculate gradients
            loss.backward()

            # Clip gradients
            if self.clip_grad_norm is not None:
                clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad_norm)

            # Update parameters based on gradients
            self.optimizer.step()

            # Set learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            tracker.add("lr", get_lr(self.optimizer))
        else:
            actor_loss, critic_loss = self.calc_loss(samples)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            if (actor_norm := self.clip_grad_norm["actor"]) is not None:
                clip_grad_norm_(self.actor_model.parameters(), max_norm=actor_norm)
            if (critic_norm := self.clip_grad_norm["critic"]) is not None:
                clip_grad_norm_(self.critic_model.parameters(), max_norm=critic_norm)

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            if self.actor_lr_scheduler is not None:
                self.actor_lr_scheduler.step()
            if self.critic_lr_scheduler is not None:
                self.critic_lr_scheduler.step()

            tracker.add("actor_lr", get_lr(self.actor_optimizer))
            tracker.add("critic_lr", get_lr(self.critic_optimizer))


class ACEvalAgent:
    def __init__(self, config: ACEvalConfig, env: Env, actor: Module) -> None:
        self.env = env
        save_path = config.save_path
        self.is_render = config.is_render
        self.is_model_shared = config.is_model_shared
        self.device = DeviceConfigs().device

        checkpoint = None
        if save_path and path.exists(save_path):
            checkpoint = load(save_path, weights_only=True)

        if self.is_model_shared:
            model = actor.to(self.device)
            if checkpoint:
                model.load_state_dict(checkpoint["shared_model"])
                logger.log("Model loaded from {}".format(save_path))
            model.eval()
            self.model = model

        else:
            actor_model = actor.to(self.device)

            if checkpoint:
                actor_model.load_state_dict(checkpoint["actor_model"])
                logger.log("Model loaded from {}".format(save_path))
            actor_model.eval()
            self.actor_model = actor_model

    def run_eval_loop(self, updates: int):
        for update in range(updates):
            obs, info = self.env.reset()
            done = False

            def steps():
                step = 0
                while not done:
                    yield step
                    step += 1

            for step in steps():
                obs = tensor(obs, dtype=float32).unsqueeze(0).to(self.device)

                if self.is_model_shared:
                    pi, _ = self.model(obs)
                else:
                    pi = self.actor_model(obs)
                action = pi.sample().squeeze(0).cpu().numpy()

                obs, reward, done, truncated, info = self.env.step(action)
                entropy_bonus = pi.entropy()

                if self.is_render:
                    self.env.render()

                if done or truncated:
                    print("Episode finished after {} timesteps".format(step + 1))

                tracker.add({"reward": reward, "entropy_bonus": entropy_bonus})
                tracker.save()

                if info is not None:
                    if "reward" in info.keys() and "length" in info.keys():
                        tracker.add("reward_sum", info["reward"])
                        tracker.add("length", info["length"])

            tracker.save()

    def destroy(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()
