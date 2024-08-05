from typing import Union, Optional
from copy import deepcopy

import numpy as np
import torch
from torch.nn import Module, MSELoss
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from labml import monit, tracker, logger

from .config import SACConfig
from ...env.worker import WorkerSet
from ...utils.replay import ReplayBuffer
from ...utils.math import soft_update


class SACAgent:
    """
    ## SACAgent
    """

    def __init__(
        self,
        config: SACConfig,
        envs: WorkerSet,
        model: Union[Module, dict[str, Module]],
        optimizer: Union[Optimizer, dict[str, Optimizer]],
        lr_scheduler: Union[Optimizer, dict[str, Optional[Optimizer]], None] = None,
    ):
        self.config = config
        self.envs = envs
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.num_envs = len(envs)
        self.batch_size = config.batch_size
        self.gamma = config.gamma
        self.tau = config.tau
        self.alpha = config.alpha
        self.target_update_interval = config.target_update_interval
        self.target_entropy = config.target_entropy

        self._sample_steps = 0

        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(
            config.replay_size, self.batch_size, self.device
        )

        # Losses
        self.mse_loss = MSELoss()

        # Target networks
        self.target_critic_1 = deepcopy(self.model["critic_1"])
        self.target_critic_2 = deepcopy(self.model["critic_2"])

        # Ensure target networks have the same initial weights as their respective critics
        soft_update(self.target_critic_1, self.model["critic_1"], 1.0)
        soft_update(self.target_critic_2, self.model["critic_2"], 1.0)

    @property
    def sample_steps(self) -> int:
        return self._sample_steps

    def sample(self) -> dict[str, torch.Tensor]:
        next_obs, info = self.envs.reset()
        self.replay_buffer.clear()
        self.replay_buffer.reset(next_obs)

        with torch.no_grad():
            while not self.envs.is_all_locked:
                pi, _ = self.model["actor"](torch.from_numpy(next_obs).to(self.device))
                a = pi.sample()
                actions = a.cpu().numpy()
                log_pis = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                next_obs, rewards, termination, truncation, infos = self.envs.step(
                    actions
                )
                done = np.logical_or(termination, truncation)

                self.replay_buffer.append(next_obs, actions, rewards, done)

                # collect episode info, which is available if an episode finished;
                for info in infos:
                    if info is not None:
                        if "reward" in info.keys() and "length" in info.keys():
                            tracker.add("reward", info["reward"])
                            tracker.add("length", info["length"])
                            self._sample_steps += info["length"]

        return self.replay_buffer.sample()

    def train(self, samples: dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """
        states, actions, rewards, next_states, dones = samples

        # Critic loss
        with torch.no_grad():
            next_action, next_log_pi = self.model["actor"](next_states)
            q1_target = self.target_critic_1(next_states, next_action)
            q2_target = self.target_critic_2(next_states, next_action)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_pi
            q_target = rewards + (1.0 - dones) * self.gamma * q_target

        q1 = self.model["critic_1"](states, actions)
        q2 = self.model["critic_2"](states, actions)
        critic_loss = self.mse_loss(q1, q_target) + self.mse_loss(q2, q_target)

        self.optimizer["critic_1"].zero_grad()
        self.optimizer["critic_2"].zero_grad()
        critic_loss.backward()
        self.optimizer["critic_1"].step()
        self.optimizer["critic_2"].step()

        # Actor loss
        pi, log_pi = self.model["actor"](states)
        q1_pi = self.model["critic_1"](states, pi)
        q2_pi = self.model["critic_2"](states, pi)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_pi - q_pi).mean()

        self.optimizer["actor"].zero_grad()
        actor_loss.backward()
        self.optimizer["actor"].step()

        # Alpha loss
        alpha_loss = -(
            self.model["log_alpha"] * (log_pi + self.target_entropy).detach()
        ).mean()

        self.optimizer["alpha"].zero_grad()
        alpha_loss.backward()
        self.optimizer["alpha"].step()

        self.alpha = self.model["log_alpha"].exp()

        if self.lr_scheduler is not None:
            self.lr_scheduler["actor"].step()
            self.lr_scheduler["critic_1"].step()
            self.lr_scheduler["critic_2"].step()
            self.lr_scheduler["alpha"].step()

        # Soft update of target networks
        if self._sample_steps % self.target_update_interval == 0:
            soft_update(self.target_critic_1, self.model["critic_1"], self.tau)
            soft_update(self.target_critic_2, self.model["critic_2"], self.tau)

    def run_training_loop(self, global_step: int = 0):
        """
        ### Run training loop
        """

        for update in monit.loop(range(global_step + 1, self.config.updates)):
            # sample with current policy
            samples = self.sample()

            # train the model
            self.train(samples)
            tracker.add("steps", self.sample_steps)

            # Save tracked indicators.
            tracker.save()
            # Add a new line to the screen periodically
            # if (update + 1) % 1_000 == 0:
            #     logger.log()

            self.save_model()
