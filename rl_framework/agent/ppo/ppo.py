from typing import Union, Optional

import numpy as np
import torch
from torch.nn import Module, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from labml import monit, tracker, logger

from .loss import ClippedPPOLoss, ClippedValueFunctionLoss
from .config import PPOConfig, AdvantageNormalizeOptions
from ...env.worker import WorkerSet
from ..ac.ac import ACAgent
from ...utils.replay import ReplayBuffer
from ...utils.gae import GAE
from ...utils.math import normalize
from ...utils.optim import get_lr


class PPOAgent(ACAgent):
    """
    ## PPOAgent
    """

    def __init__(
        self,
        config: PPOConfig,
        envs: WorkerSet,
        model: Union[Module, dict[str, Module]],
        optimizer: Union[Optimizer, dict[str, Optimizer]],
        lr_scheduler: Union[Optimizer, dict[str, Optional[Optimizer]], None] = None,
    ):
        super().__init__(config, envs, model, optimizer, lr_scheduler)

        # number of updates
        self.updates = config.updates
        # number of epochs to train the model with sampled data
        self.epochs = config.epochs
        # number of worker processes
        self.num_envs = len(envs)
        # batch_size
        self.batch_size = config.batch_size
        # mini batch size
        self.mini_batch_size = config.mini_batch_size

        self.value_loss_coef = config.value_loss_coef
        self.entropy_bonus_coef = config.entropy_bonus_coef
        self.clip_range = config.clip_range
        self.advantage_normlize_option = config.advantage_normalize_option
        self.use_clip_value_loss = config.use_clip_value_loss

        self._sample_steps = 0

        # #### Initialize

        # GAE and replay buffer
        gae = GAE(config.gamma, config.gae_lambda)
        self.replay_buffer = ReplayBuffer(len(self.envs), gae, self.device)

        # Loss
        self.ppo_loss = ClippedPPOLoss()
        if self.use_clip_value_loss:
            self.value_loss = ClippedValueFunctionLoss()
        else:
            self.value_loss = MSELoss()

    @property
    def sample_steps(self) -> int:
        return self._sample_steps

    def sample(self) -> dict[str, torch.Tensor]:
        next_obs, info = self.envs.reset()
        self.replay_buffer.clear()
        self.replay_buffer.reset(next_obs)

        with torch.no_grad():
            while not self.envs.is_all_locked:
                if self.is_model_shared:
                    pi, v = self.model(torch.from_numpy(next_obs).to(self.device))
                else:
                    pi = self.actor_model(torch.from_numpy(next_obs).to(self.device))
                    v = self.critic_model(torch.from_numpy(next_obs).to(self.device))
                values = v.cpu().numpy()
                a = pi.sample()
                actions = a.cpu().numpy()
                log_pis = pi.log_prob(a).cpu().numpy()

                # run sampled actions on each worker
                next_obs, rewards, termination, truncation, infos = self.envs.step(
                    actions
                )
                done = np.logical_or(termination, truncation)

                self.replay_buffer.append(
                    next_obs, actions, rewards, done, values, log_pis
                )

                # collect episode info, which is available if an episode finished;
                for info in infos:
                    if info is not None:
                        if "reward" in info.keys() and "length" in info.keys():
                            tracker.add("reward", info["reward"])
                            tracker.add("length", info["length"])
                            self._sample_steps += info["length"]

            # Get value of after the final step
            if self.is_model_shared:
                _, v = self.model(torch.from_numpy(next_obs).to(self.device))
            else:
                v = self.critic_model(torch.from_numpy(next_obs).to(self.device))
            last_values = v.cpu().numpy()
            self.replay_buffer.clip()
            self.replay_buffer.append_last_values(last_values)

        # calculate advantages
        self.replay_buffer.calc_advantages()

        self.replay_buffer.flatten()
        if self.advantage_normlize_option == AdvantageNormalizeOptions.batch:
            self.replay_buffer.normalize_advantages()

        return self.replay_buffer.experience

    def train(self, samples: dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """
        batch_size = len(samples["obss"])
        for _ in range(self.epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(
                SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False
            ):
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[index]
                self.train_step(mini_batch)

    def calc_loss(
        self, samples: dict[str, torch.Tensor]
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        ### Calculate total loss
        """

        # $R_t$ returns sampled from $\pi_{\theta_{OLD}}$
        sampled_return = samples["values"] + samples["advantages"]

        # $\bar{A_t} = \frac{\hat{A_t} - \mu(\hat{A_t})}{\sigma(\hat{A_t})}$,
        # where $\hat{A_t}$ is advantages sampled from $\pi_{\theta_{OLD}}$.
        # Refer to sampling function in [Main class](#main) below
        #  for the calculation of $\hat{A}_t$.
        if self.advantage_normlize_option == AdvantageNormalizeOptions.minibatch:
            sampled_normalized_advantage = normalize(samples["advantages"])
        else:
            sampled_normalized_advantage = samples["advantages"]

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        if self.is_model_shared:
            pi, value = self.model(samples["obss"])
        else:
            pi = self.actor_model(samples["obss"])
            value = self.critic_model(samples["obss"])

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        log_pi = pi.log_prob(samples["actions"])

        # Calculate policy loss
        policy_loss = self.ppo_loss(
            log_pi, samples["log_pis"], sampled_normalized_advantage, self.clip_range
        )

        # Calculate Entropy Bonus
        #
        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate value function loss
        if self.use_clip_value_loss:
            value_loss = self.value_loss(
                value, samples["values"], sampled_return, self.clip_range
            )
        else:
            value_loss = self.value_loss(value, sampled_return)

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        if self.is_model_shared:
            loss = (
                policy_loss
                + self.value_loss_coef * value_loss
                - self.entropy_bonus_coef * entropy_bonus
            )
        else:
            actor_loss = policy_loss - self.entropy_bonus_coef * entropy_bonus
            critic_loss = value_loss

        # for monitoring
        approx_kl_divergence = 0.5 * ((samples["log_pis"] - log_pi) ** 2).mean()

        # Add to tracker
        tracker.add(
            {
                "policy_reward": -policy_loss,
                "value_loss": value_loss,
                "entropy_bonus": entropy_bonus,
                "kl_div": approx_kl_divergence,
                "clip_fraction": self.ppo_loss.clip_fraction,
            }
        )

        return loss if self.is_model_shared else [actor_loss, critic_loss]

    def train_step(self, samples: dict[str, torch.Tensor]):
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

    @staticmethod
    def merge_dicts(*dicts: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        merged_dict = {}
        for d in dicts:
            for key, value in d.items():
                if key in merged_dict:
                    merged_dict[key] = torch.cat((merged_dict[key], value))
                else:
                    merged_dict[key] = value
        return merged_dict

    def run_training_loop(self, global_step: int = 0):
        """
        ### Run training loop
        """

        for update in monit.loop(range(global_step + 1, self.updates)):
            # sample with current policy
            samples = self.sample()
            while samples["actions"].shape[0] < self.batch_size:
                extra_samples = self.sample()
                samples = self.merge_dicts(samples, extra_samples)

            # train the model
            self.train(samples)
            tracker.add("steps", self.sample_steps)

            # Save tracked indicators.
            tracker.save()
            # Add a new line to the screen periodically
            # if (update + 1) % 1_000 == 0:
            #     logger.log()

            self.save_model()
