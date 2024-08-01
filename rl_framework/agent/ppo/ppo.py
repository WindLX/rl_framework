from typing import Union, Optional

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from gymnasium.vector import VectorEnv
from labml import monit, tracker, logger

from .loss import ClippedPPOLoss, ClippedValueFunctionLoss
from .config import PPOConfig, AdvantageNormalizeOptions
from ..ac.ac import ACAgent
from ...utils.replay import ReplayBuffer
from ...utils.gae import GAE
from ...utils.math import normalize


class PPOAgent(ACAgent):
    """
    ## PPOAgent
    """

    def __init__(
        self,
        config: PPOConfig,
        envs: VectorEnv,
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
        self.num_envs = envs.num_envs
        # number of steps to run on each process for a single update
        self.env_steps = config.env_steps
        # number of mini batches
        self.batches = config.batches
        # total number of samples for a single update
        self.batch_size = self.num_envs * self.env_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.batches
        assert self.batch_size % self.batches == 0

        self.value_loss_coef = config.value_loss_coef
        self.entropy_bonus_coef = config.entropy_bonus_coef
        self.clip_range = config.clip_range
        self.advantage_normlize_option = config.advantage_normalize_option

        # #### Initialize

        # GAE and replay buffer
        gae = GAE(config.gamma, config.gae_lambda, self.num_envs, self.env_steps)
        self.replay_buffer = ReplayBuffer(self.env_steps, self.envs, gae, self.device)

        # Loss
        self.ppo_loss = ClippedPPOLoss()
        self.value_loss = ClippedValueFunctionLoss()

    def sample(self) -> dict[str, torch.Tensor]:
        """
        ### Sample data with current policy
        """
        self.replay_buffer.clear()
        with torch.no_grad():
            for t in range(self.env_steps):
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `num_envs`
                if self.is_model_shared:
                    pi, v = self.model(torch.from_numpy(self.obs).to(self.device))
                else:
                    pi = self.actor_model(torch.from_numpy(self.obs).to(self.device))
                    v = self.critic_model(torch.from_numpy(self.obs).to(self.device))
                values = v.cpu().numpy()
                a = pi.sample()
                actions = a.cpu().numpy()
                log_probs = pi.log_prob(a).cpu().numpy()
                log_pis = log_probs

                # run sampled actions on each worker
                self.obs, rewards, done, truncation, info = self.envs.step(actions)

                self.replay_buffer.append(
                    t, obs, actions, rewards, done, log_pis, values
                )

                # collect episode info, which is available if an episode finished;
                if "final_info" in info.keys():
                    for info_dict in info["final_info"]:
                        if info_dict is not None:
                            if (
                                "reward" in info_dict.keys()
                                and "length" in info_dict.keys()
                            ):
                                tracker.add("reward", info_dict["reward"])
                                tracker.add("length", info_dict["length"])

            # Get value of after the final step
            if self.is_model_shared:
                _, v = self.model(torch.from_numpy(self.obs).to(self.device))
            else:
                v = self.critic_model(torch.from_numpy(self.obs).to(self.device))
            last_values = v.cpu().numpy()
            self.replay_buffer.append_last_values(last_values)

        # calculate advantages
        self.replay_buffer.calc_advantages()

        self.replay_buffer.flatten_temp(
            self.advantage_normlize_option == AdvantageNormalizeOptions.batch
        )
        return self.replay_buffer.experience_pool

    def train(self, samples: dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.epochs):
            # shuffle for each epoch
            indexes = torch.randperm(self.batch_size)

            # for each mini batch
            for start in range(0, self.batch_size, self.mini_batch_size):
                # get mini batch
                end = start + self.mini_batch_size
                mini_batch_indexes = indexes[start:end]
                mini_batch = {}
                for k, v in samples.items():
                    mini_batch[k] = v[mini_batch_indexes]

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
            pi, value = self.model(samples["obs"])
        else:
            pi = self.actor_model(samples["obs"])
            value = self.critic_model(samples["obs"])

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
        value_loss = self.value_loss(
            value, samples["values"], sampled_return, self.clip_range
        )

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

    def run_training_loop(self, global_step: int = 0):
        """
        ### Run training loop
        """

        for update in monit.loop(range(global_step + 1, self.updates)):
            # sample with current policy
            samples = self.sample()

            # train the model
            self.train(samples)

            # Save tracked indicators.
            tracker.save()
            # Add a new line to the screen periodically
            if (update + 1) % 1_000 == 0:
                logger.log()

            self.save_model()
