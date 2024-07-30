from typing import Dict, Union
from os import path

import numpy as np

import torch
from torch import optim

from labml import monit, tracker, logger
from labml_helpers.device import DeviceConfigs

from .loss import ClippedPPOLoss, ClippedValueFunctionLoss
from .config import PPOConfigs, PPOEvalConfigs
from ...utils.gae import GAE
from ...utils.math import normalize
from ...utils.error import SharedACException


class PPOAgent:
    """
    ## PPOAgent
    """

    def __init__(
        self,
        config: PPOConfigs,
    ):
        # #### Configurations

        # number of updates
        self.updates = config.updates
        # number of epochs to train the model with sampled data
        self.epochs = config.epochs
        # number of worker processes
        self.num_envs = config.envs.num_envs
        # number of steps to run on each process for a single update
        self.env_steps = config.env_steps
        # number of mini batches
        self.batches = config.batches
        # total number of samples for a single update
        self.batch_size = self.num_envs * self.env_steps
        # size of a mini batch
        self.mini_batch_size = self.batch_size // self.batches
        assert self.batch_size % self.batches == 0

        # Value loss coefficient
        self.value_loss_coef = config.value_loss_coef
        # Entropy bonus coefficient
        self.entropy_bonus_coef = config.entropy_bonus_coef

        # Clipping range
        self.clip_range = config.clip_range
        # Learning rate
        self.learning_rate = config.learning_rate

        # save_path
        self.save_path = config.save_path

        # device
        self.device = DeviceConfigs().device

        # #### Initialize

        # create workers
        self.envs = config.envs

        # initialize tensors for observations
        self.obs, info = self.envs.reset()

        # model
        self.is_model_shared = config.is_model_shared
        checkpoint = None
        if self.save_path and path.exists(self.save_path):
            checkpoint = torch.load(self.save_path, weights_only=True)

        if self.is_model_shared:
            if type(config.model) == tuple:
                raise SharedACException(
                    "Model should be single model for shared actor-critic model"
                )
            self.model = config.model.to(self.device)
            # optimizer
            self.optimizer = optim.Adam(self.model.parameters(), lr=2.5e-4)

            if checkpoint:
                self.model.load_state_dict(checkpoint["shared_model"])
                logger.log("Model loaded from {}".format(self.save_path))

        else:
            if type(config.model) != tuple:
                raise SharedACException(
                    "Model should be tuple for separate actor-critic model"
                )
            self.actor_model = config.model[0].to(self.device)
            self.critic_model = config.model[1].to(self.device)
            # optimizer
            self.actor_optimizer = optim.Adam(self.actor_model.parameters(), lr=2.5e-4)
            self.critic_optimizer = optim.Adam(
                self.critic_model.parameters(), lr=2.5e-4
            )

            if checkpoint:
                self.actor_model.load_state_dict(checkpoint["actor_model"])
                self.critic_model.load_state_dict(checkpoint["critic_model"])
                logger.log("Model loaded from {}".format(self.save_path))

        # GAE
        self.gae = GAE(
            config.gae_gamma, config.gae_lambda, self.num_envs, self.env_steps
        )

        # PPO Loss
        self.ppo_loss = ClippedPPOLoss()

        # Value Loss
        self.value_loss = ClippedValueFunctionLoss()

    def sample(self) -> Dict[str, torch.Tensor]:
        """
        ### Sample data with current policy
        """
        obs_shape = self.envs.observation_space.shape
        obs_type = self.envs.observation_space.dtype
        action_shape = self.envs.action_space.shape
        action_type = self.envs.action_space.dtype

        obs = np.zeros((self.num_envs, self.env_steps, *obs_shape[1:]), dtype=obs_type)
        rewards = np.zeros((self.num_envs, self.env_steps), dtype=np.float32)
        actions = np.zeros(
            (self.num_envs, self.env_steps, *action_shape[1:]), dtype=action_type
        )
        done = np.zeros((self.num_envs, self.env_steps), dtype=np.uint8)
        log_pis = np.zeros((self.num_envs, self.env_steps), dtype=np.float32)
        values = np.zeros((self.num_envs, self.env_steps + 1), dtype=np.float32)

        with torch.no_grad():
            # sample `worker_steps` from each worker
            for t in range(self.env_steps):
                # `self.obs` keeps track of the last observation from each worker,
                #  which is the input for the model to sample the next action
                obs[:, t] = self.obs
                # sample actions from $\pi_{\theta_{OLD}}$ for each worker;
                #  this returns arrays of size `num_envs`
                if self.is_model_shared:
                    pi, v = self.model(torch.from_numpy(self.obs).to(self.device))
                else:
                    pi = self.actor_model(torch.from_numpy(self.obs).to(self.device))
                    v = self.critic_model(torch.from_numpy(self.obs).to(self.device))
                values[:, t] = v.cpu().numpy()
                a = pi.sample()
                actions[:, t] = a.cpu().numpy()
                # log_probs = joint_log_probs(pi, a).cpu().numpy()
                log_probs = pi.log_prob(a).cpu().numpy()
                log_pis[:, t] = log_probs

                # run sampled actions on each worker
                self.obs, rewards[:, t], done[:, t], truncation, info = self.envs.step(
                    actions[:, t]
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
            values[:, self.env_steps] = v.cpu().numpy()

        # calculate advantages
        advantages = self.gae(done, rewards, values)

        samples = {
            "obs": obs,
            "actions": actions,
            "values": values[:, :-1],
            "log_pis": log_pis,
            "advantages": advantages,
        }

        # samples are currently in `[workers, time_step]` table,
        # we should flatten it for training
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == "obs":
                samples_flat[k] = normalize(torch.from_numpy(v).to(self.device))
            else:
                samples_flat[k] = torch.tensor(v, device=self.device)

        return samples_flat

    def train(self, samples: Dict[str, torch.Tensor]):
        """
        ### Train the model based on samples
        """

        # It learns faster with a higher number of epochs,
        #  but becomes a little unstable; that is,
        #  the average episode reward does not monotonically increase
        #  over time.
        # May be reducing the clipping range might solve it.
        for _ in range(self.epochs()):
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

                if self.is_model_shared:
                    # train
                    loss = self._calc_loss(mini_batch)

                    # Set learning rate
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate()
                    # Zero out the previously calculated gradients
                    self.optimizer.zero_grad()
                    # Calculate gradients
                    loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=0.5
                    )
                    # Update parameters based on gradients
                    self.optimizer.step()
                else:
                    # train
                    actor_loss, critic_loss = self._calc_loss(mini_batch)

                    # Set learning rate
                    for pg in self.actor_optimizer.param_groups:
                        pg["lr"] = self.learning_rate()
                    for pg in self.critic_optimizer.param_groups:
                        pg["lr"] = self.learning_rate()
                    # Zero out the previously calculated gradients
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    # Calculate gradients
                    actor_loss.backward()
                    critic_loss.backward()
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(
                        self.actor_model.parameters(), max_norm=0.5
                    )
                    torch.nn.utils.clip_grad_norm_(
                        self.critic_model.parameters(), max_norm=0.5
                    )
                    # Update parameters based on gradients
                    self.actor_optimizer.step()
                    self.critic_optimizer.step()

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

    def _calc_loss(
        self, samples: Dict[str, torch.Tensor]
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
        sampled_normalized_advantage = normalize(samples["advantages"])

        # Sampled observations are fed into the model to get $\pi_\theta(a_t|s_t)$ and $V^{\pi_\theta}(s_t)$;
        #  we are treating observations as state
        if self.is_model_shared:
            pi, value = self.model(samples["obs"])
        else:
            pi = self.actor_model(samples["obs"])
            value = self.critic_model(samples["obs"])

        # $-\log \pi_\theta (a_t|s_t)$, $a_t$ are actions sampled from $\pi_{\theta_{OLD}}$
        # log_pi = joint_log_probs(pi, samples["actions"])
        log_pi = pi.log_prob(samples["actions"])

        # Calculate policy loss
        policy_loss = self.ppo_loss(
            log_pi, samples["log_pis"], sampled_normalized_advantage, self.clip_range()
        )

        # Calculate Entropy Bonus
        #
        # $\mathcal{L}^{EB}(\theta) =
        #  \mathbb{E}\Bigl[ S\bigl[\pi_\theta\bigr] (s_t) \Bigr]$
        entropy_bonus = pi.entropy()
        entropy_bonus = entropy_bonus.mean()

        # Calculate value function loss
        value_loss = self.value_loss(
            value, samples["values"], sampled_return, self.clip_range()
        )

        # $\mathcal{L}^{CLIP+VF+EB} (\theta) =
        #  \mathcal{L}^{CLIP} (\theta) +
        #  c_1 \mathcal{L}^{VF} (\theta) - c_2 \mathcal{L}^{EB}(\theta)$
        if self.is_model_shared:
            loss = (
                policy_loss
                + self.value_loss_coef() * value_loss
                - self.entropy_bonus_coef() * entropy_bonus
            )
        else:
            actor_loss = policy_loss - self.entropy_bonus_coef() * entropy_bonus
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

    def run_training_loop(self, global_step: int):
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

            if self.save_path:
                torch.save({"global_step": update, **self.models_dict}, self.save_path)

    def destroy(self):
        """
        ### Destroy
        Stop the workers
        """
        self.envs.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()


class PPOEvalAgent:
    def __init__(self, config: PPOEvalConfigs) -> None:
        self.env = config.env
        model = config.model
        save_path = config.save_path
        self.is_render = config.is_render
        self.is_model_shared = config.is_model_shared
        self.device = DeviceConfigs().device

        checkpoint = None
        if save_path and path.exists(save_path):
            checkpoint = torch.load(save_path, weights_only=True)

        if self.is_model_shared:
            model = model.to(self.device)
            if checkpoint:
                model.load_state_dict(checkpoint["shared_model"])
                logger.log("Model loaded from {}".format(save_path))
            model.eval()
            self.model = model

        else:
            actor_model = model.to(self.device)

            if checkpoint:
                actor_model.load_state_dict(checkpoint["actor_model"])
                logger.log("Model loaded from {}".format(save_path))
            actor_model.eval()
            self.actor_model = actor_model

    def run_eval_loop(self, updates: int):
        for update in monit.loop(updates):
            obs, info = self.env.reset()
            done = False

            def steps():
                step = 0
                while not done:
                    yield step
                    step += 1

            for step in steps():
                obs = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )

                if self.is_model_shared:
                    pi, _ = self.model(obs)
                else:
                    pi = self.actor_model(obs)
                action = pi.sample().squeeze(0).cpu().numpy()

                obs, reward, done, truncated, info = self.env.step(action)
                entropy_bonus = pi.entropy()

                if self.is_render:
                    self.env.render()

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
