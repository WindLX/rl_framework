import numpy as np
import torch
from gymnasium.vector import VectorEnv

from .gae import GAE
from .math import normalize


class ReplayBuffer:
    def __init__(
        self,
        env_steps: int,
        envs: VectorEnv,
        gae: GAE,
        device: str,
    ):
        self.device = device
        self.gae = gae

        obs_shape = envs.observation_space.shape
        obs_type = envs.observation_space.dtype

        action_shape = envs.action_space.shape
        action_type = envs.action_space.dtype

        num_envs = obs_shape[0]

        # temp data which collected in one episode
        self.obs_buffer = np.zeros(
            (num_envs, env_steps, *obs_shape[1:]), dtype=obs_type
        )
        self.rewards = np.zeros((num_envs, env_steps), dtype=np.float32)
        self.actions = np.zeros(
            (num_envs, env_steps, *action_shape[1:]), dtype=action_type
        )
        self.done = np.zeros((num_envs, env_steps), dtype=np.uint8)
        self.log_pis = np.zeros((num_envs, env_steps), dtype=np.float32)
        self.values = np.zeros((num_envs, env_steps + 1), dtype=np.float32)

        # Experience pool
        self.experience_pool = {
            "obs": None,
            "actions": None,
            "values": None,
            "log_pis": None,
            "advantages": None,
        }

    def append(
        self,
        env_step: int,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        done: np.ndarray,
        log_pis: np.ndarray,
        values: np.ndarray,
    ):
        """append the data collected in one step to the replay buffer

        Args:
            env_step (int): the step index in one episode
            obs (np.ndarray): the observation of the environment
            actions (np.ndarray): the action taken by the agent
            rewards (np.ndarray): the reward received by the agent
            done (np.ndarray): whether the episode is finished
            log_pis (np.ndarray): the log probability of the action taken by the agent
            values (np.ndarray): the value of the state observed by the agent
        """
        self.obs_buffer[:, env_step] = obs
        self.rewards[:, env_step] = rewards
        self.actions[:, env_step] = actions
        self.done[:, env_step] = done
        self.log_pis[:, env_step] = log_pis
        self.values[:, env_step] = values

    def append_last_values(self, values: np.ndarray):
        """append the last value of the episode to the replay buffer

        Args:
            values (np.ndarray): the last value of the episode
        """
        self.values[:, -1] = values

    def calc_advantages(self):
        """calculate the advantages using Generalized Advantage Estimation (GAE), this function should be called after one episode is finished"""
        self.advantages = self.gae(self.done, self.rewards, self.values)

    def clear_temp(self):
        """clear the temp data which collected in one episode"""
        self.obs_buffer = np.zeros_like(self.obs_buffer)
        self.rewards = np.zeros_like(self.rewards)
        self.actions = np.zeros_like(self.actions)
        self.done = np.zeros_like(self.done)
        self.log_pis = np.zeros_like(self.log_pis)
        self.values = np.zeros_like(self.values)
        if hasattr(self, "advantages"):
            self.advantages = np.zeros_like(self.advantages)

    def flatten_temp(self, normalize_advantage: bool):
        """flatten the temp data which collected in one episode, convert to tensor adn move to the experience pool

        Args:
            normalize_advantage (bool, optional): whether to normalize the advantages. Defaults to True.
        """
        samples = {
            "obs": self.obs_buffer,
            "actions": self.actions,
            "values": self.values[:, :-1],
            "log_pis": self.log_pis,
        }
        if hasattr(self, "advantages"):
            samples["advantages"] = self.advantages
        samples_flat = {}
        for k, v in samples.items():
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])
            if k == "advantages" and normalize_advantage:
                samples_flat[k] = normalize(torch.tensor(v, device=self.device))
            else:
                samples_flat[k] = torch.tensor(v, device=self.device)

        for k, v in samples_flat.items():
            self.experience_pool[k] = v

    def clear(self):
        """clear the experience pool"""
        self.clear_temp()
        self.experience_pool = {
            "obs": None,
            "actions": None,
            "values": None,
            "log_pis": None,
            "advantages": None,
        }
