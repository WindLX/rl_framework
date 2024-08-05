import numpy as np
import torch

from .gae import GAE
from .math import normalize


class Trajectory:
    def __init__(self, extra: list[str] = []) -> None:
        """Initialize the trajectory

        Args:
            extra (list[str]): the extra information to be stored in the trajectory
        """
        self.obss = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.extra = list(map(lambda x: f"{x}s", extra))
        for e in self.extra:
            setattr(self, e, [])

    def clear(self) -> None:
        """Clear the trajectory"""
        self.obss = []
        self.actions = []
        self.rewards = []
        self.dones = []
        for e in self.extra:
            setattr(self, e, [])

    def reset(self, obs: np.ndarray) -> None:
        """Reset the trajectory

        Args:
            obs (np.ndarray): the observation of the environment
        """
        self.clear()
        self.obss = [obs]

    def append(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        done: bool,
        **kwargs,
    ) -> None:
        """Append the data collected in one step to the trajectory

        Args:
            obs (np.ndarray): the observation of the environment
            action (np.ndarray): the action taken by the agent
            reward (float): the reward received by the agent
            done (bool): whether the episode is finished
            kwargs (dict): the extra information to be stored in the trajectory
        """
        self.obss.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        for k, v in kwargs.items():
            setattr(self, f"{k}s", getattr(self, f"{k}s") + [v])

    def __len__(self) -> int:
        return (
            len(list(filter(lambda x: not x, self.dones))) + 1 if self.dones[-1] else 0
        )

    def clip(self) -> None:
        length = len(self)
        self.obss = self.obss[:length]
        self.actions = self.actions[:length]
        self.rewards = self.rewards[:length]
        self.dones = self.dones[:length]
        for e in self.extra:
            setattr(self, e, getattr(self, e)[:length])


class ReplayBuffer:
    def __init__(
        self,
        num_envs: int,
        gae: GAE,
        device: str,
    ):
        self.device = device
        self.gae = gae

        self.trajectories = [Trajectory(["value", "log_pi"]) for _ in range(num_envs)]

        # Experience pool
        self._experience_pool = {
            "obss": None,
            "actions": None,
            "values": None,
            "log_pis": None,
            "advantages": None,
        }

    def reset(self, obs: np.ndarray) -> None:
        for i, trajectory in enumerate(self.trajectories):
            trajectory.reset(obs[i])

    def append(
        self,
        next_obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
        log_pi: np.ndarray,
    ):
        """append the data collected in one step to the replay buffer

        Args:
            next_obs (np.ndarray): the next observation of the environment
            actions (np.ndarray): the action taken by the agent
            rewards (np.ndarray): the reward received by the agent
            done (np.ndarray): whether the episode is finished
            value (np.ndarray): the value of the state
            log_pi (np.ndarray): the log probability of the action
        """
        for i, trajectory in enumerate(self.trajectories):
            trajectory.append(
                next_obs[i],
                actions[i],
                rewards[i],
                done[i],
                value=value[i],
                log_pi=log_pi[i],
            )

    def clip(self) -> None:
        """clip the data in the replay buffer"""
        for trajectory in self.trajectories:
            trajectory.clip()

    def append_last_values(self, values: np.ndarray):
        """append the last value of the episode to the replay buffer

        Args:
            values (np.ndarray): the last value of the episode
        """
        for i, trajectory in enumerate(self.trajectories):
            trajectory.values.append(values[i])

    def calc_advantages(self):
        """calculate the advantages using Generalized Advantage Estimation (GAE), this function should be called after one episode is finished"""
        self.advantages = []
        for trajectory in self.trajectories:
            advantage = self.gae(
                trajectory.dones, trajectory.rewards, trajectory.values
            )
            self.advantages.append(advantage)

    def flatten(self):
        """flatten the temp data which collected in one episode, convert to tensor adn move to the experience pool"""
        self._experience_pool["obss"] = torch.from_numpy(
            np.concatenate([trajectory.obss for trajectory in self.trajectories])
        ).to(self.device)
        self._experience_pool["actions"] = torch.from_numpy(
            np.concatenate([trajectory.actions for trajectory in self.trajectories])
        ).to(self.device)
        self._experience_pool["values"] = torch.from_numpy(
            np.concatenate([trajectory.values[:-1] for trajectory in self.trajectories])
        ).to(self.device)
        self._experience_pool["log_pis"] = torch.from_numpy(
            np.concatenate([trajectory.log_pis for trajectory in self.trajectories])
        ).to(self.device)
        self._experience_pool["advantages"] = torch.from_numpy(
            np.concatenate(self.advantages)
        ).to(self.device)

    def normalize_advantages(self):
        """normalize the advantages in the experience pool"""
        self._experience_pool["advantages"] = normalize(
            self._experience_pool["advantages"]
        )

    def clear_trajectories(self):
        """clear the trajectories in the replay buffer"""
        for trajectory in self.trajectories:
            trajectory.clear()

    def clear(self):
        """clear the experience pool"""
        self.clear_trajectories()
        self._experience_pool = {
            "obss": None,
            "actions": None,
            "values": None,
            "log_pis": None,
            "advantages": None,
        }

    @property
    def experience(self):
        return self._experience_pool
