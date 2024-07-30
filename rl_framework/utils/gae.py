from numpy import ndarray, float32, zeros


class GAE:
    def __init__(
        self, gamma: float, lambda_: float, num_envs: int, env_steps: int
    ) -> None:
        """build GAE object

        Args:
            gamma (float): discount factor
            lambda_ (float): GAE parameter
            num_envs (int): number of env
            env_steps (int): number of steps each env takes
        """

        self.gamma = gamma
        self.lambda_ = lambda_
        self.num_envs = num_envs
        self.env_steps = env_steps

    def __call__(self, dones: ndarray, rewards: ndarray, values: ndarray) -> ndarray:
        """
        Calculate the Generalized Advantage Estimation (GAE) for each step in the trajectory.

        Args:
            dones (numpy.ndarray): An array of boolean values indicating whether the episode has ended at each step.
            rewards (numpy.ndarray): An array of rewards received at each step.
            values (numpy.ndarray): An array of value estimates at each step.

        Returns:
            numpy.ndarray: An array of GAE estimates for each step.
        """

        # advantages table
        advantages = zeros((self.num_envs, self.env_steps), dtype=float32)
        last_advantage = 0

        # $V(s_{t+1})$
        last_value = values[:, -1]

        for t in reversed(range(self.env_steps)):
            # mask if episode completed after step $t$
            mask = 1 - dones[:, t]
            last_value = last_value * mask
            last_advantage = last_advantage * mask

            # $\delta_{t}$
            delta = rewards[:, t] + self.gamma * last_value - values[:, t]

            # $\hat A_{t}=\delta_{t}+\gamma\lambda\hat A_{t+1}$
            last_advantage = delta + self.gamma * self.lambda_ * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]
        return advantages
