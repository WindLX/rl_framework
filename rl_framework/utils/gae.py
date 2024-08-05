from numpy import ndarray, float32, zeros


class GAE:
    def __init__(self, gamma: float, lambda_: float) -> None:
        """build GAE object

        Args:
            gamma (float): discount factor
            lambda_ (float): GAE parameter
        """

        self.gamma = gamma
        self.lambda_ = lambda_

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
        env_steps = len(rewards)
        assert len(values) == env_steps + 1
        assert len(dones) == env_steps

        # advantages table
        advantages = zeros((env_steps), dtype=float32)
        next_advantage = 0

        # $V(s_{t+1})$
        next_value = values[-1]

        for t in reversed(range(env_steps)):
            # mask if episode completed after step $t$
            mask = 1 - dones[t]
            next_value = next_value * mask
            next_advantage = next_advantage * mask

            # $\delta_{t}$
            delta = rewards[t] + self.gamma * next_value - values[t]

            # $\hat A_{t}=\delta_{t}+\gamma\lambda\hat A_{t+1}$
            advantages[t] = delta + self.gamma * self.lambda_ * next_advantage
            next_advantage = advantages[t]
            next_value = values[t]
        return advantages
