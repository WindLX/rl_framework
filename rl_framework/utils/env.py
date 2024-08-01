from gymnasium import Wrapper


class RewardSumWrapper(Wrapper):
    """Wrapper for RL environment"""

    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        self.rewards = []

    def reset(self, **kwargs):
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.rewards.append(reward)
        if terminated or truncated:
            reward = sum(self.rewards)
            length = len(self.rewards)
            info.update({"reward": reward, "length": length})
        return obs, reward, terminated, truncated, info
