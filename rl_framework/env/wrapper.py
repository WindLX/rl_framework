from gymnasium import Wrapper


class RewardSumWrapper(Wrapper):
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


class SyncWrapper(Wrapper):
    def __init__(self, env, auto_block: bool = True, **kwargs):
        super().__init__(env, **kwargs)
        self.env = env
        self.guard = False
        self.auto_block = auto_block
        self.finish_obs = None
        self.len = 0

    def reset(self, **kwargs):
        self.unlock()
        self.len = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        if self.guard:
            return self.finish_obs, 0.0, True, True, None
        else:
            self.len += 1
            obs, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated and self.auto_block:
                self.lock()
            self.finish_obs = obs
            return obs, reward, terminated, truncated, info

    def lock(self):
        self.guard = True

    def unlock(self):
        self.guard = False

    @property
    def locked(self):
        return self.guard

    @property
    def length(self):
        return self.len

    @property
    def finished(self):
        return self.finish_obs is not None
