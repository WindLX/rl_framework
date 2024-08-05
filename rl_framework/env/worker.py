from typing import Any, SupportsFloat, Optional

import multiprocessing
import multiprocessing.connection

import numpy as np
from gymnasium import Env


def worker_process(remote: multiprocessing.connection.Connection, env: Env):
    env = env

    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            remote.send(env.step(data))
        elif cmd == "reset":
            remote.send(env.reset(**data))
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "lock":
            env.lock()
        elif cmd == "unlock":
            env.unlock()
        elif cmd == "render":
            env.render()
        elif cmd == "locked":
            remote.send(env.get_wrapper_attr("locked"))
        else:
            raise NotImplementedError


class Worker:
    def __init__(self, env: Env):
        self.sender, receiver = multiprocessing.Pipe()
        self.process = multiprocessing.Process(
            target=worker_process, args=(receiver, env)
        )
        self.process.start()
        self._observation_space = env.observation_space
        self._action_space = env.action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        self.sender.send(("step", action))
        return self.sender.recv()

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.sender.send(("reset", {"seed": seed, "options": options, **kwargs}))
        return self.sender.recv()

    def lock(self):
        self.sender.send(("lock", None))

    def unlock(self):
        self.sender.send(("unlock", None))

    def render(self):
        self.sender.send(("render", None))

    def locked(self):
        self.sender.send(("locked", None))
        return self.sender.recv()

    def close(self):
        self.sender.send(("close", None))


class WorkerSet:
    def __init__(self, env: Env, num_workers: int):
        self.workers = [Worker(env) for _ in range(num_workers)]

    def __len__(self):
        return len(self.workers)

    def step(
        self, actions: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
        recv = []
        for worker, action in zip(self.workers, actions):
            r = worker.step(action)
            recv.append(r)
        observation = np.stack([r[0] for r in recv]).astype(np.float32)
        reward = np.stack([r[1] for r in recv]).astype(np.float32)
        terminated = np.stack([r[2] for r in recv])
        truncated = np.stack([r[3] for r in recv])
        info = [r[4] for r in recv]
        return observation, reward, terminated, truncated, info

    def reset(
        self,
        seeds: Optional[list[int]] = None,
        options: Optional[list[dict[str, Any]]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, list[dict[str, Any]]]:
        if seeds is None:
            seeds = [None] * len(self.workers)
        if options is None:
            options = [None] * len(self.workers)
        recv = []
        for worker, seed, option in zip(self.workers, seeds, options):
            r = worker.reset(seed=seed, options=option, **kwargs)
            recv.append(r)
        observation = np.stack([r[0] for r in recv]).astype(np.float32)
        info = [r[1] for r in recv]
        return observation, info

    def close(self):
        for worker in self.workers:
            worker.close()
            worker.process.join()
        self.workers = []

    @property
    def is_all_locked(self):
        return all(worker.locked() for worker in self.workers)

    @property
    def observation_space(self):
        return self.workers[0].observation_space

    @property
    def action_space(self):
        return self.workers[0].action_space

    def length(self, index: int) -> int:
        return self.workers[index].length

    def finished(self, index: int):
        self.workers[index].finished
