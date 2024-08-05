import numpy as np

from rl_framework.utils.gae import GAE
from rl_framework.utils.replay import Trajectory, ReplayBuffer


class TestTrajectory:
    def test_trajectory(self):
        obs = np.array([1.0, 2.0, 3.0])
        trajectory = Trajectory(["value", "log_pi"])
        trajectory.reset(obs)
        assert trajectory.log_pis == []
        trajectory.append(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.0, False, value=1.0, log_pi=0.5
        )
        trajectory.append(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.0, False, value=1.0, log_pi=0.5
        )
        trajectory.append(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.0, False, value=1.0, log_pi=0.5
        )
        trajectory.append(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.0, True, value=1.0, log_pi=0.5
        )
        trajectory.append(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.0, True, value=1.0, log_pi=0.5
        )
        trajectory.append(
            np.array([1.0, 2.0, 3.0]), 0.0, 0.0, True, value=1.0, log_pi=0.5
        )
        assert len(trajectory) == 4
        trajectory.clip()
        assert len(trajectory.actions) == 4
        assert len(trajectory.values) == 4
        assert len(trajectory.obss) == 5
