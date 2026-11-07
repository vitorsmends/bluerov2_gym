"""Unmodified 889a766 reset/step methods, with the current simulator injected."""
import hashlib
from pathlib import Path

import numpy as np

from bluerov2_gym.envs.bluerov_env import BlueRov
from .reference_sources import trajectory_889a766 as reference


SOURCE_SHA256 = '44758d653df6a9d843d349664cb62730d9d37887311edcee75f3ff76b5697424'


class Frozen889TrajectoryEnv(reference.TrajectoryTrackingEnv):
    """Only construction and diagnostic info differ from the archived class.

    The historical constructor cannot accept simulator configuration. Initialize
    the current BlueRov explicitly and reproduce the remaining three assignments.
    Reference generation, reset and step come from the archived source.
    """

    def __init__(self, env_config=None, dynamics_config=None):
        source = Path(reference.__file__)
        if hashlib.sha256(source.read_bytes()).hexdigest() != SOURCE_SHA256:
            raise RuntimeError('Archived 889a766 source was modified.')
        BlueRov.__init__(self, render_mode=None, env_config=env_config,
                         dynamics_config=dynamics_config)
        if not np.isclose(self.dt, .1) or not np.isclose(self.dynamics.dt, .1):
            raise ValueError('The 889a766 recipe requires dt=0.1 s.')
        self.traj = reference.TrajectoryGenerator()
        self.current_t = 0.0
        self.dt = 0.1

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        # Logging only: base BlueRov reports distance to the origin, not target.
        info['metrics/position_error_euclidean'] = info['tracking_error_m']
        info['metrics/trajectory_reward_raw'] = reward
        return obs, reward, terminated, truncated, info
