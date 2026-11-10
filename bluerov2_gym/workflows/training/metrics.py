"""Observe physical position error without changing actions, rewards or RNG."""
from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PositionErrorCallback(BaseCallback):
    """Log rollout error and the rolling mean over 100 completed episodes."""
    def __init__(self):
        super().__init__()
        self.completed = deque(maxlen=100)

    def _on_training_start(self):
        self.episode_sum = np.zeros(self.training_env.num_envs)
        self.episode_count = np.zeros(self.training_env.num_envs, dtype=int)
        self.errors = []
        self.yaw_errors = []

    def _on_rollout_start(self):
        self.errors = []
        self.yaw_errors = []

    def _on_step(self):
        for i, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            yaw = info.get('metrics/yaw_error_abs_deg')
            if yaw is not None and np.isfinite(yaw):
                self.yaw_errors.append(float(yaw))
            value = info.get('metrics/position_error_euclidean')
            if value is not None and np.isfinite(value):
                value = float(value)
                self.errors.append(value)
                self.episode_sum[i] += value
                self.episode_count[i] += 1
            if done:
                if self.episode_count[i]:
                    self.completed.append(self.episode_sum[i] / self.episode_count[i])
                self.episode_sum[i] = 0.
                self.episode_count[i] = 0
        return True

    def _on_rollout_end(self):
        if self.yaw_errors:
            yaw = np.asarray(self.yaw_errors)
            self.logger.record('rollout/yaw_error_mean_deg', float(yaw.mean()))
            self.logger.record('rollout/yaw_error_rmse_deg', float(np.sqrt(np.mean(yaw**2))))
        if self.errors:
            values = np.asarray(self.errors)
            self.logger.record('rollout/position_error_mean_m', float(values.mean()))
            self.logger.record('rollout/position_error_rmse_m', float(np.sqrt(np.mean(values**2))))
        if self.completed:
            self.logger.record('rollout/episode_position_error_mean_m', float(np.mean(self.completed)))
