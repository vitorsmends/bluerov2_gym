"""Advance curriculum only after sustained physical stationkeeping success."""
from collections import deque
import math

from stable_baselines3.common.callbacks import BaseCallback


class CurriculumEarlyStopped(RuntimeError):
    """The curriculum stopped for stagnation without completing every phase."""


class PlateauEarlyStoppingCallback(BaseCallback):
    """Optional stop after consecutive episode windows without error improvement."""
    def __init__(self, window_episodes=20, patience_windows=10,
                 min_delta_m=.01, warmup_episodes=40, report=None):
        super().__init__()
        if (window_episodes < 1 or patience_windows < 1 or warmup_episodes < 0
                or not math.isfinite(min_delta_m) or min_delta_m < 0):
            raise ValueError('Invalid early stopping window, patience, warmup or minimum improvement.')
        self.window_episodes = int(window_episodes)
        self.patience_windows = int(patience_windows)
        self.min_delta_m = float(min_delta_m)
        self.warmup_episodes = int(warmup_episodes)
        self.report = report
        self.episodes = 0
        self.best_error_m = None
        self.stale_windows = 0
        self.stopped = False
        self.window = []

    def _on_training_start(self):
        self.sums = [0.] * self.training_env.num_envs
        self.counts = [0] * self.training_env.num_envs
        self.valid = [True] * self.training_env.num_envs

    def _on_step(self):
        for index, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            error = info.get('metrics/position_error_euclidean')
            if error is None or not math.isfinite(error):
                self.valid[index] = False
            else:
                self.sums[index] += error
                self.counts[index] += 1
            if not done:
                continue
            self.episodes += 1
            if self.episodes > self.warmup_episodes and self.valid[index] and self.counts[index]:
                self.window.append(self.sums[index] / self.counts[index])
            self.sums[index] = 0.
            self.counts[index] = 0
            self.valid[index] = True
            if len(self.window) < self.window_episodes:
                continue
            error_mean = sum(self.window) / len(self.window)
            self.window.clear()
            improved = self.best_error_m is None or self.best_error_m - error_mean > self.min_delta_m
            if improved:
                self.best_error_m = error_mean
                self.stale_windows = 0
            else:
                self.stale_windows += 1
            self.stopped = self.stale_windows >= self.patience_windows
            self.logger.record('curriculum/early_stopping_error_mean_m', error_mean)
            self.logger.record('curriculum/early_stopping_stale_windows', self.stale_windows)
            if self.report:
                self.report(self, improved)
            if self.stopped:
                return False
        return True


class StationkeepingMasteryCallback(BaseCallback):
    def __init__(self, radius_m, hold_steps, consecutive_episodes, report=None):
        super().__init__()
        if not math.isfinite(radius_m) or radius_m <= 0 or hold_steps < 1 or consecutive_episodes < 1:
            raise ValueError('Mastery radius, hold steps and episode count must be positive.')
        self.radius_m = float(radius_m)
        self.hold_steps = int(hold_steps)
        self.required_episodes = int(consecutive_episodes)
        self.report = report
        self.mastered = False
        self.streak = 0
        self.episodes = 0

    def _on_training_start(self):
        self.tails = [deque(maxlen=self.hold_steps) for _ in range(self.training_env.num_envs)]

    def _on_step(self):
        for index, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            error = info.get('metrics/position_error_euclidean')
            valid = error is not None and math.isfinite(error) and error < self.radius_m
            self.tails[index].append(valid)
            if not done:
                continue
            # Only a normal time-limit ending can qualify. A physical failure
            # must never unlock the next phase, even if its position is good.
            success = (bool(info.get('TimeLimit.truncated', False))
                       and len(self.tails[index]) == self.hold_steps
                       and all(self.tails[index]))
            self.episodes += 1
            self.streak = self.streak + 1 if success else 0
            self.tails[index].clear()
            self.mastered = self.streak >= self.required_episodes
            self.logger.record('curriculum/successful_episode_streak', self.streak)
            self.logger.record('curriculum/mastered', float(self.mastered))
            if self.report:
                self.report(self)
            if self.mastered:
                return False
        return True
