"""Advance curriculum only after sustained physical stationkeeping success."""
from collections import deque
import math

from stable_baselines3.common.callbacks import BaseCallback


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
