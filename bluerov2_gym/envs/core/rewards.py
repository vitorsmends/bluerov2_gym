import numpy as np


class Reward:
    def __init__(self):
        # Original reward strategy
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_yaw = 0.5
        self.w_stab = 0.5

        self.success_radius = 0.20
        self.success_bonus = 1.0

    def _scalar(self, value):
        return float(np.asarray(value).reshape(-1)[0])

    def _wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def reset(self):
        pass

    def get_reward(self, obs, action=None):
        # action is accepted only for compatibility with the updated environment.
        # It is intentionally not used, preserving the original reward strategy.

        x = self._scalar(obs["x"])
        y = self._scalar(obs["y"])
        z = self._scalar(obs["z"])

        u = self._scalar(obs["u"])
        v = self._scalar(obs["v"])
        w = self._scalar(obs["w"])

        roll = self._scalar(obs["roll"])
        pitch = self._scalar(obs["pitch"])
        yaw = self._scalar(obs["yaw"])

        position_error = np.sqrt(x**2 + y**2 + z**2)
        velocity_penalty = np.sqrt(u**2 + v**2 + w**2)
        yaw_error = abs(self._wrap_angle(yaw))
        stability_penalty = abs(roll) + abs(pitch)

        reward = -(
            self.w_pos * position_error
            + self.w_vel * velocity_penalty
            + self.w_yaw * yaw_error
            + self.w_stab * stability_penalty
        )

        if position_error < self.success_radius:
            reward += self.success_bonus

        return float(reward)