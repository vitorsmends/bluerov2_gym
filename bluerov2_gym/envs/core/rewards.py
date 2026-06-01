import numpy as np


class Reward:
    def __init__(self):
        # Tracking / station keeping
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_yaw = 0.5
        self.w_stab = 0.5

        # Thruster effort
        self.w_thrust = 0.002
        self.w_thrust_rate = 0.001

        # Success bonus
        self.success_radius = 0.20
        self.success_bonus = 1.0

        self.prev_action = np.zeros(6, dtype=float)

    def _scalar(self, value):
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def _wrap_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))

    def reset(self):
        self.prev_action[:] = 0.0

    def get_reward(self, obs, action=None):
        x = self._scalar(obs["x"])
        y = self._scalar(obs["y"])
        z = self._scalar(obs["z"])

        u = self._scalar(obs["u"])
        v = self._scalar(obs["v"])
        w = self._scalar(obs["w"])

        roll = self._scalar(obs["roll"])
        pitch = self._scalar(obs["pitch"])
        yaw = self._scalar(obs["yaw"])

        # Position error relative to origin
        position_error = np.sqrt(x**2 + y**2 + z**2)

        # Linear velocity penalty
        velocity_penalty = np.sqrt(u**2 + v**2 + w**2)

        # Yaw error relative to zero heading
        yaw_error = abs(self._wrap_angle(yaw))

        # Passive stability penalty
        stability_penalty = abs(roll) + abs(pitch)

        reward = -(
            self.w_pos * position_error
            + self.w_vel * velocity_penalty
            + self.w_yaw * yaw_error
            + self.w_stab * stability_penalty
        )

        if action is not None:
            action = np.asarray(action, dtype=float)

            if action.shape != (6,):
                raise ValueError(
                    f"Action must have shape (6,), got {action.shape}. "
                    "Expected [T1, T2, T3, T4, T5, T6]."
                )

            action = np.clip(action, -40.0, 40.0)

            # Normalized effort: 0 to approximately 1
            thrust_effort = np.mean((action / 40.0) ** 2)

            # Penalizes abrupt thruster changes
            thrust_rate = np.mean(((action - self.prev_action) / 40.0) ** 2)

            reward -= (
                self.w_thrust * thrust_effort
                + self.w_thrust_rate * thrust_rate
            )

            self.prev_action = action.copy()

        if position_error < self.success_radius:
            reward += self.success_bonus

        return float(reward)