import numpy as np


class Reward:
    def __init__(
        self,
        use_energy_penalty: bool = False,
        w_energy: float = 0.005,
        use_smoothness_penalty: bool = True,
        w_smooth: float = 0.05,
        action_dim: int = 8,
        action_clip: float = 1.0,
    ):
        self.w_pos = 1.0
        self.w_vel = 0.1
        self.w_ang_pos = 0.5
        self.w_stab = 0.5

        self.use_energy_penalty = use_energy_penalty
        self.w_energy = w_energy

        self.use_smoothness_penalty = use_smoothness_penalty
        self.w_smooth = w_smooth

        self.action_dim = action_dim
        self.action_clip = action_clip

        self.prev_action = None

    def reset(self):
        self.prev_action = None

    def _to_scalar(self, value):
        if isinstance(value, np.ndarray):
            return float(value.item())
        return float(value)

    def _prepare_action(self, action):
        if action is None:
            return None

        action = np.asarray(action, dtype=float).reshape(-1)

        action = np.clip(action, -self.action_clip, self.action_clip)
        return action

    def _action_l2_penalty(self, action):
        if action is None:
            return 0.0

        penalty = np.sum(action ** 2) / max(1, self.action_dim)
        return float(penalty)

    def _delta_u_penalty(self, action):
        if action is None or self.prev_action is None:
            return 0.0

        delta_u = action - self.prev_action
        penalty = np.sum(delta_u ** 2) / max(1, self.action_dim)
        return float(penalty)

    def get_reward(self, obs, action=None):
        action = self._prepare_action(action)

        x = self._to_scalar(obs["x"])
        y = self._to_scalar(obs["y"])
        z = self._to_scalar(obs["z"])

        u = self._to_scalar(obs["u"])
        v = self._to_scalar(obs["v"])
        w = self._to_scalar(obs["w"])

        roll = self._to_scalar(obs["roll"])
        pitch = self._to_scalar(obs["pitch"])
        yaw = self._to_scalar(obs["yaw"])

        position_error = np.sqrt(x**2 + y**2 + z**2)

        velocity_penalty = np.sqrt(u**2 + v**2 + w**2)

        yaw_error = np.arctan2(np.sin(yaw), np.cos(yaw))
        yaw_error = abs(yaw_error)

        stability_penalty = abs(roll) + abs(pitch)

        reward = -(
            self.w_pos * position_error
            + self.w_vel * velocity_penalty
            + self.w_ang_pos * yaw_error
            + self.w_stab * stability_penalty
        )

        if self.use_energy_penalty:
            action_penalty = self._action_l2_penalty(action)
            reward -= self.w_energy * action_penalty

        if self.use_smoothness_penalty:
            smooth_penalty = self._delta_u_penalty(action)
            reward -= self.w_smooth * smooth_penalty

        if position_error < 0.2:
            reward += 1.0

        if action is not None:
            self.prev_action = action.copy()

        return float(reward)