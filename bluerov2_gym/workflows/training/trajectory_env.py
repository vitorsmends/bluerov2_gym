"""Reference trajectory feedback/reward/reset on top of the CURRENT simulator.

The historical reset returns absolute observations; subsequent steps return
virtual observations. Preserve this asymmetry, the u-only correction and the
original reward deliberately. Do not modify the underlying dynamics/disturbance.
"""
import numpy as np
from bluerov2_gym.envs.bluerov_env import BlueRov
from ..trajectory_reference import TrajectoryGenerator


class TrajectoryTrackingEnv(BlueRov):
    def __init__(self, env_config=None, dynamics_config=None):
        super().__init__(render_mode=None, env_config=env_config,
                         dynamics_config=dynamics_config)
        if not np.isclose(self.dt, 0.1) or not np.isclose(self.dynamics.dt, 0.1):
            raise ValueError('The reference trajectory trainer requires dt=0.1 s.')
        self.traj = TrajectoryGenerator()
        self.current_t = 0.0
        self.dt = 0.1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_t = np.random.uniform(0.0, 50.0)

        target_pos, target_att, _ = self.traj.get_state_at_time(self.current_t)

        noise_pos = np.random.uniform(-0.2, 0.2, 3)
        initial_pos = target_pos + noise_pos

        self.state = {
            "x": float(initial_pos[0]),
            "y": float(initial_pos[1]),
            "z": float(initial_pos[2]),
            "roll": 0.0,
            "pitch": 0.0,
            "yaw": float(target_att[2]),
            "u": 0.0,
            "v": 0.0,
            "w": 0.0,
            "p": 0.0,
            "q": 0.0,
            "r": 0.0,
        }

        if hasattr(self, "dynamics"):
            self.dynamics.reset()

        if hasattr(self, "reward_fn"):
            self.reward_fn.reset()

        return self._get_obs(), {}

    def step(self, action):
        self.current_t += self.dt

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        action = np.clip(action, -40.0, 40.0)

        obs, _, terminated, truncated, info = super().step(action)

        tgt_pos, tgt_att, tgt_vel = self.traj.get_state_at_time(self.current_t)

        curr_pos = np.array(
            [obs["x"][0], obs["y"][0], obs["z"][0]],
            dtype=np.float32,
        )

        curr_vel = np.array(
            [obs["u"][0], obs["v"][0], obs["w"][0]],
            dtype=np.float32,
        )

        error_pos = curr_pos - tgt_pos
        error_vel = curr_vel - tgt_vel

        psi = obs["yaw"][0]
        c, s = np.cos(psi), np.sin(psi)

        err_x_body = error_pos[0] * c + error_pos[1] * s
        err_y_body = -error_pos[0] * s + error_pos[1] * c
        err_z_body = error_pos[2]

        obs["x"] = np.array([err_x_body], dtype=np.float32)
        obs["y"] = np.array([err_y_body], dtype=np.float32)
        obs["z"] = np.array([err_z_body], dtype=np.float32)

        obs["u"] = np.array([error_vel[0]], dtype=np.float32)

        # Main metric: Euclidean trajectory error in meters
        dist = float(np.linalg.norm(error_pos))
        vel_err = float(np.linalg.norm(error_vel))

        act_cost = float(np.mean((action / 40.0) ** 2))

        reward = 1.0 - (2.0 * dist) - (0.1 * vel_err) - (0.001 * act_cost)

        success = dist < 0.20

        if dist > 3.0:
            terminated = True
            reward -= 10.0
            info["done_reason"] = "distance"
        else:
            info["done_reason"] = "none"

        # The base env logs distance to the world origin; the live plots
        # must report trajectory error for this task, not origin distance.
        info["metrics/position_error_euclidean"] = dist
        info["metrics/trajectory_reward_raw"] = float(reward)
        info["tracking_error_m"] = dist
        info["velocity_error"] = vel_err
        info["act_cost"] = act_cost
        info["is_success"] = success

        return obs, float(reward), terminated, truncated, info

