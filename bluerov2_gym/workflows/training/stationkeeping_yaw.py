"""Stationkeeping with the historical direct-thrust PPO recipe and yaw-aware goals.

Use ``bluerov train --config configs/training/stationkeeping_waves_yaw.yaml``.
The shared curriculum trainer handles optimization, normalization and artifacts.
"""
import numpy as np
from bluerov2_gym.envs.bluerov_env import BlueRov


def stationkeeping_observation(obs, reference):
    """World position error, relative yaw, unchanged body velocities; fixed goals only."""
    ref = np.asarray(reference, dtype=float).reshape(12)
    if not np.isfinite(ref).all() or not np.allclose(ref[6:], 0.):
        raise ValueError('Stationkeeping requires a finite, stationary reference.')
    if not np.allclose(ref[3:5], 0.):
        raise ValueError('Stationkeeping requires zero target roll and pitch.')
    result = {k: np.asarray(v, dtype=np.float32).copy() for k, v in obs.items()}
    for key, goal in zip(('x', 'y', 'z'), ref[:3]):
        result[key] = np.asarray([float(np.asarray(obs[key]).item())-goal], dtype=np.float32)
    delta = float(np.asarray(obs['yaw']).item())-ref[5]
    result['yaw'] = np.asarray([np.arctan2(np.sin(delta), np.cos(delta))], dtype=np.float32)
    return result


class StationkeepingYawEnv(BlueRov):
    """Same plant and reward law; explicit fixed target and randomized acquisition."""
    def __init__(self, render_mode=None, env_config=None, dynamics_config=None):
        cfg = env_config or {}
        goal = cfg.get('stationkeeping', {})
        self.reference = np.zeros(12)
        self.reference[:3] = goal.get('target_position', [0., 0., -5.])
        self.reference[5] = np.deg2rad(float(goal.get('target_yaw_deg', 0.)))
        self.initial_offset = np.asarray(goal.get('initial_offset_m', [.3, .3, .15]), dtype=float)
        self.initial_yaw = np.deg2rad(float(goal.get('initial_yaw_error_deg', 30.)))
        self.radius = float(goal.get('success_radius_m', .2))
        self.yaw_tolerance = np.deg2rad(float(goal.get('success_yaw_deg', 5.)))
        if (self.initial_offset.shape != (3,) or not np.isfinite(self.initial_offset).all()
                or np.any(self.initial_offset < 0) or not np.isfinite(self.initial_yaw)
                or self.initial_yaw < 0 or self.radius <= 0 or self.yaw_tolerance <= 0
                or not np.isfinite(self.reference).all()):
            raise ValueError('Invalid stationkeeping target, initial offsets or tolerances.')
        super().__init__(render_mode, env_config, dynamics_config)

    def _get_obs(self):
        return stationkeeping_observation(super()._get_obs(), self.reference)

    def reset(self, *, seed=None, options=None):
        # Explicit episode RNG: reproducible training with fresh wave phases per reset.
        import gymnasium as gym
        gym.Env.reset(self, seed=seed)
        opts = dict(options or {})
        params = dict(opts.get('jonswap_params', self.jonswap_params or {}))
        if opts.get('randomize_wave_seed', True):
            params['seed'] = int(self.np_random.integers(0, 2**31))
        opts['jonswap_params'] = params
        _, info = super().reset(options=opts)
        position = self.reference[:3] + self.np_random.uniform(-self.initial_offset, self.initial_offset)
        for key, value in zip(('x', 'y', 'z'), position):
            self.state[key] = float(value)
        self.state['yaw'] = float(self.reference[5] + self.np_random.uniform(-self.initial_yaw, self.initial_yaw))
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        distance = float(np.linalg.norm([obs[k].item() for k in ('x', 'y', 'z')]))
        yaw = abs(float(obs['yaw'].item()))
        info.update({'metrics/position_error_euclidean': distance,
                     'metrics/yaw_error_rad': yaw,
                     'metrics/yaw_error_abs_deg': float(np.rad2deg(yaw)),
                     'metrics/stationkeeping_error': max(distance/self.radius, yaw/self.yaw_tolerance)})
        return obs, reward, terminated, truncated, info


def main(argv=None):
    from pathlib import Path
    import sys
    from ..cli import main as workflow_main
    args = list(sys.argv[1:] if argv is None else argv)
    if '--config' not in args:
        args = ['--config', str(Path(__file__).resolve().parents[3]/'configs/training/stationkeeping_waves_yaw.yaml'), *args]
    workflow_main(['train', *args])


if __name__ == '__main__':
    main()
