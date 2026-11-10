"""Episode contexts for historical-mechanics pretraining and a wave curriculum."""
from copy import deepcopy
import math

import numpy as np

from bluerov2_gym.envs.core.dynamics import Dynamics
from .trajectory_frozen_env import Frozen889TrajectoryEnv
from ..configuration import merge


HISTORICAL_DAMPING = [33.8, 54.26875, 73.37135, 40., 40., 40.]


def sample_context(config, index, rng, group=0, groups=1, anchor=False):
    stage = config['curriculum'][index]
    base = config['environment']['dynamics_config']
    sea = merge(base.get('jonswap', {}), stage.get('jonswap', {}))
    ranges = stage.get('ranges', {})
    for key, bounds in ranges.items():
        if (len(bounds) != 2 or not all(math.isfinite(v) for v in bounds)
                or bounds[0] > bounds[1]):
            raise ValueError(f'Invalid range {key}: {bounds}')
        lo, hi = bounds
        # Validation groups cover disjoint Hs bands, rather than averaging away
        # poor performance at the upper end of a level.
        if key == 'Hs' and groups > 1:
            width = (hi - lo) / groups
            lo, hi = lo + group * width, lo + (group + 1) * width
        value = float(rng.uniform(lo, hi))
        if key == 'wave_direction_deg':
            angle = math.radians(value)
            sea['wave_dir'] = [math.cos(angle), math.sin(angle)]
        elif key in sea and key not in {'seed', 'N', 'enable_wave_force'}:
            sea[key] = value
        else:
            raise ValueError(f'Unsupported curriculum range: {key}')
    # Training and validation use disjoint halves of the uint32 seed space.
    sea['seed'] = int(sea.get('seed', 42)) if anchor else int(rng.integers(0, 2**31))
    return dict(stage_index=index, name=stage['name'],
                mechanics_model=stage['mechanics_model'],
                disturbance_model=deepcopy(stage['disturbance_model']), jonswap=sea)


def context_dynamics_config(config, context):
    result = deepcopy(config['environment']['dynamics_config'])
    result['mechanics_model'] = context['mechanics_model']
    result['disturbance_model'] = deepcopy(context['disturbance_model'])
    if context['mechanics_model'] == 'std':
        result.setdefault('hydrodynamics', {})['quadratic_damping'] = HISTORICAL_DAMPING.copy()
    result['jonswap'] = deepcopy(context['jonswap'])
    return result


class PhysicsCurriculumEnv(Frozen889TrajectoryEnv):
    """Preserve 889 feedback/reward; replace all mechanics consistently at reset."""
    def __init__(self, config, context_seed=None):
        self.curriculum_config = config
        tracking = config['environment'].get('yaw_tracking', {})
        self.yaw_weight = float(tracking.get('weight', 0.))
        self.yaw_feedback = bool(tracking.get('relative_observation', False))
        if not np.isfinite(self.yaw_weight) or self.yaw_weight < 0:
            raise ValueError('yaw_tracking.weight must be finite and nonnegative.')
        if self.yaw_weight and not self.yaw_feedback:
            raise ValueError('Yaw penalty requires relative_observation=true.')
        self.stage_index = 0
        self.context_rng = np.random.default_rng(context_seed)
        self.stage_resets = 0
        super().__init__(config['environment'].get('config'),
                         config['environment']['dynamics_config'])
        self.current_context = None

    def set_stage(self, index):
        if not 0 <= index < len(self.curriculum_config['curriculum']):
            raise ValueError('Invalid curriculum phase.')
        # Defer the switch until reset; never change dynamics under an old state.
        self.stage_index = index
        self.stage_resets = 0
        self.current_context = None

    def reset(self, seed=None, options=None):
        forced = (options or {}).get('curriculum_context')
        if forced is not None:
            context = deepcopy(forced)
        else:
            index = self.stage_index
            fraction = self.curriculum_config.get('rehearsal', {}).get('probability', 0.)
            eligible = [i for i in range(index)
                        if self.curriculum_config['curriculum'][i]['mechanics_model'] == 'corrected'
                        and self.curriculum_config['curriculum'][i]['disturbance_model'] ==
                        self.curriculum_config['curriculum'][index]['disturbance_model']]
            # Rehearse simpler seas only within the corrected target mechanics.
            if eligible and self.context_rng.random() < fraction:
                index = int(self.context_rng.choice(eligible))
            anchor = self.stage_resets == 0 and not self.curriculum_config['curriculum'][index].get('ranges')
            context = sample_context(self.curriculum_config, index, self.context_rng, anchor=anchor)
            self.stage_resets += 1
        self.current_context = context
        self.jonswap_params = deepcopy(context['jonswap'])
        self.dynamics = Dynamics(dynamics_config=context_dynamics_config(self.curriculum_config, context))
        obs, info = super().reset(seed=seed, options=options)
        if self.yaw_feedback:
            from ..evaluation.env_utils import build_tracking_observation
            pos, att, vel = self.traj.get_state_at_time(self.current_t)
            obs = build_tracking_observation(obs, np.concatenate([pos, att, vel, np.zeros(3)]),
                                             'legacy_yaw_error')
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        target_yaw = float(self.traj.get_state_at_time(self.current_t)[1][2])
        delta = float(self.state['yaw']) - target_yaw
        error = float(np.arctan2(np.sin(delta), np.cos(delta)))
        penalty = self.yaw_weight * abs(error)
        reward -= penalty
        if self.yaw_feedback:
            obs['yaw'] = np.array([error], dtype=np.float32)
        info['metrics/yaw_error_abs_deg'] = float(np.degrees(abs(error)))
        info['metrics/yaw_penalty'] = penalty
        info['metrics/trajectory_reward_raw'] = reward
        info['curriculum/stage'] = self.stage_index + 1
        info['curriculum/context_stage'] = self.current_context['stage_index'] + 1
        return obs, reward, terminated, truncated, info
