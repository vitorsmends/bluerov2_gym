"""Episode contexts for historical-mechanics pretraining and a wave curriculum."""
from copy import deepcopy
import math

import numpy as np

from bluerov2_gym.envs.core.dynamics import Dynamics
from .trajectory_frozen_env import Frozen889TrajectoryEnv, reference
from ..configuration import merge
from ..action_allocation import PolicyActionAllocation


HISTORICAL_DAMPING = [33.8, 54.26875, 73.37135, 40., 40., 40.]


class ConfiguredEightTrajectory(reference.TrajectoryGenerator):
    """Optional constant-depth variant; archived feedback/reset/reward stay intact."""
    def __init__(self, settings):
        super().__init__()
        self.z_target = float(settings.get('depth', self.z_target))
        self.depth_ramp = settings.get('depth_ramp', True)
        if not math.isfinite(self.z_target) or not isinstance(self.depth_ramp, bool):
            raise ValueError('Invalid trajectory_reference depth or depth_ramp.')

    def get_state_at_time(self, t):
        position, attitude, velocity = super().get_state_at_time(t)
        if not self.depth_ramp:
            position[2], velocity[2] = self.z_target, 0.0
        return position, attitude, velocity


def sample_context(config, index, rng, group=0, groups=1, anchor=False):
    stage = config['curriculum'][index]
    base = config['environment']['dynamics_config']
    sea = merge(base.get('jonswap', {}), stage.get('jonswap', {}))
    cases = stage.get('sea_cases')
    selected_case = None
    if cases is not None:
        if not isinstance(cases, list) or not cases or any('name' not in c or 'jonswap' not in c for c in cases):
            raise ValueError('sea_cases must contain named JONSWAP configurations.')
        # Assessment groups enumerate every case; training samples them uniformly.
        case_index = group % len(cases) if groups == len(cases) and groups > 1 else int(rng.integers(len(cases)))
        selected_case = cases[case_index]
        sea = merge(sea, selected_case['jonswap'])
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
                **({'sea_case': selected_case['name']} if selected_case else {}),
                **({'trajectory_reference': deepcopy(stage['trajectory_reference'])}
                   if 'trajectory_reference' in stage else {}),
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
        self.action_allocator = PolicyActionAllocation(
            self.dynamics, config['environment'].get('action_control'))
        if self.action_allocator.enabled:
            self.action_space = self.action_allocator.action_space
        self.current_context = None
        trajectory = config['environment'].get('trajectory_reference')
        if trajectory is not None:
            self.traj = ConfiguredEightTrajectory(trajectory)

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
        trajectory = context.get('trajectory_reference',
                                 self.curriculum_config['environment'].get('trajectory_reference'))
        if trajectory is not None:
            # Select before the archived reset computes the reference and initial state.
            self.traj = ConfiguredEightTrajectory(trajectory)
        self.jonswap_params = deepcopy(context['jonswap'])
        self.dynamics = Dynamics(dynamics_config=context_dynamics_config(self.curriculum_config, context))
        self.action_allocator = PolicyActionAllocation(
            self.dynamics, self.curriculum_config['environment'].get('action_control'))
        obs, info = super().reset(seed=seed, options=options)
        if self.yaw_feedback:
            from ..evaluation.env_utils import build_tracking_observation
            pos, att, vel = self.traj.get_state_at_time(self.current_t)
            obs = build_tracking_observation(obs, np.concatenate([pos, att, vel, np.zeros(3)]),
                                             'legacy_yaw_error')
        return obs, info

    def step(self, action):
        action, allocation_info = self.action_allocator.convert(action)
        obs, reward, terminated, truncated, info = super().step(action)
        info.update(allocation_info)
        info['metrics/thruster_saturation_fraction'] = float(np.mean(
            (np.asarray(action) <= self.dynamics.thruster_min + 1e-3) |
            (np.asarray(action) >= self.dynamics.thruster_max - 1e-3)))
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
