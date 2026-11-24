"""Stationkeeping task using the article's shared physics/wave curriculum engine."""
from copy import deepcopy
import math
import numpy as np
from bluerov2_gym.envs.core.dynamics import Dynamics
from .curriculum_context import sample_context, context_dynamics_config
from .stationkeeping_yaw import StationkeepingYawEnv


class StationkeepingPhysicsEnv(StationkeepingYawEnv):
    def __init__(self, config, context_seed=None):
        self.curriculum_config = config
        self.context_rng = np.random.default_rng(context_seed)
        self.stage_index = 0
        self.current_context = None
        self.yaw_feedback = True
        settings = config['environment'].get('config', {})
        self.episode_steps = math.ceil(settings.get('stationkeeping', {}).get(
            'training_episode_seconds', 100.) / settings.get('dt', .1))
        self.assessment = False
        self.elapsed_steps = 0
        super().__init__(env_config=settings, dynamics_config=config['environment']['dynamics_config'])

    def set_stage(self, index):
        if not 0 <= index < len(self.curriculum_config['curriculum']):
            raise ValueError('Invalid stationkeeping phase.')
        self.stage_index = index
        self.current_context = None

    def reset(self, *, seed=None, options=None):
        opts = dict(options or {})
        context = deepcopy(opts.get('curriculum_context'))
        if context is None:
            context = sample_context(self.curriculum_config, self.stage_index, self.context_rng)
        self.current_context = context
        self.jonswap_params = deepcopy(context['jonswap'])
        self.dynamics = Dynamics(dynamics_config=context_dynamics_config(self.curriculum_config, context))
        # The shared evaluator assigns upper-half seeds; preserve them exactly.
        opts.update(jonswap_params=self.jonswap_params, randomize_wave_seed=False)
        self.assessment = bool(opts.get('assessment', False))
        self.elapsed_steps = 0
        return super().reset(seed=seed, options=opts)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.elapsed_steps += 1
        if not self.assessment and self.elapsed_steps >= self.episode_steps and not terminated:
            truncated = True
        error = info['metrics/position_error_euclidean']
        info.update(tracking_error_m=error,
                    velocity_error=float(np.linalg.norm([self.state[k] for k in ('u','v','w')])),
                    is_success=bool(error <= self.radius),
                    **{'metrics/trajectory_reward_raw':reward,
                       'metrics/thruster_saturation_fraction':float(np.mean(np.abs(action) >= 39.999)),
                       'curriculum/stage':self.stage_index+1,
                       'curriculum/context_stage':self.current_context['stage_index']+1})
        return obs, reward, terminated, truncated, info
