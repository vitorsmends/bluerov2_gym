"""Warm-start provenance and polygon-only episodes for trajectory fine-tuning."""
import hashlib
import math
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from .curriculum_context import PhysicsCurriculumEnv
from ..configuration import read_yaml
from ..evaluation.trajectories import create_trajectory


def source_pair(config):
    """Use the configured source pair only; never guess a checkpoint by mtime."""
    settings = config['finetuning']
    run = Path(settings['source_run'])
    old = read_yaml(run / 'config.resolved.yaml')
    model = run / 'models' / (old['paths']['model_name'] + settings.get('checkpoint_suffix', '') + '.zip')
    name = Path(old['paths']['vecnormalize_name'])
    normalizer = run / 'models' / (name.stem + settings.get('checkpoint_suffix', '') + name.suffix)
    for file in [model, normalizer]:
        if not file.is_file():
            raise FileNotFoundError(f'Fine-tuning source artifact is missing: {file}')
    # Keeping the observation/reward/plant contract prevents accidental policy reuse.
    for key in ['environment', 'ppo', 'vecnormalize']:
        if old[key] != config[key]:
            raise ValueError(f'Fine-tuning must inherit source {key}; change trajectories/seas in finetuning/curriculum.')
    if not config['environment'].get('yaw_tracking', {}).get('relative_observation'):
        raise ValueError('Polygon fine-tuning requires the relative-yaw policy profile.')
    return dict(model=str(model), normalizer=str(normalizer), source_run=str(run),
                sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [model, normalizer, run / 'config.resolved.yaml']})


def validate_finetuning(config):
    settings = config['finetuning']
    trajectory = settings['trajectory']
    if trajectory['name'] not in {'square', 'star'}:
        raise ValueError('This fine-tuning routine trains and assesses only the selected polygon.')
    params = trajectory.get('params', {})
    period = float(params.get('period', 100. if trajectory['name']=='star' else 80.))
    segments = 10 if trajectory['name']=='star' else 4
    side = float(params.get('side', 1.6))
    if not all(math.isfinite(x) and x > 0 for x in [period, side, settings['episode_seconds']]):
        raise ValueError('Square period, side and episode_seconds must be finite and positive.')
    focus = settings.get('corner_reset', {})
    probability = float(focus.get('probability', 0.))
    lo, hi = focus.get('seconds_before_corner', [1., 5.])
    if not 0 <= probability <= 1 or not 0 < lo <= hi < period / segments:
        raise ValueError('Invalid corner-focused reset probability/window.')
    for stage in config['curriculum']:
        if stage['mechanics_model'] != 'corrected':
            raise ValueError('Fine-tuning never returns to preparation mechanics.')
        evaluation = dict(config['evaluation'], **stage.get('evaluation', {}))
        if evaluation['episodes'] % evaluation['groups']:
            raise ValueError('Assessment episodes must cover Hs groups equally.')
        if evaluation['duration_seconds'] - evaluation['settling_time_seconds'] < period:
            raise ValueError('Assessment after settling must cover a full polygon cycle.')
    source_pair(config)


class PolygonReference:
    """Same piecewise-linear reference and numerical derivatives as evaluation."""
    def __init__(self, name, params, start_phase):
        self.start_phase = start_phase
        self.phase_offset = None
        self.reference = create_trajectory(name, params)
        self.segment_seconds = self.reference.period / (len(self.reference.points)-1)

    def get_state_at_time(self, t):
        if self.phase_offset is None:
            self.phase_offset = self.start_phase - t
        ref = self.reference.get_reference(t + self.phase_offset)
        return ref[:3].astype(np.float32), ref[3:6].astype(np.float32), ref[6:9].astype(np.float32)

    def since_corner(self, t):
        return (t + self.phase_offset) % self.segment_seconds


class FineTuningEnv(PhysicsCurriculumEnv):
    """Retain the existing feedback/reward and change only reference/reset horizon."""
    def __init__(self, config, context_seed=None):
        self.polygon_name = config['finetuning']['trajectory']['name']
        self.polygon_settings = config['finetuning']['trajectory'].get('params', {})
        self.reset_focus = config['finetuning'].get('corner_reset', {})
        self.episode_steps = math.ceil(config['finetuning']['episode_seconds'] /
                                       config['environment']['config']['dt'])
        self.corner_window = float(config['evaluation'].get('corner_window_seconds', 3.))
        self.assessment = False
        self.steps_in_episode = 0
        super().__init__(config, context_seed)

    def reset(self, seed=None, options=None):
        options = options or {}
        self.assessment = bool(options.get('assessment', False))
        self.trajectory_name = self.polygon_name
        period = float(self.polygon_settings.get('period', 100. if self.polygon_name=='star' else 80.))
        focused = (not self.assessment and self.context_rng.random() < self.reset_focus.get('probability', 0.))
        if focused:
            segments = 10 if self.polygon_name=='star' else 4
            corner = int(self.context_rng.integers(segments)) * period / segments
            lo, hi = self.reset_focus.get('seconds_before_corner', [1., 5.])
            phase = (corner - self.context_rng.uniform(lo, hi)) % period
        else:
            phase = self.context_rng.uniform(0, period)
        self.traj = PolygonReference(self.polygon_name, self.polygon_settings, float(phase))
        self.steps_in_episode = 0
        obs, info = super().reset(seed=seed, options=options)
        self.current_context.update(trajectory=self.polygon_name, start_phase_s=float(phase), corner_focused_reset=bool(focused))
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        self.steps_in_episode += 1
        info['finetune/trajectory'] = self.trajectory_name
        info['finetune/corner_window'] = self.traj.since_corner(self.current_t) <= self.corner_window
        if not self.assessment and self.steps_in_episode >= self.episode_steps and not terminated:
            truncated = True  # Time limit, not a physical failure; SB3 bootstraps terminal observations.
        return obs, reward, terminated, truncated, info


class PolygonMetrics(BaseCallback):
    """Track corner recovery separately from aggregate square error."""
    def _on_rollout_start(self):
        self.errors, self.corner_errors = [], []

    def _on_step(self):
        for info in self.locals['infos']:
            self.errors.append(info['tracking_error_m'])
            if info['finetune/corner_window']:
                self.corner_errors.append(info['tracking_error_m'])
        return True

    def _on_rollout_end(self):
        if self.errors:
            self.logger.record('finetune/polygon_rmse_m', float(np.sqrt(np.mean(np.square(self.errors)))))
            self.logger.record('finetune/corner_step_fraction', len(self.corner_errors)/len(self.errors))
        if self.corner_errors:
            self.logger.record('finetune/corner_rmse_m', float(np.sqrt(np.mean(np.square(self.corner_errors)))))
