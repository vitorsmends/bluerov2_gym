"""Episode-wise trajectory coverage and progressive sea states on the 889 recipe."""
from pathlib import Path
import math

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .trajectory_env import TrajectoryTrackingEnv
from .trajectory import TensorboardCallback
from .curriculum import TrainingProgressCallback
from .metrics import PositionErrorCallback
from .device import resolve_training_device, report_training_device
from ..evaluation.trajectories import create_trajectory
from ..configuration import merge
from ..artifacts import write_yaml


class TrajectoryStates:
    def __init__(self, specification):
        self.reference = create_trajectory(specification['name'], specification.get('params'))

    def get_state_at_time(self, t):
        ref = np.asarray(self.reference.get_reference(t), dtype=np.float32)
        return ref[:3], ref[3:6], ref[6:9]


class TrajectoryEpisodeEnv(TrajectoryTrackingEnv):
    """Keep reference feedback/reward/actions; select a path only at reset."""
    def __init__(self, config):
        environment = config['environment']
        super().__init__(environment.get('config'), environment['dynamics_config'])
        self.stages = config['curriculum']
        self.sea_defaults = merge(self.dynamics.jonswap_params, config.get('jonswap_defaults', {}))
        self.episodes_per_trajectory = int(config['trajectory_selection']['episodes_per_trajectory'])
        if self.episodes_per_trajectory < 1:
            raise ValueError('episodes_per_trajectory must be positive.')
        self.stage_ids = []
        self.stage_references = []
        for stage in self.stages:
            ids = [item.get('id', item['name']) for item in stage['trajectories']]
            if (not ids or any(not isinstance(key, str) or not key.strip() for key in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError('Each stage needs a nonempty set of unique trajectory ids.')
            self.stage_ids.append(ids)
            self.stage_references.append([TrajectoryStates(item) for item in stage['trajectories']])
        self.set_stage(0)

    def set_stage(self, index):
        stage = self.stages[index]
        self.stage_index = index
        self.episode_index = 0
        self.trajectory_ids = self.stage_ids[index]
        self.references = self.stage_references[index]
        self.jonswap_params = merge(self.sea_defaults, stage.get('jonswap', {}))
        self.dynamics.set_jonswap_params(**self.jonswap_params)

    def reset(self, seed=None, options=None):
        selected = (self.episode_index // self.episodes_per_trajectory) % len(self.references)
        self.episode_index += 1
        self.trajectory_id = self.trajectory_ids[selected]
        self.traj = self.references[selected]
        return super().reset(seed=seed, options=options)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['curriculum/trajectory'] = self.trajectory_id
        info['curriculum/stage'] = self.stage_index + 1
        return obs, reward, terminated, truncated, info


class TrajectoryCoverageCallback(BaseCallback):
    """Require sustained episode performance separately for every trajectory."""
    def __init__(self, trajectory_ids, settings, dt, stage_index, report):
        super().__init__()
        self.streaks = dict.fromkeys(trajectory_ids, 0)
        self.required = int(settings['consecutive_successes_per_trajectory'])
        self.tolerance = float(settings['position_tolerance_m'])
        self.min_fraction = float(settings['min_tracking_fraction'])
        self.max_rmse = float(settings['max_rmse_m'])
        settling = float(settings['settling_time_seconds'])
        self.settling_steps = math.ceil(settling / dt)
        if (self.required < 1 or not math.isfinite(settling) or settling < 0
                or not 0 < self.min_fraction <= 1 or not math.isfinite(self.tolerance)
                or self.tolerance <= 0 or not math.isfinite(self.max_rmse) or self.max_rmse <= 0):
            raise ValueError('Invalid trajectory curriculum success criterion.')
        self.stage_index = stage_index
        self.report = report
        self.mastered = False
        self.episodes = 0
        self.current_trajectory = None
        self.last_result = None

    def _on_training_start(self):
        n = self.training_env.num_envs
        self.steps = [0] * n
        self.samples = [0] * n
        self.squared_error = [0.] * n
        self.within = [0] * n
        self.valid = [True] * n

    def _on_step(self):
        for index, (info, done) in enumerate(zip(self.locals['infos'], self.locals['dones'])):
            case = info['curriculum/trajectory']
            if case not in self.streaks or info['curriculum/stage'] != self.stage_index:
                raise RuntimeError('Unexpected trajectory or sea stage in curriculum feedback.')
            if case != self.current_trajectory:
                self.current_trajectory = case
                self.report(self)
            self.steps[index] += 1
            if self.steps[index] > self.settling_steps:
                error = info.get('metrics/position_error_euclidean')
                if error is None or not math.isfinite(error):
                    self.valid[index] = False
                else:
                    self.samples[index] += 1
                    self.squared_error[index] += error * error
                    self.within[index] += int(error < self.tolerance)
            if not done:
                continue
            count = self.samples[index]
            rmse = math.sqrt(self.squared_error[index] / count) if count else None
            fraction = self.within[index] / count if count else 0.
            success = (bool(info.get('TimeLimit.truncated', False)) and self.valid[index]
                       and count > 0 and fraction >= self.min_fraction and rmse <= self.max_rmse)
            self.streaks[case] = self.streaks[case] + 1 if success else 0
            self.episodes += 1
            self.last_result = dict(trajectory=case, success=success, rmse_m=rmse,
                                    tracking_fraction=fraction)
            self.mastered = all(value >= self.required for value in self.streaks.values())
            self.logger.record(f'curriculum/{case}/success_streak', self.streaks[case])
            self.logger.record('curriculum/trajectories_mastered',
                               sum(value >= self.required for value in self.streaks.values()))
            self.logger.record('curriculum/mastered', float(self.mastered))
            self.report(self)
            self.steps[index] = self.samples[index] = self.within[index] = 0
            self.squared_error[index] = 0.
            self.valid[index] = True
            if self.mastered:
                return False
        return True


def train_trajectory_curriculum(config):
    if not config.get('curriculum'):
        raise ValueError('The trajectory curriculum must have at least one stage.')
    paths = config['paths']
    models = Path(paths['model_dir'])
    models.mkdir(parents=True, exist_ok=True)
    steps_per_episode = int(config['environment']['max_episode_steps'])
    dt = float(config['environment']['config']['dt'])
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive.')
    criterion = config['curriculum_transition']
    if criterion.get('mode', 'success') != 'success':
        raise ValueError('Trajectory curriculum advances by success, not timestep limits.')
    if steps_per_episode <= math.ceil(float(criterion['settling_time_seconds']) / dt):
        raise ValueError('Episode duration must exceed the settling time.')

    def make_env():
        return Monitor(TimeLimit(TrajectoryEpisodeEnv(config), steps_per_episode),
                       info_keywords=('is_success', 'tracking_error_m'))

    env = VecNormalize(DummyVecEnv([make_env]), **config['vecnormalize'])
    model = None
    history = []
    status = {}
    status_path = models / 'curriculum_status.yaml'

    def save_status():
        temporary = status_path.with_suffix('.tmp')
        write_yaml(temporary, dict(status, history=history))
        temporary.replace(status_path)

    def save_pair(suffix):
        model.save(str(models / f"{paths['model_name']}{suffix}"))
        env.save(str(models / f"{Path(paths['vecnormalize_name']).stem}{suffix}.pkl"))

    try:
        params = dict(config['ppo'])
        policy = params.pop('policy')
        requested = config['training'].get('device', params.pop('device', 'auto'))
        device = resolve_training_device(requested)
        report_training_device(requested, device)
        model = PPO(policy, env, seed=config['training'].get('seed'),
                    device=device, tensorboard_log=paths['tensorboard_log'], **params)
        metadata = dict(config=config, task='trajectory_curriculum', execution_device=device,
                        observation_adapter='legacy', policy_kwargs=model.policy_kwargs,
                        policy_architecture=str(model.policy), effective_episode_limit=steps_per_episode)
        metadata_path = models / f"{paths['model_name']}_config.yaml"
        write_yaml(metadata_path, metadata)
        checkpoint = config['checkpoint']
        checkpoint_callback = CheckpointCallback(
            save_freq=int(checkpoint['save_freq']), save_path=str(models),
            name_prefix=checkpoint['name_prefix'], save_vecnormalize=True)
        metrics = PositionErrorCallback()
        raw_reward = TensorboardCallback()
        base_env = env.venv.envs[0].unwrapped
        for index, stage in enumerate(config['curriculum']):
            if index:
                base_env.set_stage(index)
            status = dict(index=index+1, name=stage['name'], mode='trajectory_success',
                          status='training', start_step=model.num_timesteps, step=model.num_timesteps,
                          trajectory_successes=dict.fromkeys(base_env.trajectory_ids, 0),
                          required_episodes=int(criterion['consecutive_successes_per_trajectory']))
            save_status()

            def report(value):
                status.update(step=model.num_timesteps, current_trajectory=value.current_trajectory,
                              trajectory_successes=dict(value.streaks),
                              completed_episodes=value.episodes, last_episode=value.last_result)
                save_status()

            coverage = TrajectoryCoverageCallback(base_env.trajectory_ids, criterion, dt, index+1, report)
            callbacks = CallbackList([checkpoint_callback, raw_reward, metrics,
                                      TrainingProgressCallback(None, None, stage['name'], index+1,
                                                               len(config['curriculum'])), coverage])
            print(f"[TRAJECTORY CURRICULUM] {index+1}/{len(config['curriculum'])}: {stage['name']} | "
                  f"trajectories={base_env.trajectory_ids} | no phase step limit", flush=True)
            # Ask SB3 to reset its observation/episode-start caches using the
            # newly configured env. Preserve weights, optimizer and VecNormalize.
            model._last_obs = None
            model.learn(total_timesteps=float('inf'), callback=callbacks,
                        tb_log_name=config['training']['tb_log_name'], reset_num_timesteps=index == 0)
            if not coverage.mastered:
                raise RuntimeError('Trajectory phase stopped without achieving coverage.')
            status.update(status='mastered', step=model.num_timesteps, end_step=model.num_timesteps)
            history.append(dict(status))
            save_status()
            save_pair(f'_stage_{index+1}')
        save_pair('')
        write_yaml(metadata_path, dict(metadata, actual_timesteps=model.num_timesteps))
    except KeyboardInterrupt:
        if model is not None:
            save_pair('_interrupted')
            status.update(status='interrupted', step=model.num_timesteps)
            save_status()
        raise
    finally:
        env.close()
