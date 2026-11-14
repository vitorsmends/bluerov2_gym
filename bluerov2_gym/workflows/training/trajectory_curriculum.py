"""Physics/wave curriculum with independent mastery gates on the 889 PPO recipe."""
from contextlib import contextmanager
from copy import copy, deepcopy
import math
from pathlib import Path
import pickle
import random
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
from tqdm.auto import tqdm

from .curriculum_context import PhysicsCurriculumEnv, sample_context
from .curriculum_resume import load_resume, restore_rng
from .progress_plateau import update_progress
from .device import resolve_training_device, report_training_device
from .mastery import CurriculumEarlyStopped
from .metrics import PositionErrorCallback
from .trajectory import TensorboardCallback
from ..artifacts import write_yaml


def phase_settings(config, index):
    stage = config['curriculum'][index]
    evaluation = dict(config['evaluation'], **stage.get('evaluation', {}))
    settings = dict(config['curriculum_transition'], **stage.get('transition', {}))
    if stage.get('sea_cases'):
        evaluation['groups'] = len(stage['sea_cases'])
    elif 'Hs' not in stage.get('ranges', {}):
        evaluation['groups'] = 1  # No artificial groups for identical sea conditions.
    return evaluation, settings


@contextmanager
def isolated_random_state():
    """Validation must not change future exploration or historical random resets."""
    numpy_state, python_state = np.random.get_state(), random.getstate()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        np.random.set_state(numpy_state)
        random.setstate(python_state)


def validate_config(config):
    if config['training'].get('task') == 'trajectory_finetune':
        from .finetune_polygon import validate_finetuning
        validate_finetuning(config)
    stages, settings = config.get('curriculum', []), config['curriculum_transition']
    evaluation = config['evaluation']
    if not stages or settings.get('mode') != 'success':
        raise ValueError('A nonempty curriculum with success transitions is required.')
    if config['environment'].get('max_episode_steps') is not None:
        raise ValueError('The 889 training recipe uses no episode TimeLimit; evaluation has its own horizon.')
    if not 0 <= config.get('rehearsal', {}).get('probability', 0.) < 1:
        raise ValueError('Invalid rehearsal probability.')
    for key in ['every_timesteps', 'episodes', 'groups', 'consecutive_batches']:
        value = evaluation[key] if key in evaluation else settings[key]
        if not isinstance(value, int) or value < 1:
            raise ValueError(f'{key} must be a positive integer.')
    if evaluation['episodes'] % evaluation['groups']:
        raise ValueError('Evaluation episodes must be divisible by groups.')
    if not 0 < settings['min_batch_success_rate'] <= 1 or not 0 < settings['min_group_success_rate'] <= 1:
        raise ValueError('Invalid batch/group success fraction.')
    if not 0 < settings['min_tracking_fraction'] <= 1:
        raise ValueError('Invalid tracking fraction.')
    for key in ['position_tolerance_m', 'max_rmse_m']:
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'Invalid {key}.')
    dt = config['environment']['config']['dt']
    if not math.isclose(dt, .1) or not math.isclose(config['environment']['dynamics_config']['dt'], .1):
        raise ValueError('The reference recipe requires dt=0.1.')
    if not 0 <= evaluation['settling_time_seconds'] < evaluation['duration_seconds']:
        raise ValueError('Evaluation must exceed the settling interval.')
    if config['training']['phase_planned_timesteps'] <= 0:
        raise ValueError('The planned phase length must be positive.')
    plateau = config.get('plateau_early_stopping', {})
    if plateau.get('monitor', 'rmse') not in {'rmse', 'mastery_progress'}:
        raise ValueError('Unknown plateau monitor.')
    if not 0 <= plateau.get('min_success_delta', .025) <= 1:
        raise ValueError('Invalid plateau success delta.')
    if plateau.get('enabled') and (plateau['patience_evaluations'] < 1 or
                                  plateau['warmup_evaluations'] < 0 or plateau['min_delta_m'] < 0):
        raise ValueError('Invalid optional plateau settings.')
    for i, stage in enumerate(stages):
        phase_eval, phase_gate = phase_settings(config, i)
        window = phase_gate.get('rolling_batches', 1)
        if not isinstance(window, int) or isinstance(window, bool) or window < 1:
            raise ValueError('rolling_batches must be a positive integer.')
        if (phase_eval['episodes'] < 1 or phase_eval['groups'] < 1 or
                phase_eval['episodes'] % phase_eval['groups'] or phase_eval['every_timesteps'] < 1 or
                not 0 <= phase_eval['settling_time_seconds'] < phase_eval['duration_seconds'] or
                phase_gate['consecutive_batches'] < 1):
            raise ValueError(f'Invalid evaluation overrides in phase {i+1}.')
        if stage['mechanics_model'] not in {'std', 'corrected'}:
            raise ValueError('Unknown mechanics model.')
        context = sample_context(config, i, np.random.default_rng(0))
        if context['disturbance_model'].get('normalize_spectrum') is not True and stage.get('ranges', {}).get('Hs'):
            raise ValueError('A curriculum over Hs requires normalize_spectrum=true.')


def episode_result(errors, terminated, truncated, horizon, settling_steps, settings, group):
    values = np.asarray(errors, dtype=float)
    valid = len(values) > settling_steps and bool(np.all(np.isfinite(values)))
    tail = values[settling_steps:]
    rmse = float(np.sqrt(np.mean(tail**2))) if valid else None
    fraction = float(np.mean(tail <= settings['position_tolerance_m'])) if valid else 0.
    completed = len(values) == horizon and not terminated and not truncated
    success = (completed and valid and rmse <= settings['max_rmse_m']
               and fraction >= settings['min_tracking_fraction'])
    return dict(group=group, steps=len(values), completed=completed, success=bool(success),
                terminated=bool(terminated), truncated=bool(truncated), rmse_m=rmse,
                rmse_full_m=float(np.sqrt(np.mean(values**2))) if valid else None,
                tracking_fraction=fraction)


def batch_result(episodes, settings):
    groups = sorted({e['group'] for e in episodes})
    rates = {str(g): float(np.mean([e['success'] for e in episodes if e['group'] == g])) for g in groups}
    rate = float(np.mean([e['success'] for e in episodes]))
    rmse = [e['rmse_m'] for e in episodes if e['rmse_m'] is not None]
    return dict(success_rate=rate, group_success_rates=rates,
                mean_rmse_m=float(np.mean(rmse)) if rmse else None,
                passed=bool(rate >= settings['min_batch_success_rate'] and
                            all(r >= settings['min_group_success_rate'] for r in rates.values())),
                episodes=episodes)


def curriculum_env(config, seed):
    if config['training'].get('task') == 'trajectory_finetune':
        from .finetune_polygon import FineTuningEnv
        return FineTuningEnv(config, seed)
    return PhysicsCurriculumEnv(config, seed)


def rolling_mastery(batches, settings):
    """Aggregate recent independent episodes without relaxing episode tolerances."""
    count = settings.get('rolling_batches', 1)
    if len(batches) < count:
        return None
    return batch_result([episode for batch in batches[-count:] for episode in batch], settings)


def evaluate_stage(model, normalizer, config, index, batch_index):
    """Fresh validation contexts; never step/update the training VecNormalize."""
    evaluation, settings = phase_settings(config, index)
    dt = config['environment']['config']['dt']
    horizon = math.ceil(evaluation['duration_seconds'] / dt)
    settling = math.ceil(evaluation['settling_time_seconds'] / dt)
    frozen = copy(normalizer)
    frozen.obs_rms = deepcopy(normalizer.obs_rms)
    frozen.ret_rms = deepcopy(normalizer.ret_rms)
    frozen.training, frozen.norm_reward = False, False
    episodes = []
    was_training = model.policy.training
    # A SeedSequence avoids ad-hoc overlap between phase/batch streams.
    rng = np.random.default_rng(np.random.SeedSequence([evaluation['seed'], index, batch_index]))
    with isolated_random_state():
        env = curriculum_env(config, 0)
        env.set_stage(index)
        try:
            for episode in range(evaluation['episodes']):
                group = episode % evaluation['groups']
                context = sample_context(config, index, rng, group, evaluation['groups'])
                wave_seed = int(rng.integers(2**31, 2**32))
                context['jonswap']['seed'] = wave_seed
                reset_seed = int(rng.integers(2**31, 2**32))
                np.random.seed(reset_seed)
                options = {'curriculum_context': context}
                if config['training'].get('task') == 'trajectory_finetune':
                    options['assessment'] = True
                obs, _ = env.reset(seed=reset_seed, options=options)
                errors, rewards, saturated = [], [], []
                yaw_errors, corner_errors = [], []
                terminated = truncated = False
                for _ in range(horizon):
                    action, _ = model.predict(frozen.normalize_obs(obs), deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    errors.append(info['tracking_error_m'])
                    if len(errors) > settling:
                        yaw_errors.append(info.get('metrics/yaw_error_abs_deg', float('nan')))
                        if info.get('finetune/corner_window', False):
                            corner_errors.append(errors[-1])
                    rewards.append(reward)
                    saturated.append(info.get('metrics/thruster_saturation_fraction',
                                              float(np.mean(np.abs(action) >= 39.999))))
                    if terminated or truncated or not math.isfinite(errors[-1]):
                        break
                result = episode_result(errors, terminated, truncated, horizon, settling, settings, group)
                result.update(context=context, reset_seed=reset_seed,
                              mean_reward_raw=float(np.mean(rewards))
                              if np.all(np.isfinite(rewards)) else None,
                              thruster_saturation_fraction=float(np.mean(saturated)))
                if config['training'].get('task') == 'trajectory_finetune':
                    yaw_mae = float(np.mean(yaw_errors)) if yaw_errors else float('inf')
                    corner_rmse = float(np.sqrt(np.mean(np.square(corner_errors)))) if corner_errors else None
                    corner_ok = (env.trajectory_name not in {'square', 'star'} or
                                 (corner_rmse is not None and corner_rmse <= settings['max_corner_rmse_m']))
                    result.update(trajectory=env.trajectory_name, sea_group=group,
                                  group=f'{group}:{env.trajectory_name}', yaw_mae_deg=yaw_mae,
                                  corner_rmse_m=corner_rmse)
                    result['success'] = bool(result['success'] and np.isfinite(yaw_mae) and
                                             yaw_mae <= settings['max_yaw_mae_deg'] and corner_ok)
                episodes.append(result)
        finally:
            env.close()
            model.policy.set_training_mode(was_training)
    return batch_result(episodes, settings)


class PhaseProgress(BaseCallback):
    def __init__(self, planned, start, name, report):
        super().__init__()
        self.start = start
        self.report = report
        self.bar = tqdm(total=planned, desc=name, unit='steps', dynamic_ncols=True)

    def _on_step(self):
        self.bar.update(self.model.num_timesteps - self.start - self.bar.n)
        return True

    def _on_rollout_end(self):
        self.report()


class ContinuousMetrics(PositionErrorCallback):
    """Keep episode sums when learn() returns between validation batches."""
    def _on_training_start(self):
        if not hasattr(self, 'episode_sum'):
            super()._on_training_start()

    def reset_phase(self):
        if hasattr(self, 'episode_sum'):
            self.episode_sum.fill(0.)
            self.episode_count.fill(0)


def train_trajectory_curriculum(config):
    validate_config(config)
    resume = load_resume(config)
    warm_start = None
    if config.get('finetuning') and resume is None:
        from .finetune_polygon import source_pair
        warm_start = source_pair(config)
    paths, settings = config['paths'], config['curriculum_transition']
    models = Path(paths['model_dir'])
    models.mkdir(parents=True, exist_ok=True)
    logs = models.parent / 'logs' / 'curriculum_evaluations'
    logs.mkdir(parents=True, exist_ok=True)
    status_path = models / 'curriculum_status.yaml'
    seed = config['training'].get('seed')
    raw_env = DummyVecEnv([lambda: Monitor(curriculum_env(config, seed),
                          info_keywords=('is_success', 'tracking_error_m'))])
    initial = resume or warm_start
    env = (VecNormalize.load(initial['normalizer'], raw_env) if initial else
           VecNormalize(raw_env, **config['vecnormalize']))
    env.training, env.norm_reward = True, config['vecnormalize'].get('norm_reward', True)
    base = env.venv.envs[0].unwrapped
    model, bar = None, None
    history, status = (deepcopy(resume['history']) if resume else []), {}
    metadata_path = models / f"{paths['model_name']}_config.yaml"

    def report():
        status['updated_at'] = time.time()
        if model is not None:
            status.update(step=model.num_timesteps,
                          phase_steps=model.num_timesteps - status.get('start_step', 0),
                          effective_context=deepcopy(base.current_context))
        temporary = status_path.with_suffix('.tmp')
        write_yaml(temporary, dict(status, history=history))
        temporary.replace(status_path)

    def save_pair(suffix):
        stem = f"{paths['model_name']}{suffix}"
        norm_stem = f"{Path(paths['vecnormalize_name']).stem}{suffix}"
        model.save(str(models / stem))
        env.save(str(models / f'{norm_stem}.pkl'))
        # Explicit state snapshot for forensic reproduction; automatic resume is
        # deliberately not implied by loading the PPO alone.
        state = dict(status=deepcopy(status), history=deepcopy(history),
                     numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                     context_rng=deepcopy(base.context_rng.bit_generator.state))
        with (models / f'{stem}_curriculum_state.pkl').open('wb') as stream:
            pickle.dump(state, stream)
        return dict(model_path=str(models / f'{stem}.zip'),
                    vecnormalize_path=str(models / f'{norm_stem}.pkl'))

    try:
        params = dict(config['ppo'])
        policy = params.pop('policy')
        requested = config['training'].get('device', 'auto')
        device = resolve_training_device(requested)
        report_training_device(requested, device)
        if resume:
            model = PPO.load(resume['model'], env=env, device=device, force_reset=True)
            model.tensorboard_log = paths['tensorboard_log']
            restore_rng(resume['state'], base)
            print(f"Resuming {resume['model']} at {model.num_timesteps} steps; fresh episode, same optimizer/normalizer.", flush=True)
        elif warm_start:
            model = PPO.load(warm_start['model'], env=env, device=device, force_reset=True, seed=seed)
            model.tensorboard_log = paths['tensorboard_log']
            pretrained_timesteps = model.num_timesteps
            model.num_timesteps = 0  # Fine-tuning logs count additional interactions only.
            print(f'Fine-tuning {warm_start["model"]}; optimizer and normalization retained.', flush=True)
        else:
            model = PPO(policy, env, seed=seed, device=device,
                        tensorboard_log=paths['tensorboard_log'], **params)
        metadata = dict(config=config, task=config['training']['task'], execution_device=device,
                        observation_adapter=('legacy_yaw_error' if base.yaw_feedback else 'legacy'), policy_kwargs=model.policy_kwargs,
                        policy_architecture=str(model.policy),
                        effective_episode_limit=getattr(base, 'episode_steps', None))
        if warm_start:
            metadata.update(initialization=warm_start, pretrained_timesteps=pretrained_timesteps)
            write_yaml(models / 'initialization.yaml', dict(warm_start, pretrained_timesteps=pretrained_timesteps))
        write_yaml(metadata_path, metadata)
        checkpoint = config['checkpoint']
        metrics = ContinuousMetrics()
        callbacks = [CheckpointCallback(save_freq=checkpoint['save_freq'], save_path=str(models),
                     name_prefix=checkpoint['name_prefix'], save_vecnormalize=True),
                     TensorboardCallback(), metrics]
        if config['training'].get('task') == 'trajectory_finetune':
            from .finetune_polygon import PolygonMetrics
            callbacks.append(PolygonMetrics())
        if warm_start and config['finetuning'].get('assess_source', True):
            for source_stage in range(len(config['curriculum'])):
                print(f'[FINETUNE] Assessing source policy, phase {source_stage+1}.', flush=True)
                source_result = evaluate_stage(model, env, config, source_stage, 0)
                write_yaml(logs / f'source_baseline_stage_{source_stage+1}.yaml', source_result)
        rollout = int(config['ppo']['n_steps'])
        first_learn = resume is None
        for index, stage in enumerate(config['curriculum']):
            if resume and index < resume['index']:
                continue
            evaluation, settings = phase_settings(config, index)
            chunk = math.ceil(evaluation['every_timesteps'] / rollout) * rollout
            base.set_stage(index)
            metrics.reset_phase()
            # Reset before collecting a rollout in a different mechanical model.
            model._last_obs = None
            status = dict(index=index+1, name=stage['name'], mode='physics_wave_success',
                          status='training', start_step=model.num_timesteps, step=model.num_timesteps,
                          phase_steps=0, planned_timesteps=config['training']['phase_planned_timesteps'],
                          started_at=time.time(),
                          successful_batches=0, required_batches=settings['consecutive_batches'],
                          evaluations=0, validation_timesteps=0, mechanics_model=stage['mechanics_model'],
                          disturbance_model=stage['disturbance_model'])
            if resume and index == resume['index'] and resume['status']['index'] == index+1:
                prior = resume['status']
                for key in ['start_step', 'started_at', 'evaluations', 'validation_timesteps']:
                    if key in prior:
                        status[key] = prior[key]
                status['resumed_from'] = resume['model']
                status['successful_batches'] = 0  # Recheck mastery under the new criteria.
            report()
            best_rmse, stale = None, 0
            progress = {}
            recent_batches = []  # Reassess a full window after any resume.
            bar = PhaseProgress(status['planned_timesteps'], status['start_step'], stage['name'], report)
            while status['successful_batches'] < settings['consecutive_batches']:
                # Each call returns after complete rollout optimization. Nothing
                # is discarded to advance a phase, and no old-physics buffer is reused.
                model.learn(total_timesteps=chunk, callback=CallbackList(callbacks + [bar]),
                            reset_num_timesteps=first_learn, tb_log_name=config['training']['tb_log_name'])
                first_learn = False
                result = evaluate_stage(model, env, config, index, status['evaluations'])
                window_size = settings.get('rolling_batches', 1)
                if window_size > 1:
                    recent_batches.append(result['episodes'])
                    recent_batches = recent_batches[-window_size:]
                    mastery = rolling_mastery(recent_batches, settings)
                    result['current_batch_passed'] = result['passed']
                    result['mastery_window'] = dict(
                        batches=len(recent_batches), required_batches=window_size,
                        **({k: v for k, v in mastery.items() if k != 'episodes'}
                           if mastery else {'passed': False}))
                    result['passed'] = bool(mastery and mastery['passed'])
                status['evaluations'] += 1
                status['validation_timesteps'] += sum(e['steps'] for e in result['episodes'])
                status['successful_batches'] = status['successful_batches'] + 1 if result['passed'] else 0
                status['last_evaluation'] = {k: v for k, v in result.items() if k != 'episodes'}
                rmse = result['mean_rmse_m']
                plateau = config.get('plateau_early_stopping', {})
                if plateau.get('monitor') == 'mastery_progress':
                    progress = update_progress(progress, result, plateau)
                    stale = progress.get('stale', 0)
                    status['progress_monitor'] = deepcopy(progress)
                else:
                    improved = rmse is not None and (best_rmse is None or best_rmse - rmse > plateau.get('min_delta_m', .01))
                    if improved:
                        best_rmse, stale = rmse, 0
                    elif status['evaluations'] > plateau.get('warmup_evaluations', 5):
                        stale += 1
                status['stale_evaluations'] = stale
                report()
                suffix = f'_stage_{index+1}_eval_{model.num_timesteps}'
                pair = save_pair(suffix)
                write_yaml(logs / f'stage_{index+1}_step_{model.num_timesteps}.yaml',
                           dict(step=model.num_timesteps, phase=index+1, checkpoint=pair, **result))
                model.logger.record('curriculum/evaluation_success_rate', result['success_rate'])
                model.logger.record('curriculum/successful_batches', status['successful_batches'])
                if rmse is not None:
                    model.logger.record('curriculum/evaluation_rmse_m', rmse)
                model.logger.dump(model.num_timesteps)
                print(f"[CURRICULUM] {index+1}/{len(config['curriculum'])} {stage['name']} | "
                      f"phase_steps={status['phase_steps']} (planned={status['planned_timesteps']}, extensible) | "
                      f"validation={result['success_rate']:.0%} | "
                      f"batches={status['successful_batches']}/{settings['consecutive_batches']}", flush=True)
                if (plateau.get('enabled') and stale >= plateau['patience_evaluations'] and
                        status['successful_batches'] < settings['consecutive_batches']):
                    status['status'] = 'early_stopped'
                    status['stop_reason'] = ('Validation stagnation: '
                                             f'{stale} assessments without meaningful progress '
                                             f'({plateau.get("monitor", "rmse")}).')
                    report()
                    save_pair('_incomplete')
                    raise CurriculumEarlyStopped('Independent validation stagnated; curriculum is incomplete.')
            bar.bar.close()
            bar = None
            status.update(status='mastered', end_step=model.num_timesteps)
            report()
            history.append(deepcopy(status))
            report()
            save_pair(f'_stage_{index+1}')
        status['status'] = 'completed'
        report()
        save_pair('')
        write_yaml(metadata_path, dict(metadata, actual_timesteps=model.num_timesteps,
                                      validation_timesteps=sum(s['validation_timesteps'] for s in history)))
    except KeyboardInterrupt:
        if model is not None:
            status['status'] = 'interrupted'
            report()
            save_pair('_interrupted')
        raise
    except CurriculumEarlyStopped:
        raise
    except BaseException:
        if model is not None:
            status['status'] = 'failed'
            report()
            save_pair('_failed')
        raise
    finally:
        if bar is not None:
            bar.bar.close()
        env.close()
