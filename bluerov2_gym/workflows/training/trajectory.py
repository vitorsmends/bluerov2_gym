"""Trajectory PPO recipe from 889a766, integrated with run storage and devices."""
from pathlib import Path

import numpy as np
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .curriculum import TrainingProgressCallback
from .device import resolve_training_device, report_training_device
from .metrics import PositionErrorCallback
from .trajectory_env import TrajectoryTrackingEnv
from ..artifacts import write_yaml


class TensorboardCallback(BaseCallback):
    """Preserve the reference custom tags and add raw step reward monitoring."""
    def _on_rollout_start(self):
        self.raw_rewards = []

    def _on_step(self):
        for info in self.locals.get('infos', []):
            for name in ['tracking_error_m', 'velocity_error', 'is_success']:
                if name in info:
                    self.logger.record(f'custom/{name}', float(info[name]))
            if 'metrics/trajectory_reward_raw' in info:
                self.raw_rewards.append(float(info['metrics/trajectory_reward_raw']))
        return True

    def _on_rollout_end(self):
        # Episodes have no time limit in the reference and may not finish for
        # many rollouts. Keep a meaningful reward curve even in that case.
        if self.raw_rewards:
            self.logger.record('rollout/reward_mean_per_step', float(np.mean(self.raw_rewards)))


def train_trajectory(config):
    environment = config['environment']
    paths = config['paths']
    models = Path(paths['model_dir'])
    models.mkdir(parents=True, exist_ok=True)

    def make_env():
        env = TrajectoryTrackingEnv(
            env_config=environment.get('config'),
            dynamics_config=environment['dynamics_config'],
        )
        # The source directly instantiates the subclass, so its Gym registration
        # limit is NOT applied. None reproduces the effective historical behavior.
        limit = environment.get('max_episode_steps')
        if limit is not None:
            env = TimeLimit(env, max_episode_steps=int(limit))
        return Monitor(env, info_keywords=('is_success', 'tracking_error_m'))

    env = VecNormalize(DummyVecEnv([make_env]), **config['vecnormalize'])
    try:
        params = dict(config['ppo'])
        policy = params.pop('policy')
        requested = config['training'].get('device', params.pop('device', 'auto'))
        device = resolve_training_device(requested)
        report_training_device(requested, device)
        model = PPO(policy, env, seed=config['training'].get('seed'),
                    tensorboard_log=paths['tensorboard_log'], device=device, **params)
        checkpoint = config['checkpoint']
        steps = int(config['training']['total_timesteps'])
        callbacks = CallbackList([
            CheckpointCallback(save_freq=int(checkpoint['save_freq']),
                               save_path=str(models), name_prefix=checkpoint['name_prefix'],
                               save_vecnormalize=True),
            TensorboardCallback(),
            PositionErrorCallback(),
            TrainingProgressCallback(steps, steps, 'Reference trajectory 889a766', 1, 1),
        ])
        manifest_path = models / f"{paths['model_name']}_config.yaml"
        manifest = dict(config=config, task='trajectory_tracking',
                        execution_device=device, observation_adapter='legacy',
                        effective_episode_limit=environment.get('max_episode_steps'))
        write_yaml(manifest_path, manifest)
        print('Task: trajectory_tracking | reference 889a766 | current dynamics/disturbance', flush=True)
        print(f'PPO: n_steps={model.n_steps}, batch_size={model.batch_size}, '
              f'policy_kwargs={model.policy_kwargs}, seed={model.seed}', flush=True)
        model.learn(total_timesteps=steps, callback=callbacks,
                    tb_log_name=config['training']['tb_log_name'])
        model.save(str(models / paths['model_name']))
        env.save(str(models / paths['vecnormalize_name']))
        write_yaml(manifest_path, dict(manifest, actual_timesteps=model.num_timesteps))
    finally:
        env.close()
