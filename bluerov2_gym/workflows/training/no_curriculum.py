"""Optional fixed-sea stationkeeping experiment; curriculum remains standard."""
from pathlib import Path

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from .curriculum import TrainingProgressCallback
from .metrics import PositionErrorCallback
from .device import resolve_training_device, report_training_device
from ..artifacts import write_yaml


def train_without_curriculum(config):
    environment = config['environment']
    paths = config['paths']
    models = Path(paths['model_dir'])
    models.mkdir(parents=True, exist_ok=True)

    def make_env():
        return Monitor(gym.make(
            environment['id'], render_mode=environment.get('render_mode'),
            max_episode_steps=int(environment['max_episode_steps']),
            env_config=environment.get('config'),
            dynamics_config=environment['dynamics_config'],
        ))

    env = VecNormalize(DummyVecEnv([make_env]), **config['vecnormalize'])
    try:
        params = dict(config['ppo'])
        policy = params.pop('policy')
        requested_device = config['training'].get('device', params.pop('device', 'auto'))
        selected_device = resolve_training_device(requested_device)
        report_training_device(requested_device, selected_device)
        model = PPO(policy, env, seed=config['training'].get('seed'),
                    tensorboard_log=paths['tensorboard_log'], device=selected_device, **params)
        checkpoint = config['checkpoint']
        steps = int(config['training']['total_timesteps'])
        callback = CallbackList([
            CheckpointCallback(save_freq=int(checkpoint['save_freq']),
                               save_path=str(models), name_prefix=checkpoint['name_prefix'],
                               save_vecnormalize=True),
            PositionErrorCallback(),
            TrainingProgressCallback(steps, steps, 'Fixed sea (no curriculum)', 1, 1),
        ])
        print('Task: stationkeeping_no_curriculum | fixed sea | raw state observations', flush=True)
        print(f'PPO: n_steps={model.n_steps}, batch_size={model.batch_size}, '
              f'policy_kwargs={model.policy_kwargs}, seed={model.seed}', flush=True)
        model.learn(total_timesteps=steps, callback=callback,
                    tb_log_name=config['training']['tb_log_name'])
        model.save(str(models / paths['model_name']))
        env.save(str(models / paths['vecnormalize_name']))
        write_yaml(models / f"{paths['model_name']}_config.yaml", {
            'config': config, 'actual_timesteps': model.num_timesteps,
            'task': 'stationkeeping_no_curriculum',
            'execution_device': selected_device,
        })
    finally:
        env.close()
