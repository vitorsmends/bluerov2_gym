"""Goal-based curriculum must reject failures and insufficient hold time."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from bluerov2_gym.workflows.configuration import resolve
from bluerov2_gym.workflows.training.mastery import (StationkeepingMasteryCallback,
    PlateauEarlyStoppingCallback, CurriculumEarlyStopped)
from bluerov2_gym.workflows.training import curriculum
from bluerov2_gym.workflows.plotting.training.status import training_phase

ROOT = Path(__file__).resolve().parents[1]


class MasteryTests(unittest.TestCase):
    def callback(self, required=3):
        callback = StationkeepingMasteryCallback(.2, 2, required)
        model = SimpleNamespace(get_env=lambda: SimpleNamespace(num_envs=1),
                                logger=SimpleNamespace(record=lambda *args: None))
        callback.init_callback(model)
        callback._on_training_start()
        return callback

    def step(self, callback, error, done=False, truncated=True):
        info = {'TimeLimit.truncated': truncated}
        if error is not None:
            info['metrics/position_error_euclidean'] = error
        callback.locals = {'infos': [info], 'dones': [done]}
        return callback._on_step()

    def episode(self, callback, errors, truncated=True):
        result = True
        for i, error in enumerate(errors):
            result = self.step(callback, error, i == len(errors)-1, truncated)
        return result

    def test_only_sustained_success_in_consecutive_completed_episodes_advances(self):
        c = self.callback()
        self.assertTrue(self.episode(c, [2., .1, .1]))
        self.assertTrue(self.episode(c, [.1, .1]))
        self.assertFalse(c.mastered)
        self.assertFalse(self.episode(c, [.1, .1]))
        self.assertTrue(c.mastered)

    def test_physical_failure_breaks_streak_even_with_low_position_error(self):
        c = self.callback()
        self.episode(c, [.1, .1])
        self.episode(c, [.1, .1], truncated=False)
        self.assertEqual(c.streak, 0)
        self.assertFalse(c.mastered)

    def test_missing_errors_short_episodes_and_excursions_do_not_qualify(self):
        for errors in [[.1], [None, .1], [.3, .1], [float('nan'), .1], [.2, .1]]:
            c = self.callback(1)
            self.assertTrue(self.episode(c, errors))
            self.assertFalse(c.mastered)

    def test_optional_early_stopping_saves_partial_without_advancing(self):
        _, config = resolve(ROOT/'configs/training/curriculum.yaml')
        calls = []
        class Model:
            def __init__(self, **kwargs):
                self.num_timesteps = 0
            def learn(self, **kwargs):
                calls.append(kwargs)
                self.num_timesteps += 1000001
                assert kwargs['total_timesteps'] == float('inf')
                for callback in kwargs['callback'].callbacks:
                    if isinstance(callback, PlateauEarlyStoppingCallback):
                        callback.stopped = True
            def save(self, path):
                Path(str(path)+'.zip').touch()
        with tempfile.TemporaryDirectory() as folder:
            config['paths'].update(model_dir=folder, tensorboard_log=folder)
            config['curriculum_transition']['early_stopping']['enabled'] = True
            path = Path(folder)/'input.yaml'
            path.write_text(yaml.safe_dump(config))
            with patch.object(curriculum, 'PPO', Model):
                with self.assertRaisesRegex(CurriculumEarlyStopped, 'No advance'):
                    curriculum.train_model_with_curriculum(str(path))
            self.assertEqual(len(calls), 1)
            self.assertTrue((Path(folder)/'bluerov_ppo_curriculum_phase_1_incomplete.zip').exists())
            self.assertFalse((Path(folder)/'bluerov_ppo_curriculum.zip').exists())
            status = yaml.safe_load((Path(folder)/'curriculum_status.yaml').read_text())
            self.assertEqual(status['status'], 'early_stopped')

    def test_phase_can_exceed_old_step_limits_and_advance_only_on_mastery(self):
        _, config = resolve(ROOT/'configs/training/curriculum.yaml')
        calls = []
        class Model:
            def __init__(self, **kwargs):
                self.num_timesteps = 0
            def learn(self, **kwargs):
                self.assert_unbounded = kwargs['total_timesteps'] == float('inf')
                if not self.assert_unbounded:
                    raise AssertionError('A phase still has a step limit')
                calls.append(kwargs)
                self.num_timesteps += 2000001
                for callback in kwargs['callback'].callbacks:
                    if isinstance(callback, StationkeepingMasteryCallback):
                        callback.mastered = True
            def save(self, path):
                Path(str(path)+'.zip').touch()
        with tempfile.TemporaryDirectory() as folder:
            config['paths'].update(model_dir=folder, tensorboard_log=folder)
            path = Path(folder)/'input.yaml'
            path.write_text(yaml.safe_dump(config))
            with patch.object(curriculum, 'PPO', Model):
                curriculum.train_model_with_curriculum(str(path))
            self.assertEqual(len(calls), 5)
            self.assertTrue((Path(folder)/'bluerov_ppo_curriculum.zip').exists())

    def test_early_stopping_uses_improvement_not_number_of_steps(self):
        c = PlateauEarlyStoppingCallback(window_episodes=2, patience_windows=2,
                                        min_delta_m=.01, warmup_episodes=0)
        model = SimpleNamespace(get_env=lambda: SimpleNamespace(num_envs=1),
                                logger=SimpleNamespace(record=lambda *args: None))
        c.init_callback(model)
        c._on_training_start()
        for error in [1., 1., 1., 1., .5, .5]:
            self.assertTrue(self.episode(c, [error]))
        self.assertEqual(c.stale_windows, 0)
        for error in [.5, .5, .5]:
            self.assertTrue(self.episode(c, [error]))
        self.assertFalse(self.episode(c, [.5]))
        self.assertTrue(c.stopped)
        self.assertAlmostEqual(c.best_error_m, .5)

    def test_real_ppo_updates_with_unbounded_horizon_until_goal_callback_stops(self):
        class GoalEnv(gym.Env):
            observation_space = gym.spaces.Dict({'x': gym.spaces.Box(-1., 1., (1,), dtype=np.float32)})
            action_space = gym.spaces.Box(-1., 1., (1,), dtype=np.float32)
            def reset(self, *, seed=None, options=None):
                super().reset(seed=seed)
                self.steps = 0
                return {'x': np.zeros(1, dtype=np.float32)}, {}
            def step(self, action):
                self.steps += 1
                return ({'x': np.zeros(1, dtype=np.float32)}, 1., False, self.steps == 10,
                        {'metrics/position_error_euclidean': .1})
        env = VecNormalize(DummyVecEnv([GoalEnv]))
        try:
            model = PPO('MultiInputPolicy', env, n_steps=8, batch_size=4,
                        n_epochs=1, device='cpu', seed=42)
            goal = StationkeepingMasteryCallback(.2, 2, 3)
            model.learn(total_timesteps=float('inf'), callback=goal)
            self.assertTrue(goal.mastered)
            self.assertEqual(model.num_timesteps, 30)
            self.assertGreater(model._n_updates, 0)
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder)/'agent.zip'
                model.save(path)
                self.assertEqual(PPO.load(path, device='cpu').num_timesteps, 30)
        finally:
            env.close()

    def test_dashboard_uses_recorded_phase_not_old_timestep_thresholds(self):
        config = {'curriculum_transition': {'mode': 'success'}, 'curriculum': [
            {'name': 'Calm', 'timesteps': 100, 'jonswap': {'Hs': 1.}},
            {'name': 'Storm', 'timesteps': 100, 'jonswap': {'Hs': 4.5}}]}
        runtime = {'index': 2, 'mode': 'success', 'successful_episodes': 1,
                   'required_episodes': 3, 'history': [{'index': 1, 'end_step': 15}]}
        phase, sea, boundaries = training_phase(config, 20, runtime)
        self.assertIn('Storm', phase)
        self.assertIn('1/3', phase)
        self.assertIn('Hs=4.5', sea)
        self.assertEqual(boundaries, [15])


if __name__ == '__main__':
    unittest.main()
