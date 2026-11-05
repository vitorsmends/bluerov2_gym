"""Goal-based curriculum must reject failures and insufficient hold time."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import yaml

from bluerov2_gym.workflows.configuration import resolve
from bluerov2_gym.workflows.training.mastery import StationkeepingMasteryCallback
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

    def test_budget_without_mastery_stops_before_next_phase_and_saves_partial(self):
        _, config = resolve(ROOT/'configs/training/curriculum.yaml')
        calls = []
        class Model:
            def __init__(self, **kwargs):
                self.num_timesteps = 0
            def learn(self, **kwargs):
                calls.append(kwargs)
                self.num_timesteps += kwargs['total_timesteps']
            def save(self, path):
                Path(str(path)+'.zip').touch()
        with tempfile.TemporaryDirectory() as folder:
            config['paths'].update(model_dir=folder, tensorboard_log=folder)
            path = Path(folder)/'input.yaml'
            path.write_text(yaml.safe_dump(config))
            with patch.object(curriculum, 'PPO', Model):
                with self.assertRaisesRegex(RuntimeError, 'No advance'):
                    curriculum.train_model_with_curriculum(str(path))
            self.assertEqual(len(calls), 1)
            self.assertTrue((Path(folder)/'bluerov_ppo_curriculum_phase_1_incomplete.zip').exists())
            self.assertFalse((Path(folder)/'bluerov_ppo_curriculum.zip').exists())
            status = yaml.safe_load((Path(folder)/'curriculum_status.yaml').read_text())
            self.assertEqual(status['status'], 'budget_exhausted')

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
