"""Exercise mechanics switches, isolated validation and the complete curriculum loop."""
from copy import deepcopy
import json
from pathlib import Path
import random
import pickle
import tempfile
import unittest

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import torch
import yaml

from bluerov2_gym.workflows.configuration import resolve
from bluerov2_gym.workflows.training.curriculum_context import (
    PhysicsCurriculumEnv, HISTORICAL_DAMPING, sample_context,
)
from bluerov2_gym.workflows.training.trajectory_curriculum import (
    batch_result, episode_result, evaluate_stage, train_trajectory_curriculum,
)
from bluerov2_gym.workflows.training.mastery import CurriculumEarlyStopped
from bluerov2_gym.workflows.plotting.training.browser import TrainingData


ROOT = Path(__file__).resolve().parents[1]


def configuration():
    return resolve(ROOT / 'configs/training/trajectory_curriculum.yaml')[1]


def smoke_configuration(folder):
    config = configuration()
    for stage in config['curriculum']:
        stage.pop('evaluation', None)
        stage.pop('transition', None)
    config['paths']['model_dir'] = str(folder / 'models')
    config['paths']['tensorboard_log'] = str(folder / 'logs/tensorboard')
    config['ppo'].update(n_steps=8, batch_size=4, n_epochs=1, verbose=0)
    config['training']['device'] = 'cpu'
    config['training']['phase_planned_timesteps'] = 8
    config['evaluation'].update(every_timesteps=8, duration_seconds=.2,
                                settling_time_seconds=0., episodes=2, groups=1)
    config['curriculum_transition'].update(position_tolerance_m=100., max_rmse_m=100.,
                                          min_tracking_fraction=1., consecutive_batches=2)
    return config


class PhysicsWaveCurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_phase_switch_rebuilds_mass_and_damping_only_on_reset(self):
        config = configuration()
        env = PhysicsCurriculumEnv(config, 42)
        try:
            env.reset(seed=42)
            self.assertEqual(env.dynamics.mechanics_model, 'std')
            old_mass = env.dynamics.M.copy()
            np.testing.assert_array_equal(env.dynamics.D_quad, HISTORICAL_DAMPING)
            env.step(np.zeros(6))
            np.testing.assert_array_equal(env.state['tau_wave'], np.zeros(6))
            np.testing.assert_array_equal(env.state['nu_current'], np.zeros(3))
            self.assertEqual(env.state['wave_elevation'], 0.)
            env.set_stage(1)
            env.reset(seed=42)
            self.assertEqual(env.dynamics.get_jonswap_params()['seed'], 42)
            self.assertEqual(env.dynamics.get_jonswap_params()['Hs'], 2.)
            self.assertFalse(env.dynamics.disturbance_model['normalize_spectrum'])
            env.set_stage(2)
            self.assertEqual(env.dynamics.mechanics_model, 'std')
            env.reset(seed=42)
            self.assertEqual(env.dynamics.mechanics_model, 'corrected')
            self.assertFalse(np.array_equal(old_mass, env.dynamics.M))
            self.assertTrue(env.dynamics.disturbance_model['normalize_spectrum'])
            dw = env.dynamics._waves[1]['omega'] - env.dynamics._waves[0]['omega']
            energy = sum(w['S'] for w in env.dynamics._waves) * dw
            self.assertAlmostEqual(energy, (env.dynamics.Hs / 4.)**2)
        finally:
            env.close()

    def test_rehearsal_never_reintroduces_old_mechanics_in_target_phases(self):
        config = configuration()
        env = PhysicsCurriculumEnv(config, 23)
        try:
            env.set_stage(5)
            contexts = []
            for _ in range(25):
                env.reset()
                contexts.append(env.current_context['stage_index'])
                self.assertGreaterEqual(contexts[-1], 2)
                self.assertEqual(env.dynamics.mechanics_model, 'corrected')
                self.assertTrue(env.dynamics.disturbance_model['normalize_spectrum'])
            self.assertTrue(any(i < 5 for i in contexts))
        finally:
            env.close()

    def test_batch_cannot_hide_failure_in_a_harder_group(self):
        settings = configuration()['curriculum_transition']
        episodes = [dict(group=0, success=True, rmse_m=.01) for _ in range(19)]
        episodes.append(dict(group=1, success=False, rmse_m=1.))
        result = batch_result(episodes, settings)
        self.assertGreater(result['success_rate'], .9)
        self.assertFalse(result['passed'])
        result = episode_result([.01]*100, True, False, 100, 10, settings, 0)
        self.assertFalse(result['success'])

    def test_validation_preserves_training_rng_and_normalizer(self):
        with tempfile.TemporaryDirectory() as folder:
            config = smoke_configuration(Path(folder))
            env = VecNormalize(DummyVecEnv([lambda: PhysicsCurriculumEnv(config, 42)]),
                               **config['vecnormalize'])
            try:
                model = PPO('MultiInputPolicy', env, n_steps=8, batch_size=4, n_epochs=1, device='cpu')
                env.reset()
                numpy_state, python_state = np.random.get_state(), random.getstate()
                torch_state = torch.get_rng_state().clone()
                counts = {k: v.count for k, v in env.obs_rms.items()}
                means = {k: v.mean.copy() for k, v in env.obs_rms.items()}
                for batch in range(2):
                    result = evaluate_stage(model, env, config, 2, batch)
                    for episode in result['episodes']:
                        self.assertGreaterEqual(episode['context']['jonswap']['seed'], 2**31)
                    if batch == 0:
                        seeds = [e['reset_seed'] for e in result['episodes']]
                    else:
                        self.assertNotEqual(seeds, [e['reset_seed'] for e in result['episodes']])
                np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
                self.assertEqual(np.random.get_state()[2:], numpy_state[2:])
                self.assertEqual(random.getstate(), python_state)
                self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
                for key in means:
                    self.assertEqual(env.obs_rms[key].count, counts[key])
                    np.testing.assert_array_equal(env.obs_rms[key].mean, means[key])
            finally:
                env.close()

    def test_complete_six_phase_training_keeps_optimizer_and_normalization(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = smoke_configuration(root)
            (root/'config.resolved.yaml').write_text(yaml.safe_dump(config))
            train_trajectory_curriculum(config)
            model_dir = root/'models'
            status = yaml.safe_load((model_dir/'curriculum_status.yaml').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(len(status['history']), 6)
            self.assertEqual(sum(s['validation_timesteps'] for s in status['history']), 48)
            # Planned 8 steps does NOT force an advance: two successful batches
            # require 16 training steps for every stage in this smoke experiment.
            self.assertTrue(all(s['phase_steps'] == 16 for s in status['history']))
            model = PPO.load(model_dir/'ppo_trajectory_curriculum.zip', device='cpu')
            self.assertEqual(model.num_timesteps, 96)
            self.assertEqual(model._n_updates, 12)
            self.assertEqual(model.policy_kwargs, {})
            with (model_dir/'vec_normalize_trajectory_curriculum.pkl').open('rb') as stream:
                saved_normalizer = pickle.load(stream)
            self.assertGreater(saved_normalizer.obs_rms['x'].count, 96)
            for stage in range(1, 7):
                self.assertTrue((model_dir/f'ppo_trajectory_curriculum_stage_{stage}.zip').exists())
                self.assertTrue((model_dir/f'vec_normalize_trajectory_curriculum_stage_{stage}.pkl').exists())
            evaluations = list((root/'logs/curriculum_evaluations').glob('*.yaml'))
            self.assertEqual(len(evaluations), 12)
            viewer = TrainingData(root, interval=0).read()
            json.dumps(viewer, allow_nan=False)
            self.assertIn('6/6', viewer['phase'])
            self.assertIn('phase_plan', viewer['progress'])
            self.assertIsNone(viewer['progress']['phase_plan']['remaining_seconds'])

    def test_mastery_can_finish_before_planned_steps(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = smoke_configuration(root)
            config['curriculum'] = config['curriculum'][:1]
            config['training']['phase_planned_timesteps'] = 1000
            train_trajectory_curriculum(config)
            status = yaml.safe_load((root/'models/curriculum_status.yaml').read_text())
            self.assertEqual(status['status'], 'completed')
            self.assertEqual(status['phase_steps'], 16)
            self.assertLess(status['phase_steps'], status['planned_timesteps'])

    def test_optional_plateau_stops_incomplete_without_skipping_stage(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = smoke_configuration(root)
            config['curriculum_transition'].update(position_tolerance_m=1e-12, max_rmse_m=1e-12)
            config['plateau_early_stopping'].update(enabled=True, patience_evaluations=1,
                                                   warmup_evaluations=0, min_delta_m=100.)
            with self.assertRaises(CurriculumEarlyStopped):
                train_trajectory_curriculum(config)
            status = yaml.safe_load((root/'models/curriculum_status.yaml').read_text())
            self.assertEqual(status['index'], 1)
            self.assertEqual(status['status'], 'early_stopped')
            self.assertEqual(status['history'], [])
            self.assertFalse((root/'models/ppo_trajectory_curriculum.zip').exists())
            self.assertTrue((root/'models/ppo_trajectory_curriculum_incomplete.zip').exists())

    def test_resume_preserves_steps_optimizer_and_normalizer(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root/'source'
            source.mkdir()
            config = smoke_configuration(source)
            config['curriculum'] = config['curriculum'][:1]
            config['curriculum_transition'].update(position_tolerance_m=1e-12, max_rmse_m=1e-12)
            config['plateau_early_stopping'].update(enabled=True, patience_evaluations=1,
                                                   warmup_evaluations=0, min_delta_m=100.)
            (source/'config.resolved.yaml').write_text(yaml.safe_dump(config))
            with self.assertRaises(CurriculumEarlyStopped):
                train_trajectory_curriculum(config)
            continued = deepcopy(config)
            continued['training']['resume_run'] = str(source)
            continued['paths']['model_dir'] = str(root/'continued/models')
            continued['paths']['tensorboard_log'] = str(root/'continued/logs/tensorboard')
            continued['plateau_early_stopping']['enabled'] = False
            continued['curriculum_transition'].update(position_tolerance_m=100., max_rmse_m=100.)
            train_trajectory_curriculum(continued)
            model = PPO.load(root/'continued/models/ppo_trajectory_curriculum.zip', device='cpu')
            self.assertEqual(model.num_timesteps, 32)
            self.assertEqual(model._n_updates, 4)
            with (root/'continued/models/vec_normalize_trajectory_curriculum.pkl').open('rb') as stream:
                normalizer = pickle.load(stream)
            self.assertGreater(normalizer.obs_rms['x'].count, 32)


if __name__ == '__main__':
    unittest.main()
