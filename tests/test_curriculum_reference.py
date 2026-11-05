"""Protect the user's validated 0436ce3 curriculum recipe and phase lifecycle."""
import ast
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import yaml

from bluerov2_gym.workflows.configuration import resolve
from bluerov2_gym.workflows.training import curriculum

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = '0436ce3482e95105cb2376681b2bd6cbec72d99c'


def reference_source(path):
    return subprocess.check_output(['git', 'show', f'{REFERENCE}:{path}'], cwd=ROOT, text=True)


class CurriculumReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = reference_source('training/train_curriculum.py')
        cls.reference = types.ModuleType('validated_curriculum_reference')
        exec(compile(cls.source, '0436ce3/training/train_curriculum.py', 'exec'), cls.reference.__dict__)

    def test_ppo_normalization_curriculum_and_reward_match_user_reference(self):
        _, config = resolve(ROOT/'configs/training/curriculum.yaml')
        defaults = self.reference.DEFAULT_CONFIG
        for key in ['ppo', 'vecnormalize', 'checkpoint', 'curriculum', 'jonswap_defaults', 'initial_jonswap']:
            self.assertEqual(config[key], defaults[key], key)
        original_yaml = yaml.safe_load(reference_source('config/training_config.yaml'))
        self.assertEqual(config['environment']['config'], original_yaml['environment']['config'])
        self.assertIsNone(config['training']['seed'])
        self.assertEqual(config['environment']['dynamics_config']['mechanics_model'], 'corrected')
        self.assertEqual(config['environment']['dynamics_config']['disturbance_model']['preset'], 'std')

    def test_parameter_and_phase_update_functions_preserve_original_calculation(self):
        trees = [ast.parse(self.source), ast.parse(Path(curriculum.__file__).read_text())]
        for name in ['make_env', 'build_jonswap_params', 'set_curriculum_level']:
            nodes = [next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
                     for tree in trees]
            self.assertEqual(ast.dump(nodes[0]), ast.dump(nodes[1]), name)

    def test_one_policy_and_normalizer_continue_across_the_same_five_phases(self):
        _, config = resolve(ROOT/'configs/training/curriculum.yaml')
        traces = []
        constructors = []
        for module in [self.reference, curriculum]:
            trace, constructed = [], []
            class Model:
                def __init__(self, **kwargs):
                    self.env = kwargs['env']
                    self.n_steps = kwargs['n_steps']
                    self.num_timesteps = 0
                    kwargs.setdefault('seed', None)
                    constructed.append({k: v for k, v in kwargs.items()
                                        if k not in ['env', 'tensorboard_log', 'device']})
                    self.env_ids = []
                def learn(self, total_timesteps, callback, tb_log_name, reset_num_timesteps):
                    self.env_ids.append(id(self.env))
                    dynamics = self.env.envs[0].unwrapped.dynamics
                    # Real normalization and physical environment, one step per
                    # phase; avoid repeating millions of optimization steps.
                    self.env.step(np.zeros((1, 6), dtype=np.float32))
                    trace.append((dict(dynamics.jonswap_params), reset_num_timesteps,
                                  total_timesteps, float(self.env.obs_rms['x'].count)))
                    self.num_timesteps += int(np.ceil(total_timesteps/self.n_steps))*self.n_steps
                    return self
                def save(self, path):
                    self.assert_single_normalizer = len(set(self.env_ids)) == 1
                    if not self.assert_single_normalizer:
                        raise AssertionError('The policy switched normalizers between phases')
            with tempfile.TemporaryDirectory() as folder:
                copied = dict(config)
                # Compare the original step-based rule separately from the
                # newly requested goal-based transition mode.
                copied['curriculum_transition'] = {'mode': 'timesteps'}
                copied['paths'] = dict(config['paths'], model_dir=folder, tensorboard_log=folder)
                path = Path(folder)/'config.yaml'
                path.write_text(yaml.safe_dump(copied))
                with patch.object(module, 'PPO', Model), redirect_stdout(StringIO()):
                    module.train_model_with_curriculum(str(path))
            self.assertEqual(len(constructed), 1)
            self.assertEqual([t[1] for t in trace], [True, False, False, False, False])
            self.assertTrue(all(trace[i][3] > trace[i-1][3] for i in range(1, len(trace))))
            traces.append(trace)
            constructors.append(constructed)
        self.assertEqual(constructors[0], constructors[1])
        self.assertEqual(traces[0], traces[1])


if __name__ == '__main__':
    unittest.main()
