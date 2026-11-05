"""Checks for physical error logging and paired checkpoint selection."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np

from bluerov2_gym.workflows.training.metrics import PositionErrorCallback
from bluerov2_gym.workflows.evaluation.checkpoints import discover_pairs


class TrainingMetricTests(unittest.TestCase):
    def callback(self, n_envs):
        values={}
        model=SimpleNamespace(get_env=lambda:SimpleNamespace(num_envs=n_envs),
                              logger=SimpleNamespace(record=lambda key,value:values.update({key:value})))
        callback=PositionErrorCallback()
        callback.init_callback(model)
        callback._on_training_start()
        callback._on_rollout_start()
        return callback,values

    def test_error_is_measured_in_meters_and_handles_vector_episodes(self):
        callback,values=self.callback(2)
        callback.locals={'infos':[{'metrics/position_error_euclidean':3.},
                                  {'metrics/position_error_euclidean':4.}], 'dones':[False,True]}
        self.assertTrue(callback._on_step())
        callback.locals={'infos':[{}, {'metrics/position_error_euclidean':0.}], 'dones':[True,True]}
        callback._on_step();callback._on_rollout_end()
        self.assertAlmostEqual(values['rollout/position_error_mean_m'],7/3)
        self.assertAlmostEqual(values['rollout/position_error_rmse_m'],np.sqrt(25/3))
        self.assertAlmostEqual(values['rollout/episode_position_error_mean_m'],7/3)
        self.assertEqual(list(callback.completed),[4.,3.,0.])

    def test_missing_error_is_not_reported_as_zero(self):
        callback,values=self.callback(1)
        callback.locals={'infos':[{}],'dones':[True]}
        callback._on_step();callback._on_rollout_end()
        self.assertEqual(values,{})

    def test_checkpoint_requires_matching_normalizer(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder);models=run/'models';models.mkdir()
            checkpoint=models/'agent_checkpoint_seed_42_100_steps.zip';checkpoint.touch()
            config={'paths':{'model_name':'agent','vecnormalize_name':'normalizer.pkl'}}
            with self.assertRaises(FileNotFoundError):discover_pairs(run,config)
            norm=models/'agent_checkpoint_seed_42_vecnormalize_100_steps.pkl';norm.touch()
            final=models/'agent_seed_42.zip';final.touch()
            final_norm=models/'normalizer_seed_42.pkl';final_norm.touch()
            pairs=discover_pairs(run,config)
            self.assertEqual([(p[1],p[2]) for p in pairs],[(checkpoint,norm),(final,final_norm)])


if __name__=='__main__':unittest.main()
