"""The yaw-aware recipe must not advance on position success alone."""
from types import SimpleNamespace
import unittest
import numpy as np
from bluerov2_gym.workflows.training.mastery import StationkeepingMasteryCallback, PlateauEarlyStoppingCallback
from bluerov2_gym.workflows.training.stationkeeping_yaw import stationkeeping_observation


class StationkeepingYawTests(unittest.TestCase):
    def bind(self, callback):
        callback.init_callback(SimpleNamespace(get_env=lambda: SimpleNamespace(num_envs=1),
            logger=SimpleNamespace(record=lambda *args: None)))
        callback._on_training_start()
        return callback

    def test_yaw_failure_blocks_advancement_despite_good_position(self):
        c=self.bind(StationkeepingMasteryCallback(.2, 2, 1, yaw_tolerance_deg=5.))
        for angle in [20., 2.]:
            for done in [False, True]:
                c.locals={'infos':[{'metrics/position_error_euclidean':.01,
                    'metrics/yaw_error_rad':np.deg2rad(angle),'TimeLimit.truncated':True}],
                    'dones':[done]}
                c._on_step()
            self.assertEqual(c.mastered, angle == 2.)

    def test_relative_yaw_wrap_and_world_position_error(self):
        obs={k:np.array([0.],dtype=np.float32) for k in ['x','y','z','yaw']}
        obs['x'][0]=2.;obs['yaw'][0]=np.deg2rad(-179.)
        ref=np.zeros(12);ref[0]=1.;ref[5]=np.deg2rad(179.)
        result=stationkeeping_observation(obs,ref)
        self.assertAlmostEqual(result['x'].item(),1.)
        self.assertAlmostEqual(np.rad2deg(result['yaw'].item()),2.,places=4)

    def test_yaw_improvement_resets_stagnation_despite_constant_position_error(self):
        c=self.bind(PlateauEarlyStoppingCallback(window_episodes=1,patience_windows=2,
            min_delta_m=.05,warmup_episodes=0,metric_key='metrics/stationkeeping_error'))
        for score in [4.,4.,2.,2.]:
            c.locals={'infos':[{'metrics/position_error_euclidean':.01,
                'metrics/stationkeeping_error':score}],'dones':[True]}
            self.assertTrue(c._on_step())
        self.assertEqual(c.best_error_m,2.)
        self.assertEqual(c.stale_windows,1)


if __name__=='__main__':unittest.main()
