import unittest
from bluerov2_gym.workflows.training.progress_plateau import update_progress


class ProgressTests(unittest.TestCase):
    settings = dict(warmup_evaluations=0, min_delta_m=.005, min_success_delta=.025)

    def result(self, success=.9, worst=.8, rmse=.08):
        return dict(success_rate=success, group_success_rates={'0': 1., '1': worst}, mean_rmse_m=rmse)

    def test_success_improves_even_without_rmse_improvement(self):
        state = update_progress({}, self.result(), self.settings)
        state = update_progress(state, self.result(), self.settings)
        self.assertEqual(state['stale'], 1)
        state = update_progress(state, self.result(worst=.85), self.settings)
        self.assertEqual(state['stale'], 0)
        self.assertIn('worst_case', state['improved'])

    def test_recovery_and_small_fluctuations_do_not_reset(self):
        state = update_progress({}, self.result(), self.settings)
        for result in [self.result(worst=.7), self.result(), self.result(rmse=.079)]:
            state = update_progress(state, result, self.settings)
        self.assertEqual(state['stale'], 3)
        state = update_progress(state, self.result(rmse=.07), self.settings)
        self.assertEqual(state['stale'], 0)

    def test_incomplete_window_does_not_consume_patience(self):
        result = self.result()
        result['mastery_window'] = dict(batches=1, required_batches=3)
        self.assertEqual(update_progress({}, result, self.settings), {})

    def test_acquisition_does_not_exhaust_patience(self):
        settings = dict(self.settings, activation_success_rate=.5)
        state = {}
        for _ in range(50):
            state = update_progress(state, self.result(success=0., worst=0., rmse=None), settings)
        self.assertFalse(state['active'])
        self.assertEqual(state['stale'], 0)
        state = update_progress(state, self.result(success=.6), settings)
        self.assertTrue(state['active'])
        state = update_progress(state, self.result(success=0., worst=0., rmse=None), settings)
        self.assertTrue(state['active'])
        self.assertEqual(state['stale'], 1)

    def test_finetuning_yaw_improvement_counts_as_progress(self):
        settings = dict(self.settings, monitor_corner_yaw=True, min_yaw_delta_deg=.25)
        result = dict(self.result(), mean_yaw_mae_deg=10., mean_corner_rmse_m=.2)
        state = update_progress({}, result, settings)
        state = update_progress(state, result, settings)
        self.assertEqual(state['stale'], 1)
        result['mean_yaw_mae_deg'] = 9.5
        state = update_progress(state, result, settings)
        self.assertEqual(state['stale'], 0)
        self.assertIn('yaw', state['improved'])
