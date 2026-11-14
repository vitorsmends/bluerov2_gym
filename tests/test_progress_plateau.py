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
