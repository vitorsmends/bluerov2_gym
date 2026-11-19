import unittest
from bluerov2_gym.workflows.training.curriculum_progression import (
    credit_window, plateau_advances, completion_status,
)
from bluerov2_gym.workflows.training.progress_plateau import update_progress
from bluerov2_gym.workflows.plotting.training.status import execution_status


class ProgressionTests(unittest.TestCase):
    def test_three_independent_windows_require_nine_batches(self):
        settings = dict(rolling_batches=3, nonoverlapping_windows=True)
        count = 0
        for i in range(1, 10):
            count = credit_window(count, True, i, settings)
            self.assertEqual(count, i // 3)
        count = credit_window(count, False, 12, settings)
        self.assertEqual(count, 0)

    def test_wave_exposure_can_advance_without_initial_success(self):
        settings = dict(activation_success_rate=0., require_finite_rmse_for_activation=False,
                        warmup_evaluations=0)
        result = dict(success_rate=0., group_success_rates={'0': 0.}, mean_rmse_m=None)
        state = update_progress({}, result, settings)
        state = update_progress(state, result, settings)
        self.assertTrue(state['active'])
        self.assertEqual(state['stale'], 1)
        self.assertTrue(plateau_advances({'on_plateau': 'advance'}))
        self.assertFalse(plateau_advances({}))

    def test_exposure_is_not_full_mastery(self):
        history = [{'status': 'mastered'}, {'status': 'advanced_on_plateau'}]
        self.assertEqual(completion_status(history), 'exposure_completed')
        status = execution_status({'status': completion_status(history), 'unmastered_stages': [2]})
        self.assertTrue(status['terminal'])
        self.assertEqual(status['unmastered_stages'], [2])
        self.assertEqual(completion_status([{'status': 'mastered'}]), 'completed')
