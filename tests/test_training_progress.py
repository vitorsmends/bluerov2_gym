"""Progress must use training timestamps and avoid unsupported ETA claims."""
import unittest

from bluerov2_gym.workflows.plotting.training.progress import training_progress


class TrainingProgressTests(unittest.TestCase):
    def test_fixed_budget_rounds_rollouts_and_uses_recent_speed(self):
        config = {'training': {'total_timesteps': 10000}, 'ppo': {'n_steps': 2048}}
        p = training_progress(config, 4096, [(2048, 110.), (4096, 120.)],
                              100., 122., 'created')
        self.assertEqual(p['total_steps'], 10240)
        self.assertEqual(p['percent'], 40.)
        self.assertEqual(p['elapsed_seconds'], 22.)
        self.assertEqual(p['remaining_seconds'], 30.)
        self.assertEqual(p['estimated_total_seconds'], 52.)

    def test_success_curriculum_has_no_step_budget_or_eta(self):
        config = {'training': {'total_timesteps': 10000},
                  'curriculum_transition': {'mode': 'success'}}
        p = training_progress(config, 4096, [(2048, 110.), (4096, 120.)],
                              100., 122., 'created')
        self.assertTrue(p['unlimited'])
        self.assertIsNone(p['percent'])
        self.assertIsNone(p['remaining_seconds'])

    def test_finished_time_does_not_keep_growing(self):
        config = {'training': {'total_timesteps': 4096}}
        p = training_progress(config, 4096, [(2048, 110.), (4096, 120.)],
                              100., 999., 'completed')
        self.assertEqual(p['elapsed_seconds'], 20.)
        self.assertEqual(p['remaining_seconds'], 0.)
        self.assertEqual(p['percent'], 100.)
        p = training_progress(config, 2048, [(2048, 110.)], 100., 999., 'interrupted')
        self.assertEqual(p['elapsed_seconds'], 10.)
        self.assertIsNone(p['remaining_seconds'])

    def test_stale_events_have_no_eta(self):
        p = training_progress({'training': {'total_timesteps': 10000}},
                              4096, [(2048, 110.), (4096, 120.)], 100., 500., 'created')
        self.assertTrue(p['stale'])
        self.assertIsNone(p['remaining_seconds'])

    def test_waits_for_events_and_does_not_claim_completion_during_last_update(self):
        config = {'training': {'total_timesteps': 4096}}
        p = training_progress(config, 0, [], None, 500., 'created')
        self.assertIsNone(p['elapsed_seconds'])
        self.assertIsNone(p['remaining_seconds'])
        p = training_progress(config, 4096, [(4096, 120.)], 100., 122., 'created')
        self.assertEqual(p['percent'], 100.)
        self.assertIsNone(p['remaining_seconds'])

    def test_timed_curriculum_rounds_each_phase(self):
        config = {'ppo': {'n_steps': 2048}, 'curriculum': [
            {'timesteps': 3000}, {'timesteps': 3000}]}
        p = training_progress(config, 4096, [], None, 500., 'created')
        self.assertEqual(p['total_steps'], 8192)
        self.assertEqual(p['percent'], 50.)


if __name__ == '__main__':
    unittest.main()
