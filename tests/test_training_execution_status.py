import unittest
from bluerov2_gym.workflows.plotting.training.status import execution_status


class ExecutionStatusTests(unittest.TestCase):
    def test_runtime_exit_is_visible_before_manifest_updates(self):
        result = execution_status({'status': 'early_stopped', 'stale_evaluations': 12,
                                   'stop_reason': 'Validation plateau', 'updated_at': 123},
                                  {'status': 'created'})
        self.assertTrue(result['terminal'])
        self.assertEqual(result['state'], 'early_stopped')
        self.assertIn('12', result['reason'])
        self.assertEqual(result['ended_at'], 123)

    def test_manifest_failure_overrides_running_state(self):
        result = execution_status({'status': 'training'}, {'status': 'failed', 'error': 'Disk full'})
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['recorded_reason'], 'Disk full')

    def test_completed_is_distinct_from_interruption(self):
        for state in ['completed', 'interrupted']:
            result = execution_status({'status': state})
            self.assertEqual(result['state'], state)
            self.assertTrue(result['terminal'])

    def test_acquisition_and_unknown_are_not_reported_as_finished(self):
        result = execution_status({'status': 'training', 'progress_monitor': {'active': False}})
        self.assertIn('Aquisição', result['reason'])
        self.assertFalse(result['terminal'])
        self.assertFalse(execution_status()['terminal'])
