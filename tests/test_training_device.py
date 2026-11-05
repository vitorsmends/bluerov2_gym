"""Device selection must not silently replace an explicitly requested GPU."""
import unittest
from unittest.mock import patch

from bluerov2_gym.workflows.training.device import resolve_training_device


class DeviceTests(unittest.TestCase):
    def test_auto_selects_cuda_when_pytorch_can_access_it(self):
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(resolve_training_device('auto'), 'cuda')

    def test_auto_uses_cpu_without_cuda(self):
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(resolve_training_device('auto'), 'cpu')

    def test_explicit_cuda_does_not_silently_fall_back(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                resolve_training_device('cuda')

    def test_explicit_cpu_and_gpu_index(self):
        with patch('torch.cuda.is_available', return_value=True), patch('torch.cuda.device_count', return_value=2):
            self.assertEqual(resolve_training_device('cpu'), 'cpu')
            self.assertEqual(resolve_training_device('cuda:1'), 'cuda:1')
            with self.assertRaises(ValueError):
                resolve_training_device('cuda:2')
        with self.assertRaises(ValueError):
            resolve_training_device('invalid')


if __name__ == '__main__':
    unittest.main()
