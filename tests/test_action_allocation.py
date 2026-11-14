"""Verify the policy/actuator contract and physical saturation."""
import unittest
import numpy as np
from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.workflows.action_allocation import PolicyActionAllocation


class AllocationTests(unittest.TestCase):
    def setUp(self):
        self.dynamics = Dynamics()
        self.allocator = PolicyActionAllocation(self.dynamics, {'mode': 'body_wrench'})

    def test_unsaturated_requested_body_axes_are_realized(self):
        for axis in range(5):
            action = np.eye(5)[axis] * .01
            thrust, info = self.allocator.convert(action)
            actual = self.dynamics.allocation_matrix @ thrust
            np.testing.assert_allclose(actual[self.allocator.axes],
                                       action * self.allocator.limits, atol=1e-6)
            self.assertEqual(info['allocation/scale'], 1.)

    def test_saturation_preserves_selected_wrench_direction(self):
        action = np.array([1., -1., 1., 1., -1.])
        thrust, info = self.allocator.convert(action)
        self.assertTrue(np.all(thrust >= self.dynamics.thruster_min))
        self.assertTrue(np.all(thrust <= self.dynamics.thruster_max))
        self.assertLess(info['allocation/scale'], 1.)
        np.testing.assert_allclose(info['allocation/achieved_wrench'][self.allocator.axes],
                                   action * self.allocator.limits * info['allocation/scale'], atol=1e-5)

    def test_legacy_default_is_identity(self):
        action = np.arange(6, dtype=float)
        legacy = PolicyActionAllocation(self.dynamics)
        self.assertIs(legacy.convert(action)[0], action)

    def test_invalid_interface_fails(self):
        with self.assertRaises(ValueError):
            self.allocator.convert(np.zeros(6))
        with self.assertRaises(ValueError):
            self.allocator.convert(np.full(5, np.nan))
