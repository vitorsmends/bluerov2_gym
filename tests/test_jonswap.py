import unittest
import numpy as np
from bluerov2_gym.envs.core.dynamics import Dynamics


class JonswapTests(unittest.TestCase):
    def test_height_scales_amplitudes_and_energy(self):
        a = Dynamics(jonswap_params={'Hs': 2.5})
        b = Dynamics(jonswap_params={'Hs': 4.5})
        np.testing.assert_allclose([w['A'] for w in b._waves], np.array([w['A'] for w in a._waves]) * 1.8)
        np.testing.assert_allclose([w['S'] for w in b._waves], np.array([w['S'] for w in a._waves]) * 1.8**2)

    def test_zero_height_removes_wave_components(self):
        d = Dynamics(jonswap_params={'Hs': 0, 'noise_std': 0})
        self.assertTrue(all(w['A'] == 0 for w in d._waves))
        for _ in range(10):
            velocity, load = d._jonswap_disturbance()
            np.testing.assert_array_equal(velocity, np.zeros(3))
            np.testing.assert_array_equal(load, np.zeros(6))

    def test_continuous_spectral_energy(self):
        # Wide, fine frequency grid verifies the Fossen/DNV approximation,
        # independently of the production discretization (cutoff = 3 wp).
        for gamma in (1, 1.5, 3.3, 5):
            hs, wp = 2.5, 2 * np.pi / 8
            omega = np.linspace(wp * .05, wp * 30, 100000)
            sigma = np.where(omega <= wp, .07, .09)
            r = np.exp(-.5 * ((omega - wp)/(sigma * wp))**2)
            spectrum = (1 - .287*np.log(gamma)) * 5/16 * hs**2 * wp**4 * omega**-5 * np.exp(-1.25*(wp/omega)**4) * gamma**r
            reconstructed = 4 * np.sqrt(np.trapezoid(spectrum, omega))
            self.assertLess(abs(reconstructed / hs - 1), .02)

    def test_invalid_height(self):
        for height in (-1, float('nan')):
            with self.assertRaises(ValueError):
                Dynamics(jonswap_params={'Hs': height})


if __name__ == '__main__':
    unittest.main()
