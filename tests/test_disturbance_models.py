"""Regression and physical checks for independently configurable wave corrections."""
import itertools
import unittest

import numpy as np

from bluerov2_gym.envs.core.dynamics import Dynamics


KEYS = ['x', 'y', 'z', 'roll', 'pitch', 'yaw', 'u', 'v', 'w', 'p', 'q', 'r']


def state():
    return dict(zip(KEYS, [.2, -.4, -2., .1, -.2, .7, .15, -.1, .05, .01, -.02, .03]))


def model(flags=None, **waves):
    return Dynamics(dynamics_config={'disturbance_model': flags or {}}, jonswap_params=waves)


class DisturbanceModelTests(unittest.TestCase):
    def test_std_matches_original_trajectory(self):
        # Captured from the unmodified implementation, with noise, nonzero
        # attitude and a varying six-thruster input over 40 integration steps.
        expected = [
            .6866618467972668, -.9177650886278532, -2.4057755630892066,
            -.014899545685324478, -.02314428344067326, .9142408677445492,
            -.18562533144796484, -.10222421614756352, -.03859965514677332,
            -.018712514342445224, .0244246932698752, .05943122009800554,
        ]
        for flags in [None, {'preset': 'std'}]:
            d = Dynamics(dynamics_config={"mechanics_model": "std",
                                          "disturbance_model": flags or {}})
            s = state()
            for i in range(40):
                d.step(s, 3 * np.sin(np.arange(6) + i * .07))
            np.testing.assert_allclose([s[k] for k in KEYS], expected, rtol=1e-13, atol=1e-14)
            np.testing.assert_allclose(s['nu_current'],
                [-.07968108832777086, -.10607198791612968, .18689781741669206], rtol=1e-13)
            self.assertAlmostEqual(s['wave_elevation'], 1.1714173426536385, places=13)

    def test_normalization_controls_discrete_energy_and_preserves_realization(self):
        baseline = model()
        for hs in [0., 1., 2., 4.]:
            for alpha in [0., .0081, .02]:
                d = model({'normalize_spectrum': True}, Hs=hs, alpha_js=alpha)
                self.assertAlmostEqual(sum(w['A']**2 / 2 for w in d._waves), (hs / 4)**2)
                for old, new in zip(baseline._waves, d._waves):
                    for key in ['omega', 'phase', 'theta', 'k']:
                        self.assertEqual(old[key], new[key])
        dry = model({'normalize_spectrum': True}, Hs=0, noise_std=0)
        velocity, load = dry._jonswap_disturbance()
        np.testing.assert_array_equal(velocity, np.zeros(3))
        np.testing.assert_array_equal(load, np.zeros(6))

    def test_finite_depth_satisfies_surface_and_seabed_conditions(self):
        d = model({'finite_depth': True, 'wave_filter': False}, noise_std=0,
                  scale=1, max_current=100, x_eval=0, y_eval=0, z_eval=0)
        omega, amplitude = .5, .2
        k = d._solve_wave_number(omega, d.water_depth)
        phase = np.pi / 4
        d._waves = [dict(omega=omega, A=amplitude, phase=phase, theta=0., k=k)]
        velocity, _ = d._jonswap_disturbance()
        arg = phase - omega * d.dt
        # w(surface) = d eta / dt for the upward-positive wave field.
        self.assertAlmostEqual(velocity[2], amplitude * omega * np.sin(arg))
        self.assertAlmostEqual(velocity[0], amplitude * omega / np.tanh(k*d.water_depth) * np.cos(arg))
        d.z_eval = -d.water_depth
        velocity, _ = d._jonswap_disturbance()
        self.assertAlmostEqual(velocity[2], 0.)

    def test_body_transform_uses_full_attitude_and_recomputes_load(self):
        for angles in [(0., 0., np.pi/2), (.3, -.4, .7)]:
            d, s = model({'body_frame': True}), state()
            s.update(zip(['roll', 'pitch', 'yaw'], angles))
            world = np.array([.2, -.1, .05])
            d._jonswap_disturbance = lambda: (world.copy(), np.zeros(6))
            eta = np.array([s[k] for k in KEYS[:6]])
            # Derive columns independently through the existing kinematics.
            rotation = np.column_stack([
                d._body_to_world_kinematics(eta, np.eye(6)[i])[:3] for i in range(3)
            ])
            body = rotation.T @ world
            expected_load = d._compute_wave_force(body)
            d.step(s, np.zeros(6))
            np.testing.assert_allclose(s['nu_current'], body, atol=1e-15)
            np.testing.assert_allclose(s['tau_wave'], expected_load, atol=1e-15)
            np.testing.assert_allclose(s['nu_rel'][:3], np.array([.15, -.1, .05]) - body)

    def test_filters_are_independent(self):
        for wave_filter, noise_filter in itertools.product([False, True], repeat=2):
            d = model(dict(wave_filter=wave_filter, noise_filter=noise_filter), noise_std=.1)
            d._waves = []
            reference_rng = np.random.default_rng()
            reference_rng.bit_generator.state = d._rng.bit_generator.state
            noise = .1 * reference_rng.standard_normal(3)
            expected = noise * (d.alpha_noise if noise_filter else 1.)
            expected *= d.alpha_wave if wave_filter else 1.
            velocity, _ = d._jonswap_disturbance()
            np.testing.assert_allclose(velocity, expected)

    def test_flags_survive_curriculum_resets_and_all_combinations_run(self):
        names = ['normalize_spectrum', 'finite_depth', 'body_frame', 'wave_filter', 'noise_filter']
        for values in itertools.product([False, True], repeat=len(names)):
            flags = dict(zip(names, values))
            d = model(flags)
            d.set_jonswap_params(Hs=1., Tp=8.)
            d.reset()
            d.reset(jonswap_params={'Hs': 2.})
            self.assertEqual(d.disturbance_model, {'preset': 'std', **flags})
            s = d.step(state(), np.zeros(6))
            self.assertTrue(np.isfinite([s[k] for k in KEYS]).all())

    def test_invalid_flags_fail_instead_of_silently_changing_ablation(self):
        for flags in [{'preset': 'typo'}, {'normalise_spectrum': True}, {'wave_filter': 'false'}]:
            with self.assertRaises(ValueError):
                model(flags)
        for waves in [{'Hs': -1}, {'Tp': 0}, {'gamma': 0}, {'N': 0}]:
            with self.assertRaises(ValueError):
                model({'preset': 'corrected'}, **waves)

    def test_env_metadata_and_reset_overrides(self):
        from bluerov2_gym.workflows.evaluation.env_utils import make_env
        env = make_env(dynamics_config={'disturbance_model': {'preset': 'corrected'}},
                       jonswap_params={'Hs': 0, 'noise_std': 0})
        try:
            _, info = env.reset()
            self.assertTrue(info['disturbance_model']['normalize_spectrum'])
            _, _, _, _, info = env.step(np.zeros(6))
            self.assertFalse(info['disturbance_model']['wave_filter'])
            np.testing.assert_array_equal(info['nu_current'], np.zeros(3))
            _, info = env.reset(options={'jonswap_params': {'Hs': 1.}})
            self.assertTrue(info['disturbance_model']['body_frame'])
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
