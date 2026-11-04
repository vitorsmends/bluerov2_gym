"""Physical invariants for the Z-up rigid-body model."""
import unittest

import numpy as np

from bluerov2_gym.envs.core.dynamics import Dynamics
from bluerov2_gym.envs.core.config_utils import load_yaml


class MechanicsTests(unittest.TestCase):
    def test_parallel_axis_and_positive_kinetic_energy(self):
        d = Dynamics()
        self.assertEqual(d.mechanics_model, 'corrected')
        np.testing.assert_allclose(d.M_RB, d.M_RB.T)
        self.assertGreater(np.linalg.eigvalsh(d.M_RB).min(), 0)
        rng = np.random.default_rng(10)
        for _ in range(50):
            nu = rng.normal(size=6)
            cg_velocity = nu[:3] + np.cross(nu[3:], d.r_g)
            expected = .5*d.m*(cg_velocity@cg_velocity) + .5*nu[3:]@d.I_g@nu[3:]
            self.assertAlmostEqual(.5*nu@d.M_RB@nu, expected, places=11)

    def test_coriolis_conserves_energy(self):
        d = Dynamics()
        rng = np.random.default_rng(123)
        for _ in range(100):
            nu = rng.normal(size=6)
            self.assertAlmostEqual(nu@d._rigid_body_coriolis_force(nu), 0., places=10)
            self.assertAlmostEqual(nu@d._added_mass_coriolis_force(nu), 0., places=10)

    def test_translational_added_mass_generates_yaw_coupling(self):
        d = Dynamics()
        nu = np.array([1., 1., 0., 0., 0., 0.])
        np.testing.assert_allclose(d._rigid_body_coriolis_force(nu), np.zeros(6), atol=1e-14)
        np.testing.assert_allclose(d._added_mass_coriolis_force(nu), [0,0,0,0,0,7.2])

    def test_hydrostatics_with_arbitrary_centers(self):
        d = Dynamics(dynamics_config={'buoyancy': {
            'center_of_gravity': [.02,-.01,-.12],
            'center_of_buoyancy': [-.01,.03,-.02]}})
        for phi,theta,psi in [(0.,0.,0.),(.3,-.5,.8),(-.2,.4,-1.)]:
            eta = np.array([0,0,0,phi,theta,psi])
            rot = np.column_stack([d._body_to_world_kinematics(eta,np.eye(6)[i])[:3]
                                   for i in range(3)])
            g = d._restoring_forces(phi,theta)
            np.testing.assert_allclose(rot@(-g[:3]), [0,0,d.B_force-d.W], atol=1e-12)
            world_moment = (np.cross(rot@d.r_g,[0,0,-d.W]) +
                            np.cross(rot@d.r_b,[0,0,d.B_force]))
            np.testing.assert_allclose(rot@(-g[3:]), world_moment, atol=1e-12)

    def test_positive_buoyancy_and_restoring_signs(self):
        d = Dynamics(jonswap_params={'scale':0.,'noise_std':0.,'enable_wave_force':False})
        s = dict.fromkeys(['x','y','z','roll','pitch','yaw','u','v','w','p','q','r'], 0.)
        s['z'] = -2.
        d.step(s, np.zeros(6))
        self.assertGreater(s['w'], 0.)
        self.assertGreater(s['z'], -2.)
        self.assertLess(-d._restoring_forces(.1,0.)[3], 0.)
        self.assertLess(-d._restoring_forces(0.,.1)[4], 0.)

    def test_default_coefficients_match_yaml(self):
        direct = Dynamics()
        configured = Dynamics(dynamics_config=load_yaml('config/dynamics_config.yaml'))
        for name in ['M_RB','M_A','D_lin','D_quad']:
            np.testing.assert_array_equal(getattr(direct,name), getattr(configured,name))
        nu = np.array([.3,-.2,.1,-.4,.5,-.6])
        drag = direct.D_lin*nu + direct.D_quad*nu*np.abs(nu)
        self.assertGreater(nu@drag,0.)

    def test_legacy_is_explicit_and_independent_from_wave_preset(self):
        d = Dynamics(dynamics_config={'mechanics_model':'std',
                                     'disturbance_model':{'preset':'corrected'}})
        np.testing.assert_array_equal(d.M_RB[3:,3:], d.I_g)
        np.testing.assert_array_equal(d.D_quad, [33.8,54.26875,73.37135,40,40,40])
        self.assertTrue(d.disturbance_model['normalize_spectrum'])
        with self.assertRaises(ValueError):
            Dynamics(dynamics_config={'mechanics_model':'typo'})


if __name__ == '__main__':
    unittest.main()
