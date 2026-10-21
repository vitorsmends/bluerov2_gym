import unittest
import numpy as np
import casadi as ca
from bluerov2_gym.envs.core.dynamics import Dynamics
from experiments.controllers.nmpc import NMPCController
from experiments.env_utils import reference_velocity_error_body


class PhysicsTests(unittest.TestCase):
    def test_coriolis_conserves_energy_and_matches_momentum_form(self):
        d = Dynamics()
        rng = np.random.default_rng(12)
        for _ in range(30):
            nu = rng.normal(size=6)
            for mass, force in [(d.M_RB, d._rigid_body_coriolis_force),
                                (d.M_A, d._added_mass_coriolis_force)]:
                p, h = np.split(mass @ nu, 2)
                expected = np.r_[np.cross(nu[3:], p),
                                 np.cross(nu[:3], p)+np.cross(nu[3:], h)]
                np.testing.assert_allclose(force(nu), expected, atol=1e-12)
                self.assertAlmostEqual(float(nu @ force(nu)), 0, places=10)

    def test_parallel_axis_matches_kinetic_energy(self):
        d = Dynamics()
        nu = np.array([.4, -.2, .1, .1, -.15, .2])
        vcg = nu[:3] + np.cross(nu[3:], d.r_g)
        expected = .5*d.m*(vcg@vcg) + .5*nu[3:]@d.I_g@nu[3:]
        self.assertAlmostEqual(.5*nu@d.M_RB@nu, expected)
        self.assertTrue(np.all(np.linalg.eigvalsh(d.M_RB) > 0))
        d2 = Dynamics(dynamics_config={'vehicle': {'inertia': {'reference': 'body_origin'}}})
        np.testing.assert_allclose(d2.I_O, [[d2.Ixx,d2.Ixy,d2.Ixz],[d2.Ixy,d2.Iyy,d2.Iyz],[d2.Ixz,d2.Iyz,d2.Izz]])

    def test_restoring_opposes_tilt_and_buoyancy_points_up(self):
        d = Dynamics()
        self.assertGreater(-d._restoring_forces(0, 0)[2], 0)
        for angle in [-.2, .2]:
            self.assertLess(-d._restoring_forces(angle, 0)[3]*angle, 0)
            self.assertLess(-d._restoring_forces(0, angle)[4]*angle, 0)
        # General (off-axis) centers: verify directly against rotated wrenches.
        d = Dynamics(dynamics_config={'buoyancy': {'center_of_gravity': [.02,-.01,-.1],
                                                   'center_of_buoyancy': [-.01,.03,-.02]}})
        eta = np.array([0,0,0,.2,-.3,.7])
        R = d.body_to_world_rotation(eta)
        fg = R.T @ np.array([0,0,-d.W]);fb=R.T @ np.array([0,0,d.B_force])
        np.testing.assert_allclose(d._restoring_forces(eta[3],eta[4]),
                                  -np.r_[fg+fb,np.cross(d.r_g,fg)+np.cross(d.r_b,fb)])

    def single_wave(self, z):
        d = Dynamics(jonswap_params={'Hs':1,'z_eval':z,'noise_std':0,
                                     'scale':1,'max_current':100,'alpha_wave':1})
        omega=.8;k=d._solve_wave_number(omega,d.water_depth)
        d._waves=[dict(A=.2,omega=omega,k=k,phase=.9,theta=0)]
        return d

    def test_airy_surface_and_bottom_boundary_conditions(self):
        d=self.single_wave(0)
        velocity,_=d._jonswap_disturbance()
        phase=.9-.8*d.dt
        self.assertAlmostEqual(velocity[2],.2*.8*np.sin(phase))
        self.assertAlmostEqual(velocity[0],.2*.8/np.tanh(d._waves[0]['k']*d.water_depth)*np.cos(phase))
        d=self.single_wave(-30)
        self.assertAlmostEqual(d._jonswap_disturbance()[0][2],0)

    def test_current_rotation_and_reference_velocity_frames(self):
        d=self.single_wave(-2);e=self.single_wave(-2)
        eta=np.array([0,0,0,.2,-.3,1.1]);R=d.body_to_world_rotation(eta)
        world,_=d._jonswap_disturbance();body,_=e._jonswap_disturbance(eta)
        np.testing.assert_allclose(R@body,world)
        ref=np.zeros(12);ref[6:9]=[.4,-.2,.1]
        state=np.r_[eta,R.T@ref[6:9],np.zeros(3)]
        np.testing.assert_allclose(reference_velocity_error_body(state,ref),0,atol=1e-15)

    def test_casadi_and_plant_steps_agree(self):
        d = Dynamics(jonswap_params={'Hs':0,'noise_std':0})
        # Test the symbolic predictor without building or running an optimizer.
        n = NMPCController.__new__(NMPCController);n.dyn=d;n.dt=d.dt
        x=ca.MX.sym('x',12);u=ca.MX.sym('u',6);dw=ca.MX.sym('dw',9)
        f=ca.Function('step',[x,u,dw],[n._casadi_dynamics_step(x,u,dw)])
        rng=np.random.default_rng(24)
        keys=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
        for _ in range(10):
            initial=rng.uniform(-.3,.3,12);control=rng.uniform(-2,2,6)
            state=dict(zip(keys,initial));d.reset();d.step(state,control)
            expected=np.array([state[k] for k in keys])
            np.testing.assert_allclose(np.array(f(initial,control,np.zeros(9))).ravel(),expected,atol=1e-11)
        # The same predictor must agree with a nonzero instantaneous disturbance.
        d.set_jonswap_params(Hs=2.5, noise_std=.01, alpha_wave=.2)
        for _ in range(10):
            initial=rng.uniform(-.3,.3,12);control=rng.uniform(-2,2,6)
            state=dict(zip(keys,initial));d.step(state,control)
            disturbance=np.r_[state['nu_current'],state['tau_wave']]
            expected=np.array([state[k] for k in keys])
            np.testing.assert_allclose(np.array(f(initial,control,disturbance)).ravel(),expected,atol=1e-11)


if __name__=='__main__':
    unittest.main()
