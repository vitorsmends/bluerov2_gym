"""Read-only algebraic audit against the local reference Xacros (no Gazebo run).

Run from the repository root: python scripts/validate_gazebo_dynamics.py
The report prints differences; it does not change model parameters.
"""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bluerov2_gym.envs.core.config_utils import load_yaml
from bluerov2_gym.envs.core.dynamics import Dynamics


def rotation(angles):
    phi, theta, psi = angles
    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)
    cy, sy = np.cos(psi), np.sin(psi)
    return np.array([[cy*ct, cy*st*sp-sy*cp, cy*st*cp+sy*sp],
                     [sy*ct, sy*st*sp+cy*cp, sy*st*cp-cy*sp],
                     [-st, ct*sp, ct*cp]])


def coriolis_force(mass, velocity):
    momentum = mass @ velocity
    v, omega = velocity[:3], velocity[3:]
    return np.r_[np.cross(omega, momentum[:3]),
                 np.cross(v, momentum[:3]) + np.cross(omega, momentum[3:])]


def main():
    np.set_printoptions(precision=9, suppress=True)
    ref = ROOT / '_references/bluerov2/bluerov2_description/urdf'
    ns = '{http://www.ros.org/wiki/xacro}'
    base = ET.parse(ref / 'bluerov2.xacro').getroot()
    props = {e.attrib['name']: e.attrib['value'] for e in base.findall(ns+'property')}
    m_base, m_prop = float(props['mass']), float(props['prop_mass'])
    sizes = np.array([float(props[k+'_size']) for k in 'xyz'])
    rho = float(next(e for e in base.findall(ns+'arg') if e.attrib['name']=='density').attrib['default'])
    # Explicit formula used in this reference Xacro; no arbitrary expression eval.
    correction = 1.01 * ((m_base + 6*m_prop)/(rho*np.prod(sizes)))**(1/3)
    rg = np.array([0., 0., -correction*sizes[2]])
    rb = rg/4
    volume = np.prod(sizes*correction)
    inertial = base.find('link/inertial/inertia').attrib
    tensor = np.array([
        [float(inertial['ixx']), float(inertial['ixy']), float(inertial['ixz'])],
        [float(inertial['ixy']), float(inertial['iyy']), float(inertial['iyz'])],
        [float(inertial['ixz']), float(inertial['iyz']), float(inertial['izz'])],
    ])
    d = Dynamics(dynamics_config=load_yaml(ROOT/'config/dynamics_config.yaml'))
    hydro = ET.parse(ref/'hydrodynamics.xacro').getroot().find('gazebo/plugin')
    for label, tags, actual in [
        ('added_mass', ['xDotU','yDotV','zDotW','kDotP','mDotQ','nDotR'], d.added_mass),
        ('linear_damping', ['xU','yV','zW','kP','mQ','nR'], d.D_lin),
        ('quadratic_damping', ['xUabsU','yVabsV','zWabsW','kPabsP','mQabsQ','nRabsR'], d.D_quad),
    ]:
        expected = -np.array([float(hydro.find(tag).text) for tag in tags])
        print(label, 'MATCH' if np.array_equal(expected, actual) else 'DIFF', actual)
    tree = ET.parse(ref/'thrusters.xacro').getroot()
    rotor_props = {e.attrib['name']: e.attrib['value'] for e in tree.findall(ns+'property')}
    length, radius = float(rotor_props['prop_l']), float(rotor_props['prop_r'])
    columns, rotor_positions, rotations = [], [], []
    for thruster in tree.findall(ns+'thruster'):
        i = int(thruster.attrib['thruster_id'])
        origin = thruster.find('origin')
        r = np.fromstring(origin.attrib['xyz'], sep=' ')
        rot = rotation(np.fromstring(origin.attrib['rpy'], sep=' '))
        # Macro explicitly uses +X for IDs 1,2,5; -X for IDs 3,4,6.
        direction = rot @ np.array([1 if i in [1,2,5] else -1, 0, 0])
        columns.append(np.r_[direction, np.cross(r, direction)])
        rotor_positions.append(r+rot@np.array([length/2,0,0]))
        rotations.append(rot)
    allocation = np.column_stack(columns)
    print('URDF allocation:\n', allocation)
    mapping = np.array([-1,-1,-1,-1,1,1])
    print('allocation max error after horizontal command sign mapping:',
          np.max(np.abs(allocation*mapping-d.allocation_matrix)))
    print('base+rotor mass:', m_base+6*m_prop, 'environment:', d.m)
    print('reference volume:', volume, 'environment:', d.volume, 'centers:', rg, rb)
    camera = ET.parse(ref/'sensors.xacro').getroot()
    camera_link = next(e for e in camera.findall('link') if e.attrib['name']=='${ns}/camera')
    m_camera = float(camera_link.find('inertial/mass').attrib['value'])
    x_camera = float(next(e for e in camera.findall(ns+'property') if e.attrib['name']=='x_camera').attrib['value'])
    rc = np.array([x_camera+.15, 0., 0.])
    total_mass = m_base+6*m_prop+m_camera
    cg = (m_base*rg+m_prop*np.sum(rotor_positions,axis=0)+m_camera*rc)/total_mass
    inertia = tensor-m_base*d._skew(rg)@d._skew(rg)
    rotor_tensor = np.diag([m_prop*radius**2/2,
        m_prop*radius**2/4+m_prop*length**2/12,
        m_prop*radius**2/4+m_prop*length**2/12])
    for rot, r in zip(rotations, rotor_positions):
        inertia += rot@rotor_tensor@rot.T-m_prop*d._skew(r)@d._skew(r)
    camera_diagonal = np.array([float(camera_link.find('inertial/inertia').attrib['i'+a+a]) for a in 'xyz'])
    inertia += np.diag(camera_diagonal)-m_camera*d._skew(rc)@d._skew(rc)
    print('locked assembly mass/CG:', total_mass, cg)
    print('locked assembly inertia at base origin:\n', inertia)
    print('environment inertia at base origin:\n', d.M_RB[3:,3:])
    expected_vertical = d.B_force-d.W
    print('net Z-up buoyancy for environment mass:', expected_vertical,
          'current model:', -d._restoring_forces(0.,0.)[2])
    print('volume buoyancy ratio vs base+rotors weight:', d.B_force/d.W)
    for velocity in [np.array([1.,1.,0.,0.,0.,0.]), np.array([1.,.5,-.2,.1,-.2,.3])]:
        print('velocity:', velocity)
        for name, matrix, method in [('RB', d.M_RB, d._rigid_body_coriolis_force),
                                      ('MA', d.M_A, d._added_mass_coriolis_force)]:
            actual, expected = method(velocity), coriolis_force(matrix,velocity)
            print(name, 'current:', actual, 'expected:', expected)
            print(name, 'current/expected power:', velocity@actual, velocity@expected)
    print('default quadratic damping without YAML:', Dynamics().D_quad)
    print('Algebraic comparison only; no Gazebo simulation was run.')


if __name__ == '__main__':
    main()
