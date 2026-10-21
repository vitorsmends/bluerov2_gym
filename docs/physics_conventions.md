# Physical model v2

`Dynamics.MODEL_VERSION = zup-airy-coriolis-v2` identifies the model introduced
after the equation audit. Controller design scripts import this same plant;
NMPC has an equivalent symbolic predictor checked against the numerical step.

## Coordinates and coefficients

Both inertial and body frames are right-handed, Z upward. In the body frame,
X is forward and Y is left. Submerged targets have negative world Z. The
rotation is Rz(yaw) Ry(pitch) Rx(roll). Angular rates are body-frame rates.
Trajectory reference translational velocities are inertial; controllers and
velocity-error metrics transform them before comparing with body velocities.
The fixed PPO observation retains its legacy yaw-only position transform and
only modifies the surge channel, but reference surge is now transformed into
the current body frame before subtraction.

`vehicle.inertia.reference` explicitly specifies `center_of_gravity` (default)
or `body_origin`. The supplied YAML declares the former, consistent with the
article's original description. It is a modeling assumption for the existing
coefficients, not an independent measurement of their provenance. The rigid-body
matrix uses I_O=I_G-m S(r_g)² and the corresponding off-diagonal mass blocks.
If coefficients known to be measured about the origin become available, use
`body_origin` to avoid applying the parallel-axis shift twice.

Hydrostatics are computed from gravity [0,0,-W] and buoyancy [0,0,B] in the
inertial frame, transformed to the body and applied at r_g and r_b. The restoring
vector is minus their wrench because it is on the left of the dynamics.
The signed center coordinates are used; an absolute center offset is not
substituted. With the default CG below CB, passive roll/pitch torques oppose tilt.

For momenta [P;H]=M[v;omega], Coriolis forces are
[omega×P; v×P+omega×H]. Their instantaneous power is zero for matching velocity
and momentum, independently verified for rigid-body and added-mass terms.

## Waves

The JONSWAP shape follows Fossen MSS case 7 with analytical DNV correction;
Hs controls elevation energy. There is no numerical energy normalization.
Airy particle velocities use A omega / sinh(k d), consistent with finite-depth
dispersion, zero normal velocity at the bottom and w=d(eta)/dt at the surface.

The field is sampled at the configured fixed x_eval/y_eval/z_eval. Spectral
velocities are scaled and norm-limited; filtered noise is then added, and a
second low-pass filter is applied. These operations are inertial-frame
operations. The filtered velocity is rotated into current body coordinates
before evaluating relative drag and empirical direct wave loads. Wave-force
gains and the application point are body-fixed. `nu_current` is body-frame;
`nu_current_world` and the raw wave-velocity diagnostics are inertial-frame.

The load K v|v| with force/moment norm limits remains an empirical additional
load, not a full Morison/radiation/diffraction calculation. Fluid-acceleration
forcing is not explicitly modeled. Relative-velocity damping plus this load is
a declared simulation approximation, not an exact physical decomposition.
NMPC uses a zero-disturbance forecast, as before; it is not given future waves.

## Data and execution

Existing policies are not retrained automatically. New inference is evaluated
in the corrected plant; old results in `results-jonswap` belong to the earlier
model and must not be relabeled as v2.

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python run_oceans.py --results-root results-physics-v2
```

Each new campaign stores its physics version, physical configuration, scenario
configuration, and the selected policy/normalizer hashes in its manifest.
