# Diagnosis of the stopped Walker PPO curriculum

Training run: artifacts/training/physics_wave_curriculum_walker. Interrupted by user at 4,239,360 PPO steps, during phase B (std W1). No training restarted and no policy/reward/mechanics changed during this audit.

## Recorded learning

| Assessment | PPO steps | Mean post-settling position RMSE (m) | Episode success | Completed | Mean action saturation fraction |
|---|---:|---:|---:|---:|---:|
| Last A | 1,075,200 | 0.0954 | 0.95 | 20/20 | 0.0000 |
| First B | 1,085,440 | 1.1807 | 0.00 | 20/20 | 0.0000 |
| Middle B | 2,662,400 | 0.7241 | 0.00 | 20/20 | 0.0000 |
| Last complete B assessment | 4,229,120 | 0.5718 | 0.00 | 20/20 | 0.0192 |

These records show initial learning and adaptation, followed by failure to reach the inherited 0.2 m criterion. The agent did not fail phase A. Assessments are stochastic wave/reset batches, not a controlled learning-rate ablation.

## Wave-input diagnostic

Dynamics._jonswap_disturbance sampled at dt=0.1 s for 100 s, seed 42, ignoring the first 10 s; fixed level/yaw-zero orientation, so no orientation-dependent body rotation is applied in this generator-only diagnostic. RMS is sqrt(mean(sum(component^2))). The diagnostic does not include relative-velocity hydrodynamic loading or thrust.

| Configuration | Coupling-speed norm RMS (m/s) | Empirical direct-force norm RMS (N) |
|---|---:|---:|
| Article preparation B | 0.2032 | 1.4409 |
| New preparation B, W1 | 0.5204 | 11.9288 |
| Same new B with only wave_filter enabled | 0.1091 | 0.5409 |

Old versus new B differs in spectrum, period, depth, direction sampling, scaling/caps and filters; it is not an isolated filter comparison. The last two rows hold the new W1 wave configuration fixed and differ only in the wave filter. This demonstrates a major change in the input distribution, not that the unfiltered forcing is physically impossible or that a filter is necessary for learning.

## Other implementation factors

- A jumps directly from no waves to full W1, Hs=2.78 m, in B. W1 is not a calm or near-zero-amplitude stage. The absence of intermediate amplitudes is a plausible curriculum-design cause, not a causally isolated conclusion.
- B uses std mechanics and HISTORICAL_DAMPING=[33.8,54.26875,73.37135,40,40,40]. Target quadratic rotational damping is [0.1246,0.1246,0.1857]. In particular yaw quadratic damping is 40 in std versus 0.1857 in target. A simplified model is not necessarily easier for heading dynamics.
- Historical feedback/reward retain u_body - xdot_ref_world and an incomplete linear-velocity-error observation (only u replaced). This convention already existed in the successful policy, so it is not evidence of a new regression by itself; it remains a limitation worth isolating under stronger motion/heading errors.
- The intermediate evaluation used target mechanics, whereas the policy had only reached std W1 training. Its RMSE/yaw are not final-curriculum performance.
- Allocation rank is five. Pure yaw is available: B@[1,-1,1,-1,0,0] produces only tau_z=0.683065 Nm. Missing independent pitch is a physical limitation but does not establish a yaw-actuation deficiency.
- In target W1 evaluation, new-policy roll/pitch RMS were 1.78/4.09 degrees, versus NMPC 2.01/4.10 degrees. New-policy action saturation was 1.14%; this does not suggest persistent global actuator saturation, and similar mean pitch with better NMPC tracking does not support pitch alone as the explanation. Short peaks or actuator allocation limitations are not excluded by these averages.

## Proposed next controlled training design

Retain PPO architecture and hyperparameters. Avoid changing multiple learning choices at once: use target mechanics without waves first, then a gradual Hs bridge to W1 at fixed period/depth/kinematics, then introduce W2/W3 and verify each separately. Treat observation-frame corrections and reward/yaw reweighting as explicit, separate ablations. Do not relax the tracking threshold just to label the stage mastered. No guarantee of convergence follows from this proposal; validate on paired runs before attributing causality.
