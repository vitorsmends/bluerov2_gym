# Walker references versus the current wave generator

Reviewed 2026-11-13. Research and code inspection only; active training and evaluation defaults unchanged.

## Published parameter families

Walker et al., *Experimental Validation of Wave Induced Disturbances for Predictive Station Keeping of a Remotely Operated Vehicle* (RA-L, 2021), Table II and Section III:
https://www.pure.ed.ac.uk/ws/portalfiles/portal/214232565/RA_L_Experimental_Validation_of_Wave_Induced_Disturbances_for_Predictive_Station_Keeping_of_an_ROV.pdf

| Case | Hs (m) | Peak frequency (Hz) | Tp (s) | Basin depth (m) | ROV depth below still water (m) |
|---|---:|---:|---:|---:|---:|
| JS01 | 0.2 | 0.5 | 2 | 2 | 1 |
| JS02 | 0.4 | 0.5 | 2 | 2 | 1 |
| JS03 | 0.6 | 0.5 | 2 | 2 | 1 |

These are restrained-ROV load-validation experiments, not equivalent controller tracking benchmarks. Do not confuse regular-wave cases R01--R03 with these irregular JONSWAP cases. Gamma and directional spread must be specified separately when constructing an approximate analogue; they are not supplied by this table. The model includes fluid-acceleration/inertia terms and distributed pitching load in addition to drag, and experimental sensor data were smoothed; this is different from filtering the plant's wave-velocity forcing.

Walker, Jordan and Giorgio-Serchi, *Nonlinear model predictive dynamic positioning of a remotely operated vehicle with wave disturbance preview*, DOI 10.1177/02783649241286909, Table 2 / Section 5.1 (online 2024):
https://journals.sagepub.com/doi/pdf/10.1177/02783649241286909

| Case | Hs (m) | Tp (s) | Water depth (m) | ROV depth (m) |
|---|---:|---:|---:|---:|
| W1 | 2.78 | 7.1 | 54 | 5 |
| W2 | 3.47 | 9.5 | 54 | 5 |
| W3 | 3.24 | 11.1 | 54 | 5 |

These cases use buoy-derived spectra and station keeping with a BlueROV2 Heavy; Hs/Tp alone do not reconstruct their exact spectra. NMPC with preview and a blind NMPC baseline are both considered. The IROS 2023 tracking paper uses the same W1--W3 summaries but a moving-depth square mission, so its depth must not be reduced to one fixed value:
https://arxiv.org/html/2307.14834v1

Public implementation:
https://github.com/kyleliamwalker/dswp_mpc
https://raw.githubusercontent.com/kyleliamwalker/dswp_mpc/main/launchpads/dswp_mpc_simulation.m
The code evaluates particles at the current ROV position and filters noisy wave measurements before prediction. This measurement filter does not establish that plant wave velocities should be low-pass filtered.

## Current local configuration

Quick unfiltered Moderate test: Hs=2.5 m, Tp=8 s, water_depth=30 m, fixed sampling point (0,0,-2) m, scale=0.5, velocity norm cap=0.5 m/s; wave and noise filters disabled and noise_std=0. The body-frame rotation remains disabled, and a separate empirical quadratic wave force is added to relative-velocity hydrodynamics.

Relevant implementation: bluerov2_gym/envs/core/dynamics.py, _jonswap_disturbance, _compute_wave_force and step. Sampling uses x_eval/y_eval/z_eval, not current vehicle position. Therefore there is no instantaneous spatial wave “hot spot” caused by moving along the current reference; body motion still affects relative-velocity loads and controller response. Reference depth and disturbance sampling depth are different quantities in this implementation.

The generated force cannot be compared through Hs alone: spectrum, frequency band, depth kernel, spatial sampling, scaling/caps, frame transformation, added-mass treatment, empirical gains and actuator authority all matter. Relative-velocity damping plus an empirical force is not identical to Walker's experimentally checked loading model; overlapping contributions should be audited before asserting physical load equivalence or double counting.

## Proposed controlled comparison

1. Use JS01--JS03 as a laboratory-scale reference family for generator/load characterization; match d=2 m and z_eval=-1 m. Keeping the current z_eval=-2 m would place sampling on the tank bottom and is not the published experiment.
2. Use W1--W3 as an energetic offshore reference family, matching d=54 m and z_eval=-5 m for a fixed-depth diagnostic. Label analytic JONSWAP reconstruction from Hs/Tp as an analogue rather than exact replication of buoy spectra.
3. Before interpreting controller rankings, compare raw and processed velocity RMS/peaks, force RMS/peaks, cap occupancy, and demanded versus available thrust. Separate a prescribed fixed-point disturbance experiment from a moving-vehicle wave-field model.
4. Keep physical plant, policy, reference, seed and evaluation horizon fixed within each ablation. Do not change an ongoing training distribution silently. Filters-off alone does not certify a physically validated ocean disturbance model.
