# Critical literature review: learning-based underwater trajectory tracking under wave disturbances

## Scope and access note

The supplied “current introduction” field contains a placeholder, so these additions cannot be checked for overlap. This is a focused evidence review, not an exhaustive systematic review. Full texts were available for selected wave-control, six-DOF RL, and robust-PPO studies; several other entries rely on abstracts. Peer-review status is not assumed for preprints, and the retrieved metadata did not establish the venue of the 2026 curriculum record.

## A. Evidence table

See the companion source-linked table, **Underwater control literature evidence table**. It records task type, disturbance representation, learning/transfer strategy, metrics, and validation status for the selected studies. Important distinctions are preserved: station keeping regulates a fixed pose; trajectory tracking follows a time-indexed moving reference; path following seeks proximity to a geometric curve without necessarily matching a prescribed timing. One source explicitly separates path generation, path following, and autopilot, illustrating why these labels should not be treated as synonyms. [^1]

## B. Introduction-ready evidence statements

1. **Waves are not interchangeable with a generic disturbance input or a quasi-steady current.** Walker and Giorgio-Serchi distinguish calmer-water MPC and quasi-constant current models from explicitly modeled wave disturbances, and argue that wave prediction can provide preview information for control. Their trajectory-tracking architecture propagates spectral information from an upstream measurement to estimate impending loads; it is a three-DOF simulation study rather than a six-DOF hardware demonstration. [^2]

2. **Published underwater RL disturbance tests span substantially different physical abstractions.** For example, robust PPO navigation models swirls, constant fields, and horizontal/vertical current fields, with adversarial disturbance generation; this is not the same disturbance class as a sea-state spectrum mapped through finite-depth wave kinematics to time-varying vehicle loads. [^3] Thus, “tested under disturbances” should be qualified by disturbance type, temporal structure, and how the disturbance enters the dynamics.

3. **Curriculum learning is relevant, but the directly matching evidence retrieved is not yet a clean precedent for this control problem.** A 2026 abstract describes adaptive curriculum for AUV path planning/data collection, increasing wave amplitude and turbulence through an adaptive success-rate rule, and reports simulation-based gains. Its task is planning/data collection rather than low-level trajectory tracking; full text and publication venue were not available here. [^4] Separately, a 2020 transfer-learning study reports training a control policy over disturbance waveforms and adding a disturbance-identification/transition-mismatch compensation module for pose regulation in simulation; this supports progressive or multi-condition exposure as a plausible training design, but does not establish that a staged disturbance ramp improves PPO tracking. [^5]

4. **Transfer across dynamics has underwater precedents, but the approaches and claims differ.** Dynamics-aligned transfer RL transfers a policy between source and target models using a distance metric and tests numerical pose-regulation scenarios that include altered thruster characteristics. [^6] Another general-motion framework trains against a virtual model containing known terms, then compensates the learned policy at deployment using extended-state observers for unknown terms. [^7] These studies motivate explicit reporting of the simplified-to-target model mismatch and the transfer protocol; they do not demonstrate that a specific mechanics curriculum causes improved performance in a PPO-based, wave-disturbed moving-reference task.

5. **Wave-aware MPC offers a strong comparator rationale but mostly addresses different objectives and levels of validation.** The Walker–Giorgio-Serchi trajectory-tracking NMPC uses deterministic sea-wave prediction, compares with feedback and feed-forward controllers, and reports position/attitude RMSE and maximum error across three simulated sea states. [^2] The related 2024 Walker, Jordan, and Giorgio-Serchi study is explicitly station keeping: it experimentally validates wave prediction, then simulates NMPC performance under wave conditions. [^8] Walker et al.’s earlier study likewise experimentally validates wave-induced force estimates for a BlueROV2-type ROV, including irregular JONSWAP cases, but evaluates the MPC-versus-PD station-keeping controller comparison in simulation. [^9] These precedents support distinguishing wave-model validation from closed-loop experimental controller validation.

6. **RMSE alone does not express whether a controller remains acceptably close for most of a mission.** In the wave-tracking NMPC study, RMSE is accompanied by maximum error; in the geometric path-following literature, path performance is treated as a distinct geometric assessment problem. [^2][^10] The proposed percentage of elapsed time inside a prescribed distance can therefore be framed as a mission-level tolerance-occupancy measure complementary to RMSE, not as a metric already established by the cited sources. For trajectory tracking it should be calculated against the moving reference at the same time index; for path following it would instead use distance to the nearest point on the geometric path. That distinction prevents a path-proximity score from masking timing lag.

7. **Simulation-to-real RL demonstrations exist, but they do not answer robustness to stochastic surface waves.** The 2024 “Learning to Swim” study reports a six-DOF thruster-driven RL controller, domain randomization over physical parameters, and zero-shot deployment on a real AUV, with outcomes comparable to hand-tuned PID. [^11] Its reported robustness is not equivalent to testing under controlled JONSWAP wave forcing, and does not validate the present simulation-only study experimentally.

## C. Carefully bounded research gap

**Documented findings:** Underwater RL studies include disturbance-robust navigation against analytic current/swirl/constant fields; transfer-learning studies address dynamics mismatch or disturbance waveforms in pose regulation; recent curriculum work addresses AUV planning/data collection in increasingly complex wave/turbulence environments; and wave-preview MPC studies address tracking or station keeping using forecast wave loads. [^3][^5][^4][^2][^8]

**Inference from this search, not a claim of absence:** The retrieved evidence does not establish a single prior evaluation combining (i) PPO generalized-force/moment commands through a fixed thruster allocator, (ii) staged transfer from simplified to target six-degree-of-freedom mechanics, (iii) progressively intensified energy-normalized JONSWAP forcing with finite-depth Airy kinematics, and (iv) matched comparisons against NMPC, PID, and SMC on multiple timed trajectories using both error and time-within-tolerance outcomes. Adjacent studies cover subsets of these elements, including adaptive curricula in AUV path planning, wave-disturbance trajectory tracking, disturbance-preview NMPC, and transfer across underwater vehicle dynamics. [^4][^12][^2][^6] This combination may define a useful, tightly scoped comparison question, but the search does not warrant an originality claim: abstract-only and inaccessible records limit coverage, and an expanded search may identify closer combinations. The paper should state its contribution as the specific controlled evaluation it performs, not as the first use of PPO, curriculum learning, transfer, waves, or underwater tracking. The simulation-only design also cannot substantiate experimental robustness or a causal benefit of curriculum learning unless curriculum is isolated through an appropriate ablation or matched control.

## D. Two proposed paragraphs in scientific English

Underwater learning-based control has been evaluated against disturbance models ranging from analytic current and swirl fields to wave-related forcing, but these cases represent different physical challenges and should not be grouped under an undifferentiated label of environmental disturbance. Robust PPO has, for example, been studied in navigation tasks with constant fields, currents, and swirls, whereas wave-preview model predictive control reconstructs spectral wave information to estimate time-varying hydrodynamic loads along a prediction horizon. [^3][^2] Related wave-control studies provide useful comparators, although their tasks include three-degree-of-freedom trajectory tracking or fixed-pose station keeping, and experimental validation of wave-load prediction should be distinguished from experimental validation of closed-loop tracking performance. [^2][^9]

Curriculum and transfer methods provide a rationale for exposing an underwater controller to progressively more demanding conditions, while existing studies also show that transfer across vehicle models can require explicit treatment of transition mismatch or model uncertainty. [^4][^5][^6] In a simulation study of timed trajectory tracking, reporting the fraction of mission time spent within a specified distance of the instantaneous reference can complement position RMSE by quantifying sustained tolerance attainment; maximum error and yaw error can further characterize excursions and orientation performance. This is an operationally interpretable metric proposal rather than an established standard identified in the reviewed sources, and it should not be conflated with geometric path-following error, which does not necessarily penalize timing deviation. [^2][^10]

## Interpretation and limitations

The reviewed studies distinguish model-transfer approaches, disturbance-generation assumptions, controller objectives, and outcome measures: underwater dynamics-transfer papers use pose regulation and source-to-target model changes, while wave-preview papers study either reduced-DOF trajectory tracking or station keeping. [^6][^2][^8] A causal statement that curriculum learning improved the reported controller requires a comparison with training without the curriculum under otherwise matched conditions; simulated wave robustness also does not demonstrate experimental robustness. Exact model fidelity, thruster allocation, filtering, and disturbance normalization should therefore be described as study-specific implementation choices rather than implied replications of prior work.

[^1]: Rober & Cichella, 2020. Geometric Path Following of Underwater Vehicles. AIAA Scitech 2021 Forum.

[^2]: Walker & Giorgio-Serchi, 2023. Disturbance Preview for Non-Linear Model Predictive Trajectory Tracking of Underwater Vehicles in Wave Dominated Environments. IEEE/RJS International Conference on Intelligent RObots and Systems.

[^3]: Parras & Zazo, 2021. Robust Deep Reinforcement Learning for Underwater Navigation with Unknown Disturbances. IEEE International Conference on Acoustics, Speech, and Signal Processing.

[^4]: Li et al., 2026. Adaptive curriculum learning-based path planning and data collection for autonomous underwater vehicles under extreme sea conditions. Ocean Engineering.

[^5]: Wang et al., 2020. Modular transfer learning with transition mismatch compensation for excessive disturbance rejection. International Journal of Machine Learning and Cybernetics.

[^6]: Cheng et al., 2022. Dynamics-Aligned Transfer Reinforcement Learning For Autonomous Underwater Vehicle Control. International Conference on Advanced Robotics and Mechatronics.

[^7]: Xu et al., 2023. A general motion control framework for an autonomous underwater vehicle through deep reinforcement learning and disturbance observers. Journal of the Franklin Institute.

[^8]: Walker et al., 2024. Nonlinear model predictive dynamic positioning of a remotely operated vehicle with wave disturbance preview. Int. J. Robotics Res.

[^9]: Walker et al., 2021. Experimental Validation of Wave Induced Disturbances for Predictive Station Keeping of a Remotely Operated Vehicle. IEEE Robotics and Automation Letters.

[^10]: Saggini & Torrente, 2016. An Euclidean norm based criterion to assess robots' 2D path-following performance.

[^11]: Cai et al., 2024. Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-Driven Autonomous Underwater Vehicles. IEEE International Conference on Robotics and Automation.

[^12]: Zhang & Zhang, 2024. Reinforcement Learning-based AUV Trajectory Tracking and Obstacle Avoidance Control in Wave Disturbance Environments. 2024 IEEE International Conference on Unmanned Systems (ICUS).