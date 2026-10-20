# Reproducing the article's trajectory experiments

The article benchmark is configured independently from station-keeping. It
contains five trajectories, four controllers, the moderate and severe sea
states, and five paired disturbance seeds per case.

Run the PPO curriculum training from the repository root:

```bash
python training/train_curriculum.py --config config/training_config.yaml
```

This deliberately uses the existing training setup and checkpoint naming. Its
curriculum schedule remains the one in `config/training_config.yaml`; the
benchmark changes below do not retrain or replace the PPO policy.

For a reproducibility/robustness campaign, run five independent trainings
instead of interpreting the five benchmark disturbance repetitions as training
replicates:

```bash
python training/train_curriculum.py \
  --config config/training_config.yaml \
  --seeds 42 43 44 45 46
```

Each seed is passed to Stable-Baselines3/PyTorch/Python/NumPy through the PPO
seed setting and to the JONSWAP disturbance generator. The models,
VecNormalize statistics, checkpoints, TensorBoard logs, and effective YAML
configuration are saved separately per seed. The existing single-run command
and origin-stabilization objective remain unchanged.

The article trajectory YAML currently selects one PPO model path. To measure
policy-training variability end to end, evaluate each seed-specific model with
its matching VecNormalize file, keep each evaluation output separate, and
record the policy-training seed alongside the benchmark seed. The training
script does not launch this combined policy-seed/evaluation-seed campaign
automatically; the existing article figures and CSVs have not been regenerated
from five PPO policies.

Predefined training ablations are selected with `--ablation`:
`position_only`, `no_velocity_penalty`, `no_yaw_penalty`,
`no_stability_penalty`, `no_success_bonus`, `no_curriculum`,
`no_direct_wave_force`, `no_current`, `no_noise`, `learning_rate_1e-4`, and
`learning_rate_6e-4`. Train each configuration with the same five seeds and
the same timestep budget before treating the comparison as evidence.
`position_only` is a combined reward ablation that retains the success bonus
while removing velocity, yaw, and roll/pitch terms; the three `no_*_penalty`
variants isolate those terms individually. `no_curriculum` trains directly in
the final configured sea state for the full curriculum budget; the other
environment ablations alter only their named component.

Run the trajectory benchmark after the final PPO model and its `VecNormalize`
statistics have been saved:

```bash
python experiments/run_experiment.py \
  --config experiments/config/article_trajectories.yaml
```

The runner writes one CSV per controller and trajectory under
`results-moderate/trajectory_experiments/` and
`results-severe/trajectory_experiments/`. Each CSV contains the five
repetitions, with `repetition_id`, `seed`, sea-state parameters, and
per-step measurements. The same seed is used for all controllers and
trajectories within a repetition index and sea state, so comparisons are
paired while repetitions use independent wave realizations.

The existing CSVs in these directories are legacy outputs: they contain one
1000-step run per controller/trajectory/sea-state and do not contain
`repetition_id` or `seed`. They therefore do not provide the five-repetition
evidence described by the current campaign configuration. Rerun the benchmark
before using those files to support claims about paired disturbance seeds or
evaluation variability.

Generate the figures referenced by the article from those CSV files:

```bash
python plots/run_all.py
python experiments/plots/plot_trajectory.py \
  --scenario severe --repetition-id 0
```

The first command generates the aggregate figures. The second generates
representative XY and 3D trajectory figures from the selected repetition. Both
write paper figures to `oceans2026/figs/`; summary tables are written to
`article_tables_simple/` and scenario-specific `results-*/tables_trajectory_experiments/`.
RMSE and Pareto statistics are computed per trajectory and repetition;
time-series and energy bands aggregate the individual trajectory/repetition
runs.

The general-purpose configuration remains available at
`experiments/config/experiment.yaml`. Use the article-specific configuration
above when reproducing the paper's two-sea-state trajectory campaign.
