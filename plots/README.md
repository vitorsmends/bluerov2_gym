# Article result figures

Run `.venv/bin/python run_oceans.py` from the repository root to collect the
moderate/severe benchmark and generate all figures. See
[experiment instructions](../experiments/README.md) for policy selection,
output locations and the corrected spectrum.

Regenerate figures from an existing campaign:

```bash
.venv/bin/python run_oceans.py --plots-only --results-root training_runs/ppo_physics_v2_seed42/evaluation
```

The default campaign uses one fixed PPO policy and one disturbance realization
per condition. Aggregate plots describe variation across tasks and sea states.
Outputs are isolated in `training_runs/ppo_physics_v2_seed42/evaluation/figures` and `training_runs/ppo_physics_v2_seed42/evaluation/tables`.
The plotting pipeline requires all 40 CSVs and CPU timing columns before writing
aggregate figures. Energy is estimated using `plots/t200_performance_16v.csv`.

For single-run training diagnostics, run:

```bash
python training/plot_training.py
```

The default output directory is `plots_training/`. To summarize the existing
five baseline PPO training seeds:

```bash
python training/plot_training.py \
  --group "Baseline=PPO_BlueROV2_Curriculum_seed_*" \
  --output-dir oceans2026/figs
```

Multi-seed training produces one article-oriented figure,
`ppo_training_summary_multi_seed.pdf`, with three vertically stacked subplots:
mean episode reward, explained variance, and PPO loss. Each subplot has its
own y-axis and sample-standard-deviation band. Colors identify training
groups, shown in a lower-right legend. Reward, loss, and environment-step axes
use scientific notation when needed to keep large values readable. Runs are
aligned over their shared training interval, and at least five independent
training runs are required. This summary does not include episode length or
the remaining PPO optimization diagnostics; those remain available in the
individual-run diagnostic plots.

Evaluation repetitions are not treated as training seeds. Additional
configurations can be included with repeated `--group LABEL=RUN_GLOB`
arguments, but only when their training-seed logs are available.
