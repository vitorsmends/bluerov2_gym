# Article result figures

Run the article trajectory benchmark first, from the repository root:

```bash
python experiments/run_experiment.py \
  --config experiments/config/article_trajectories.yaml
```

This produces one CSV per controller and trajectory in each sea-state
directory. Every file contains all five repetitions and identifies them with
`repetition_id` and `seed`.

Legacy CSVs already present in the results directories have only one run per
condition and no seed/repetition columns. The plotting pipeline treats those
as one run; its standard deviations then describe variation across the
different trajectories or conditions, not statistical uncertainty from
repeated evaluation. Rerun the campaign above to produce the configured
repeated-evaluation dataset.

Generate the article figures and summary tables with:

```bash
python plots/run_all.py
```

The figure scripts write PDFs and PNGs to `oceans2026/figs/` and tables to
`article_tables_simple/`. Time-series plots compute statistics over
trajectory/repetition runs; RMSE and Pareto tables first compute one metric
per trajectory and repetition. The energy plot interpolates the T200 thrust-
power data in `plots/t200_performance_16v.csv`.

For single-run training diagnostics, run:

```bash
python training/plot_training.py
```

The default output directory is `plots_training/`. To create learning curves
across five independent PPO trainings, first run each seed with separate model,
normalization, checkpoint, and TensorBoard outputs:

```bash
python training/train_curriculum.py \
  --config config/training_config.yaml \
  --seeds 42 43 44 45 46
python training/plot_training.py
```

Aggregate figures align runs over their shared environment-step interval and
show the mean with the sample-standard-deviation band. The script includes
episode reward, episode length, and available PPO optimization diagnostics.
It requires at least five runs for an aggregate plot and will not treat
evaluation repetitions as training seeds.

To compare an ablation with the baseline, run the same seed set and pass
TensorBoard run-name patterns (quote globs so the shell does not expand them):

```bash
python training/train_curriculum.py \
  --config config/training_config.yaml \
  --ablation no_curriculum \
  --seeds 42 43 44 45 46
python training/plot_training.py \
  --group "baseline=PPO_BlueROV2_Curriculum_seed_*" \
  --group "no curriculum=PPO_BlueROV2_Curriculum_ablation_no_curriculum_seed_*"
```

Repeat `--group` for additional configurations. Every group must match at least
five event-log directories. Use `--run RUN_NAME` for one-run diagnostics or
`--min-seeds N` only when intentionally exploring a smaller pilot; such a
pilot is not evidence of the requested five-seed robustness.
