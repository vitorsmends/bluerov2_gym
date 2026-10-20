from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


METRICS = (
    ("rollout/ep_rew_mean", "Mean episode reward", "Episode reward", "episode_reward", "#59A14F"),
    ("rollout/ep_len_mean", "Mean episode length", "Steps per episode", "episode_length", "#4E79A7"),
    ("train/loss", "PPO total loss", "Loss", "total_loss", "#E15759"),
    ("train/policy_gradient_loss", "Policy gradient loss", "Loss", "policy_gradient_loss", "#B07AA1"),
    ("train/value_loss", "Value loss", "Loss", "value_loss", "#F28E2B"),
    ("train/entropy_loss", "Entropy loss", "Loss", "entropy_loss", "#76B7B2"),
    ("train/explained_variance", "Explained variance", "Explained variance", "explained_variance", "#EDC948"),
    ("train/approx_kl", "Approximate KL divergence", "KL divergence", "approx_kl", "#FF9DA7"),
    ("train/clip_fraction", "PPO clip fraction", "Fraction", "clip_fraction", "#9C755F"),
    ("train/std", "Policy action standard deviation", "Standard deviation", "action_std", "#BAB0AC"),
)
ScalarMetrics = dict[str, tuple[np.ndarray, np.ndarray]]
MINIMUM_SEEDS = 5


def load_scalars(run_dir: Path) -> ScalarMetrics:
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    available_tags = set(accumulator.Tags().get("scalars", []))
    metrics = {}

    for tag, _, _, _, _ in METRICS:
        if tag not in available_tags:
            continue

        events = accumulator.Scalars(tag)
        steps = np.asarray([event.step for event in events], dtype=np.int64)
        values = np.asarray([event.value for event in events], dtype=float)
        valid = np.isfinite(values)
        latest_values = {int(step): float(value) for step, value in zip(steps[valid], values[valid])}
        ordered_steps = np.asarray(sorted(latest_values), dtype=np.int64)
        metrics[tag] = (
            ordered_steps,
            np.asarray([latest_values[int(step)] for step in ordered_steps], dtype=float),
        )

    return metrics


def find_run(log_dir: Path, requested_run: str | None) -> tuple[Path, ScalarMetrics]:
    event_files = list(log_dir.rglob("events.out.tfevents.*"))
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in: {log_dir}")

    run_dirs = {path.parent for path in event_files}
    if requested_run is not None:
        run_dir = (log_dir / requested_run).resolve()
        if run_dir not in {path.resolve() for path in run_dirs}:
            available = ", ".join(sorted(path.relative_to(log_dir).as_posix() for path in run_dirs))
            raise FileNotFoundError(
                f"No TensorBoard run named '{requested_run}' in {log_dir}. "
                f"Available runs: {available}"
            )
        metrics = load_scalars(run_dir)
        if not metrics:
            raise RuntimeError(f"No supported PPO scalar metrics found in: {run_dir}")
        return run_dir, metrics

    sorted_runs = sorted(
        run_dirs,
        key=lambda path: max(event.stat().st_mtime for event in path.glob("events.out.tfevents.*")),
        reverse=True,
    )
    for run_dir in sorted_runs:
        metrics = load_scalars(run_dir)
        if metrics:
            return run_dir, metrics

    raise RuntimeError(f"No supported PPO scalar metrics found in TensorBoard logs under: {log_dir}")


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = min(window, len(values))
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return (cumulative[window:] - cumulative[:-window]) / window


def plot_training(
    run_dir: Path,
    metrics: ScalarMetrics,
    output_dir: Path,
    smooth_window: int,
) -> list[Path]:
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for tag, title, ylabel, filename, color in METRICS:
        data = metrics.get(tag)
        if data is None or len(data[0]) == 0:
            print(f"[WARN] Skipped {title}: metric not logged in {run_dir.name}")
            continue

        fig, ax = plt.subplots(figsize=(6.5, 4.2))
        steps, values = data
        smoothed = moving_average(values, smooth_window)
        ax.plot(steps, values, color=color, alpha=0.25, linewidth=0.9, label="Raw")
        ax.plot(
            steps[len(steps) - len(smoothed):],
            smoothed,
            color=color,
            linewidth=1.8,
            label=f"Moving average ({min(smooth_window, len(values))} points)",
        )
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        pdf_path = output_dir / f"ppo_{filename}_{run_dir.name}.pdf"
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(pdf_path)

    if not output_paths:
        raise RuntimeError(f"No supported PPO scalar metrics found in: {run_dir}")

    return output_paths


def parse_groups(log_dir: Path, group_specs: list[str], min_seeds: int) -> dict[str, list[Path]]:
    if group_specs:
        groups = {}
        for spec in group_specs:
            if "=" not in spec:
                raise ValueError(f"Invalid group '{spec}'. Expected LABEL=RUN_GLOB.")
            label, pattern = spec.split("=", 1)
            if not label or not pattern:
                raise ValueError(f"Invalid group '{spec}'. Expected LABEL=RUN_GLOB.")
            if label in groups:
                raise ValueError(f"Duplicate group label: {label}")
            matches = sorted(
                path for path in log_dir.glob(pattern)
                if path.is_dir() and list(path.glob("events.out.tfevents.*"))
            )
            if len(matches) < min_seeds:
                raise ValueError(
                    f"Group '{label}' matched {len(matches)} TensorBoard runs; "
                    f"at least {min_seeds} are required."
                )
            groups[label] = matches
        return groups

    seed_runs = sorted(
        path for path in log_dir.rglob("*")
        if path.is_dir()
        and "_seed_" in path.name
        and list(path.glob("events.out.tfevents.*"))
    )
    if not seed_runs:
        return {}
    groups: dict[str, list[Path]] = {}
    for run_dir in seed_runs:
        match = re.match(r"^(.*)_seed_-?\d+(?:_|$)", run_dir.name)
        if match is None:
            continue
        groups.setdefault(match.group(1), []).append(run_dir)
    if seed_runs and not groups:
        raise ValueError(
            "Found seed-specific TensorBoard directories, but their names do not follow "
            "the expected '<run>_seed_<integer>' pattern. Select them with --group."
        )
    incomplete = {label: len(runs) for label, runs in groups.items() if len(runs) < min_seeds}
    if incomplete:
        summary = ", ".join(f"{label}: {count}" for label, count in incomplete.items())
        raise ValueError(
            f"Seed-specific groups with fewer than {min_seeds} runs were found ({summary}). "
            "Complete the runs or select complete groups explicitly."
        )
    return groups


def plot_seed_groups(
    groups: dict[str, list[Path]],
    output_dir: Path,
    min_seeds: int,
) -> list[Path]:
    loaded = {
        label: [(run_dir, load_scalars(run_dir)) for run_dir in run_dirs]
        for label, run_dirs in groups.items()
    }
    required_tag = "rollout/ep_rew_mean"
    for label, runs in loaded.items():
        usable = sum(required_tag in metrics for _, metrics in runs)
        if usable < min_seeds:
            raise RuntimeError(
                f"Group '{label}' has {usable} runs with {required_tag}; "
                f"at least {min_seeds} are required."
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    output_paths = []

    for tag, title, ylabel, filename, color in METRICS:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        plotted_groups = 0
        for label, runs in loaded.items():
            valid_runs = [
                (run_dir, metrics[tag])
                for run_dir, metrics in runs
                if tag in metrics and len(metrics[tag][0]) >= 2
            ]
            if len(valid_runs) < min_seeds:
                print(
                    f"[WARN] Skipped {label} / {title}: "
                    f"{len(valid_runs)} runs contain this metric; need {min_seeds}."
                )
                continue

            start = max(data[0][0] for _, data in valid_runs)
            end = min(data[0][-1] for _, data in valid_runs)
            if start >= end:
                print(f"[WARN] Skipped {label} / {title}: runs have no shared timestep range.")
                continue

            grid = np.unique(np.concatenate([
                steps[(steps >= start) & (steps <= end)]
                for _, (steps, _) in valid_runs
            ]))
            aligned = np.vstack([
                np.interp(grid, steps, values)
                for _, (steps, values) in valid_runs
            ])
            mean = np.mean(aligned, axis=0)
            std = np.std(aligned, axis=0, ddof=1)
            ax.plot(grid, mean, linewidth=1.9, label=f"{label} (n={len(valid_runs)})")
            ax.fill_between(grid, mean - std, mean + std, alpha=0.18)
            plotted_groups += 1

        if not plotted_groups:
            plt.close(fig)
            continue
        ax.set_title(title)
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(frameon=True, fontsize=9)
        fig.tight_layout()
        safe_filename = re.sub(r"[^A-Za-z0-9_-]+", "_", filename)
        output_path = output_dir / f"ppo_{safe_filename}_multi_seed.pdf"
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(output_path)

    if not output_paths:
        raise RuntimeError("No supported TensorBoard metrics were available for multi-seed plots.")
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot individual PPO diagnostics or multi-seed mean and sample-standard-deviation bands."
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("bluerov_tensorboard"),
        help="Directory containing TensorBoard run folders (default: bluerov_tensorboard).",
    )
    parser.add_argument(
        "--run",
        help="Run folder name or relative path. Defaults to the most recently updated run with PPO metrics.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots_training"),
        help="Directory for generated PDF figures (default: plots_training).",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=10,
        help="Moving-average window in logged points (default: 10).",
    )
    parser.add_argument(
        "--group",
        action="append",
        default=[],
        metavar="LABEL=RUN_GLOB",
        help=(
            "Aggregate matching TensorBoard run directories as one group. "
            "Repeat for ablation groups; each must contain at least --min-seeds runs."
        ),
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=MINIMUM_SEEDS,
        help=f"Minimum independent runs required for aggregate plots (default: {MINIMUM_SEEDS}).",
    )
    args = parser.parse_args()
    if args.smooth_window < 1:
        parser.error("--smooth-window must be greater than zero.")
    if args.min_seeds < 2:
        parser.error("--min-seeds must be at least 2 to estimate a standard deviation.")
    return args


def main() -> None:
    args = parse_args()
    log_dir = args.log_dir.expanduser().resolve()
    groups = parse_groups(log_dir, args.group, args.min_seeds) if args.run is None else {}
    if groups and args.run is None:
        output_paths = plot_seed_groups(groups, args.output_dir, args.min_seeds)
        for output_path in output_paths:
            print(f"[OK] Saved: {output_path}")
        return

    run_dir, metrics = find_run(log_dir, args.run)
    output_paths = plot_training(run_dir, metrics, args.output_dir, args.smooth_window)
    if not groups:
        print(
            "[WARN] These are single-run diagnostics, not multi-seed evidence. "
            "Use --seeds during training and --group for training-seed or ablation comparisons."
        )
    print(f"TensorBoard run: {run_dir}")
    for output_path in output_paths:
        print(f"[OK] Saved: {output_path}")


if __name__ == "__main__":
    main()
