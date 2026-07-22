"""Command-line entry point for BlueROV2 experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allows both:
#   python experiments/run_experiment.py
#   python -m experiments.run_experiment
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config_loader import load_config
from experiments.runner import ExperimentRunner

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config" / "experiment.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run BlueROV2 PID/PPO experiments from a YAML configuration file."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Experiment YAML file (default: {DEFAULT_CONFIG})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config)
    ExperimentRunner(config).run()


if __name__ == "__main__":
    main()
