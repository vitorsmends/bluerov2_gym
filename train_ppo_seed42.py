"""Train the commit-26857c4 PPO pipeline with only the Physics 2 dynamics."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml

import gymnasium
import numpy
import scipy
import stable_baselines3
import torch

from training.train_curriculum import load_training_config, train_model_with_curriculum

ROOT = Path(__file__).resolve().parent
REFERENCE_COMMIT = "26857c4456e11c6173b04678b21b6806d77b5b4c"


def verify_reference_environment():
    expected = {
        "Python": ("3.10", ".".join(map(str, sys.version_info[:2]))),
        "NumPy": ("2.2.6", numpy.__version__),
        "SciPy": ("1.15.3", scipy.__version__),
        "PyYAML": ("6.0.2", yaml.__version__),
        "Gymnasium": ("1.2.3", gymnasium.__version__),
        "Stable-Baselines3": ("2.8.0", stable_baselines3.__version__),
        "Torch": ("2.11.0", torch.__version__.split("+")[0]),
    }
    mismatches = [f"{name}: expected {want}, found {got}"
                  for name, (want, got) in expected.items() if want != got]
    if mismatches:
        raise RuntimeError(
            "The PPO software environment differs from the validated artifact:\n  "
            + "\n  ".join(mismatches)
            + "\nCreate a Python 3.10 environment and install requirements-ppo-reference.txt."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "training_runs/ppo_26857_physics2_seed42")
    parser.add_argument("--evaluate", action="store_true",
                        help="After training, compare all four controllers and generate figures.")
    parser.add_argument("--reuse-results", type=Path,
                        help="After training, simulate only PPO and compare with saved PID/SMC/NMPC results.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Only verify the training pipeline (320 steps); not a scientific training run.")
    parser.add_argument("--allow-version-drift", action="store_true",
                        help="Diagnostic only: bypass the validated PPO package-version check.")
    args = parser.parse_args()
    if args.smoke_test and (args.evaluate or args.reuse_results):
        parser.error("--evaluate requires the full training run; omit --smoke-test")
    if not args.allow_version_drift:
        verify_reference_environment()
    os.chdir(ROOT)
    output = args.output_dir.expanduser().resolve()
    if args.smoke_test:
        output = output / "smoke_test"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}. Choose another --output-dir.")
    if args.reuse_results:
        from run_oceans import prepare_reused_results
        from experiments.config_loader import load_config
        from bluerov2_gym.envs.core.dynamics import Dynamics
        comparison = load_config(ROOT / 'experiments/config/article_trajectories.yaml')
        physics = yaml.safe_load((ROOT / 'config/dynamics_config.yaml').read_text())
        source, files = prepare_reused_results(args.reuse_results, comparison, physics, Dynamics.MODEL_VERSION)
        print(f'Validated {len(files)} saved PID/SMC/NMPC files from {source}', flush=True)

    # The complete policy, reward, normalization, optimizer and curriculum
    # configuration comes from the reference commit. Only storage paths and the
    # location of the current Physics 2 dynamics configuration are resolved here.
    config = load_training_config(ROOT / "config/training_config.yaml")
    config["environment"]["dynamics_config_file"] = str(ROOT / "config/dynamics_config.yaml")
    config["paths"].update(
        model_dir=str(output / "models"),
        tensorboard_log=str(output / "tensorboard"),
        model_name="bluerov_ppo_curriculum",
        vecnormalize_name="bluerov_vec_normalize_curriculum.pkl",
    )
    if args.smoke_test:
        config["ppo"].update(n_steps=64, batch_size=64, n_epochs=1)
        for stage in config["curriculum"]:
            stage["timesteps"] = 64
    config["experiment"] = {
        "purpose": "pipeline_smoke_test" if args.smoke_test else "reference_26857_physics2_retraining",
        "training_reference_commit": REFERENCE_COMMIT,
        "allowed_functional_difference": "Physics 2 dynamics and dynamics_config only",
        "initialization": "fresh PPO weights and VecNormalize, as in the reference pipeline",
    }
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Reference: {REFERENCE_COMMIT}", flush=True)
    print(f"Seed: 42 | registered episode limit: 5000 | nominal steps: {sum(s['timesteps'] for s in config['curriculum']):,}", flush=True)
    train_model_with_curriculum(config_path=config_path, seed=42, separate_run=True)
    model = output / "models/bluerov_ppo_curriculum_seed_42.zip"
    normalizer = output / "models/bluerov_vec_normalize_curriculum_seed_42.pkl"
    print(f"\nModel: {model}\nNormalizer: {normalizer}", flush=True)
    if args.evaluate or args.reuse_results:
        command = [sys.executable, str(ROOT / "run_oceans.py"),
                        "--model", str(model), "--vecnormalize", str(normalizer),
                        "--results-root", str(output / "evaluation")]
        if args.reuse_results:
            command += ["--reuse-results", str(args.reuse_results.expanduser().resolve())]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
