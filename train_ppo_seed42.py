"""Train a fresh seed-42 PPO policy in the corrected physical environment."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import yaml

from training.train_curriculum import load_training_config, train_model_with_curriculum

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "training_runs/ppo_physics_v2_seed42")
    parser.add_argument("--evaluate", action="store_true",
                        help="After training, compare all four controllers and generate figures.")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Only verify the training pipeline (320 steps); not a scientific training run.")
    args = parser.parse_args()
    if args.smoke_test and args.evaluate:
        parser.error("--evaluate requires the full training run; omit --smoke-test")
    os.chdir(ROOT)
    output = args.output_dir.expanduser().resolve()
    if args.smoke_test:
        output = output / "smoke_test"
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Training output is not empty: {output}. Choose another --output-dir.")

    config = load_training_config(ROOT / "config/training_config.yaml")
    config["environment"]["dynamics_config_file"] = str(ROOT / "config/dynamics_config.yaml")
    config["environment"]["max_episode_steps"] = 1000
    config["training"]["seed"] = 42
    config["training"]["tb_log_name"] = "PPO_Physics_v2"
    config["paths"].update(
        model_dir=str(output / "models"),
        tensorboard_log=str(output / "tensorboard"),
        model_name="bluerov_ppo_physics_v2",
        vecnormalize_name="bluerov_vecnormalize_physics_v2.pkl",
    )
    config["checkpoint"]["name_prefix"] = "ppo_physics_v2_checkpoint"
    if args.smoke_test:
        config["ppo"].update(n_steps=64, batch_size=64, n_epochs=1)
        for stage in config["curriculum"]:
            stage["timesteps"] = 64
    config["experiment"] = {"purpose": "pipeline_smoke_test" if args.smoke_test else "physics_v2_retraining"}
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Seed: 42 | episode limit: 1000 | nominal steps: {sum(s['timesteps'] for s in config['curriculum']):,}", flush=True)
    train_model_with_curriculum(config_path=config_path, seed=42, separate_run=True)
    model = output / "models/bluerov_ppo_physics_v2_seed_42.zip"
    normalizer = output / "models/bluerov_vecnormalize_physics_v2_seed_42.pkl"
    print(f"\nModel: {model}\nNormalizer: {normalizer}", flush=True)
    if args.evaluate:
        subprocess.run([sys.executable, str(ROOT / "run_oceans.py"),
                        "--model", str(model), "--vecnormalize", str(normalizer),
                        "--results-root", str(output / "evaluation")], check=True)


if __name__ == "__main__":
    main()
