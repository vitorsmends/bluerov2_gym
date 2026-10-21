"""Run the fixed-policy moderate/severe benchmark and regenerate its figures."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/config/article_trajectories.yaml')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'results-physics-v2')
    parser.add_argument('--steps', type=int, help='Override steps per run (use a small value for a smoke test).')
    parser.add_argument('--model', help='Fixed PPO model used for every comparison.')
    parser.add_argument('--vecnormalize', help='Matching VecNormalize file; required with --model.')
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.results_root.resolve()
    if not args.plots_only:
        from experiments.config_loader import load_config
        from experiments.runner import ExperimentRunner
        from experiments.controllers.ppo import PPOController
        config_path = args.config.resolve()
        cfg = load_config(config_path)
        disturbance_path = Path(cfg['disturbance']['config_path'])
        if not disturbance_path.is_absolute():
            relative = config_path.parent / disturbance_path
            disturbance_path = relative if relative.exists() else ROOT / disturbance_path
        cfg['disturbance']['config_path'] = str(disturbance_path.resolve())
        cfg['disturbance']['scenario_names'] = ['moderate', 'severe']
        cfg['output']['directory_template'] = str(output / '{scenario}/trajectory_experiments')
        if args.steps is not None:
            if args.steps < 1:
                parser.error('--steps must be positive')
            cfg['experiment']['steps'] = args.steps
        if bool(args.model) != bool(args.vecnormalize):
            parser.error('Supply --model and --vecnormalize together')
        ppo_cfg = next(c for c in cfg['controllers'] if c['name'] == 'ppo')['params']
        if args.model:
            ppo_cfg.update(model_path=args.model, vecnormalize_path=args.vecnormalize)
        # Preflight the actual policy/normalizer before creating result files.
        policy = PPOController(**ppo_cfg)
        artifacts = {}
        for label, path in [('model', policy.model_path), ('vecnormalize', policy.vecnormalize_path)]:
            artifacts[label] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        policy.vec_env.close()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f'Result directory is not empty: {output}. Choose a new --results-root.')
        output.mkdir(parents=True, exist_ok=True)
        runner = ExperimentRunner(cfg)
        from bluerov2_gym.envs.core.dynamics import Dynamics
        from bluerov2_gym.envs.core.config_utils import load_yaml
        manifest = {'dynamics_version': Dynamics.MODEL_VERSION, 'coordinate_convention': 'right-handed z-up (world and body)', 'physical_config': load_yaml(ROOT / 'config/dynamics_config.yaml'), 'spectrum': 'Fossen MSS JONSWAP case 7, DNV correction', 'artifacts': artifacts, 'config': cfg}
        (output / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        runner.run()
    if not args.skip_plots:
        env = dict(os.environ, BLUEROV_RESULTS_ROOT=str(output), BLUEROV_FIGURES_ROOT=str(output / 'figures'), MPLBACKEND='Agg')
        subprocess.run([sys.executable, str(ROOT / 'plots/run_all.py')], check=True, env=env)
        for scenario in ['moderate', 'severe']:
            subprocess.run([sys.executable, str(ROOT / 'experiments/plots/plot_trajectory.py'), '--scenario', scenario, '--results-root', str(output)], check=True, env=env)


if __name__ == '__main__':
    main()
