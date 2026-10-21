"""Run the fixed-policy moderate/severe benchmark and regenerate its figures."""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import shutil
import yaml

ROOT = Path(__file__).resolve().parent


def prepare_reused_results(source, cfg, physical_config, dynamics_version):
    """Validate saved conventional runs before creating the new comparison."""
    source = source.expanduser().resolve()
    saved = yaml.safe_load((source / 'manifest.yaml').read_text())
    if saved['dynamics_version'] != dynamics_version or saved['physical_config'] != physical_config:
        raise ValueError('Saved results use different physics; cannot reuse this comparison.')
    previous = saved['config']
    if previous['experiment'] != cfg['experiment'] or previous['trajectories'] != cfg['trajectories']:
        raise ValueError('Saved experiment settings or trajectories differ from this run.')
    files = []
    for controller in ('pid', 'smc', 'nmpc'):
        old = next(c for c in previous['controllers'] if c['name'] == controller)
        new = next(c for c in cfg['controllers'] if c['name'] == controller)
        if old.get('params', {}) != new.get('params', {}):
            raise ValueError(f'Saved {controller} parameters differ from this run.')
        for sea in ('moderate', 'severe'):
            for trajectory in cfg['trajectories']:
                if not trajectory.get('enabled', True):
                    continue
                relative = Path(sea) / 'trajectory_experiments' / f'{controller}_{trajectory["name"]}.csv'
                path = source / relative
                with path.open(newline='') as stream:
                    rows = list(csv.DictReader(stream))
                expected = cfg['experiment']['steps'] * cfg['experiment'].get('repetitions', 1)
                if len(rows) != expected:
                    raise ValueError(f'Incomplete saved results: {path} ({len(rows)} rows, expected {expected})')
                files.append((path, relative))
    return source, files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'experiments/config/article_trajectories.yaml')
    parser.add_argument('--results-root', type=Path, default=ROOT / 'training_runs/ppo_physics_v2_seed42/evaluation')
    parser.add_argument('--steps', type=int, help='Override steps per run (use a small value for a smoke test).')
    parser.add_argument('--model', help='Fixed PPO model used for every comparison.')
    parser.add_argument('--vecnormalize', help='Matching VecNormalize file; required with --model.')
    parser.add_argument('--plots-only', action='store_true')
    parser.add_argument('--skip-plots', action='store_true')
    parser.add_argument('--reuse-results', type=Path,
                        help='Reuse saved PID/SMC/NMPC CSVs; simulate only PPO in a new results directory.')
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
        from bluerov2_gym.envs.core.dynamics import Dynamics
        from bluerov2_gym.envs.core.config_utils import load_yaml
        manifest = {'dynamics_version': Dynamics.MODEL_VERSION, 'coordinate_convention': 'right-handed z-up (world and body)', 'physical_config': load_yaml(ROOT / 'config/dynamics_config.yaml'), 'spectrum': 'Fossen MSS JONSWAP case 7, DNV correction', 'artifacts': artifacts, 'config': cfg}
        if args.reuse_results:
            source, files = prepare_reused_results(args.reuse_results, cfg,
                                                   manifest['physical_config'], Dynamics.MODEL_VERSION)
            provenance = []
            for path, relative in files:
                destination = output / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                provenance.append({'file': str(relative), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            manifest['reused_results'] = {'source': str(source), 'files': provenance,
                                          'simulated_controllers': ['ppo']}
            # Keep the complete comparison configuration in the manifest.
            import copy
            execution_cfg = copy.deepcopy(cfg)
            for controller in execution_cfg['controllers']:
                controller['enabled'] = controller['name'] == 'ppo'
            print(f'Reusing {len(files)} saved CSVs from {source}; simulating PPO only.', flush=True)
        else:
            execution_cfg = cfg
        runner = ExperimentRunner(execution_cfg)
        (output / 'manifest.yaml').write_text(yaml.safe_dump(manifest, sort_keys=False))
        runner.run()
    if not args.skip_plots:
        env = dict(os.environ, BLUEROV_RESULTS_ROOT=str(output), BLUEROV_FIGURES_ROOT=str(output / 'figures'), MPLBACKEND='Agg')
        subprocess.run([sys.executable, str(ROOT / 'plots/run_all.py')], check=True, env=env)
        for scenario in ['moderate', 'severe']:
            subprocess.run([sys.executable, str(ROOT / 'experiments/plots/plot_trajectory.py'), '--scenario', scenario, '--results-root', str(output)], check=True, env=env)


if __name__ == '__main__':
    main()
