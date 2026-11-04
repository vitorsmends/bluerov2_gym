"""Move historical generated files into artifacts, preserving SHA-256 checksums.

Dry run by default. Use --apply to move files and prune empty project folders.
Source, reference, environment and Git directories are never traversed for pruning.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bluerov2_gym.workflows.paths import ARTIFACTS_ROOT

MAPPINGS = {
    'models': 'models/curriculum', 'models_ftc': 'models/ftc',
    'bluerov_tensorboard': 'logs/tensorboard', 'bluerov_tensorboard_ftc': 'logs/tensorboard_ftc',
    'ppo_traj_tensorboard': 'logs/trajectory_tensorboard',
    'training_runs': 'training/imported', 'runs': 'imported_runs',
    'results': 'evaluation/imported/results',
    'results-moderate': 'evaluation/imported/results-moderate',
    'results-severe': 'evaluation/imported/results-severe',
    'article_plots_simple': 'figures/imported/article_plots_simple',
    'article_tables_simple': 'tables/imported/article_tables_simple',
    'plots_training': 'figures/imported/training',
    'plots_optuna': 'figures/imported/optuna', 'optuna_plots': 'figures/imported/optuna',
    'optuna_results': 'optimization/imported',
    'bluerov_ppo.zip': 'models/current/bluerov_ppo.zip',
    'bluerov_vec_normalize.pkl': 'models/current/bluerov_vec_normalize.pkl',
    'ppo_trajectory_final.zip': 'models/current/ppo_trajectory_final.zip',
    'vec_normalize.pkl': 'models/current/vec_normalize.pkl',
    'policy.pth': 'models/extracted/policy.pth',
    'policy.optimizer.pth': 'models/extracted/policy.optimizer.pth',
    'pytorch_variables.pth': 'models/extracted/pytorch_variables.pth',
    '_stable_baselines3_version': 'models/extracted/_stable_baselines3_version',
    'experiments.zip': 'archive/experiments.zip',
}


def checksum(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args=parser.parse_args(argv)
    plan=[]
    for source,destination in MAPPINGS.items():
        src,dst=ROOT/source,ARTIFACTS_ROOT/destination
        if not src.exists():continue
        if dst.exists():raise FileExistsError(f'Destination already exists: {dst}')
        plan.append((src,dst))
    # Validate every destination before making any changes.
    if not args.apply:
        for src,dst in plan:print(f'{src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}')
        return
    records=[]
    tracked=set(subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines())
    for src,dst in plan:
        files=[src] if src.is_file() else sorted(p for p in src.rglob('*') if p.is_file())
        entries=[dict(source=str(p.relative_to(ROOT)),destination=str((dst if src.is_file() else dst/p.relative_to(src)).relative_to(ROOT)),
                      bytes=p.stat().st_size,sha256=checksum(p)) for p in files]
        dst.parent.mkdir(parents=True,exist_ok=True)
        source=str(src.relative_to(ROOT));destination=str(dst.relative_to(ROOT))
        if any(p==source or p.startswith(source+'/') for p in tracked):
            subprocess.run(['git','mv',source,destination],cwd=ROOT,check=True)
        else:shutil.move(str(src),str(dst))
        for entry in entries:
            if checksum(ROOT/entry['destination'])!=entry['sha256']:
                raise RuntimeError(f'Checksum mismatch: {entry}')
        records.extend(entries)
        print(f'Moved {source}: {len(entries)} files, all checksums match')
    # Keep existing manifests; each application has its own name.
    report=None
    if records:
        report=ARTIFACTS_ROOT/'migration'/f'{datetime.now(timezone.utc):%Y%m%dT%H%M%S%fZ}.json'
        report.parent.mkdir(parents=True,exist_ok=True)
        report.write_text(json.dumps(records,indent=2))
    # Metadata caches are disposable; source and reference files are preserved.
    protected={'.git','.venv','.venv-ppo-reference','_references'}
    def prune(directory):
        for child in list(directory.iterdir()):
            if child.is_symlink():continue
            if child.is_dir():
                if child.name in protected or child.name.startswith('.') or child.name.endswith('.egg-info'):continue
                prune(child)
                if not any(child.iterdir()):child.rmdir()
    prune(ROOT)
    print(f'Preserved {len(records)} files. Manifest: {report}')

if __name__=='__main__':main()
