"""Create exclusive run directories and record resolved inputs."""
from datetime import datetime
import hashlib
import importlib.metadata
from pathlib import Path
import subprocess
from zoneinfo import ZoneInfo
import yaml


def write_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(value,sort_keys=False),encoding='utf-8')


def create_run(kind, requested, resolved, output=None):
    stamp=datetime.now(ZoneInfo('America/Bahia')).strftime('%Y%m%d-%H%M%S-%f')
    run=Path(output or Path('runs')/kind/stamp).expanduser().resolve()
    run.mkdir(parents=True,exist_ok=False)
    for name in ['data','figures','tables','logs']: (run/name).mkdir()
    write_yaml(run/'config.requested.yaml',requested)
    write_yaml(run/'config.resolved.yaml',resolved)
    repo=Path(__file__).resolve().parents[2]
    def git(*args):
        result=subprocess.run(['git','-C',str(repo),*args],capture_output=True,text=True)
        return result.stdout.strip() if result.returncode==0 else None
    versions={}
    for package in ['numpy','scipy','matplotlib','gymnasium','stable-baselines3','torch']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    artifacts=[]
    for item in resolved.get('controllers',[]):
        if item.get('enabled',True):
            for key in ['model_path','vecnormalize_path']:
                if item.get('params',{}).get(key):artifacts.append(item['params'][key])
    artifacts += [resolved[k] for k in ['model','normalizer'] if resolved.get(k)]
    hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in artifacts if Path(p).is_file()}
    write_yaml(run/'manifest.yaml',dict(status='created',git_commit=git('rev-parse','HEAD'),
        git_status=git('status','--porcelain'),versions=versions,artifact_sha256=hashes,
        training_seed=resolved.get('training',{}).get('seed'),
        disturbance_seed=resolved.get('experiment',{}).get('seed',resolved.get('seed',42))))
    return run


def update_manifest(run, **fields):
    from .configuration import read_yaml
    path=Path(run)/'manifest.yaml'
    value=read_yaml(path);value.update(fields);write_yaml(path,value)
