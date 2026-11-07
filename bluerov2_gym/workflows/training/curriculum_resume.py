"""Resume a saved curriculum pair at a fresh simulation episode."""
from copy import deepcopy
from pathlib import Path
import pickle
import random

import numpy as np
import torch

from ..configuration import read_yaml


def load_resume(config):
    directory = config['training'].get('resume_run')
    if not directory:
        return None
    run = Path(directory).expanduser().resolve()
    manifest = run / 'manifest.yaml'
    if manifest.exists() and read_yaml(manifest).get('status') == 'created':
        raise ValueError('Stop the source training with Ctrl+C before using --resume-run.')
    old = read_yaml(run / 'config.resolved.yaml')
    for key in ['ppo', 'vecnormalize', 'environment']:
        if old[key] != config[key]:
            raise ValueError(f'Resume cannot change {key}; only evaluation/transition criteria may change.')
    old_stages = [{k: v for k, v in s.items() if k not in {'evaluation', 'transition'}} for s in old['curriculum']]
    new_stages = [{k: v for k, v in s.items() if k not in {'evaluation', 'transition'}} for s in config['curriculum']]
    if old_stages != new_stages or old['training'].get('seed') != config['training'].get('seed'):
        raise ValueError('Resume must preserve physical stages and training seed.')
    model_name = old['paths']['model_name']
    norm_name = Path(old['paths']['vecnormalize_name']).stem
    candidates = sorted((run/'models').glob(f'{model_name}*_curriculum_state.pkl'),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    for path in candidates:
        stem = path.name.removesuffix('_curriculum_state.pkl')
        model = path.with_name(stem + '.zip')
        normalizer = path.with_name(norm_name + stem[len(model_name):] + '.pkl')
        if not model.is_file() or not normalizer.is_file():
            continue
        with path.open('rb') as stream:
            state = pickle.load(stream)  # Locally generated trusted run artifacts.
        status = state['status']
        if status.get('status') == 'completed':
            raise ValueError('The source curriculum already completed.')
        index = status['index'] if status.get('status') == 'mastered' else status['index'] - 1
        if index >= len(config['curriculum']):
            raise ValueError('The source has already mastered all phases.')
        return dict(model=str(model), normalizer=str(normalizer), state=state,
                    index=index, status=deepcopy(status),
                    history=[s for s in state.get('history', []) if s['index'] <= index])
    raise FileNotFoundError(f'No complete PPO/normalizer/curriculum-state checkpoint pair in {run}')


def restore_rng(state, base_env):
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    torch.set_rng_state(state['torch_rng'])
    if torch.cuda.is_available() and len(state.get('cuda_rng', [])) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    base_env.context_rng.bit_generator.state = state['context_rng']
