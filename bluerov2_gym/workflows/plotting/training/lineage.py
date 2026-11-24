"""Build a derived training-history view along the actual checkpoint ancestry.

Original runs/checkpoints remain untouched. TensorBoard scalars retain recorded
steps and timestamps; no missing samples or discarded branches are interpolated.
"""
from pathlib import Path
import argparse
from copy import deepcopy
import hashlib
import json
import math
import pickle
import shutil

import yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary


def read(path):
    return yaml.safe_load(Path(path).read_text())


def ancestry(run, end=None, seen=None):
    run = Path(run).expanduser().resolve()
    seen = set() if seen is None else seen
    if run in seen:
        raise ValueError('Cycle in training checkpoint ancestry.')
    seen.add(run)
    config = read(run/'config.resolved.yaml')
    status = read(run/'models/curriculum_status.yaml')
    end = int(status['step']) if end is None else int(end)
    training = config['training']
    parent = training.get('resume_run')
    if not parent:
        return [dict(run=str(run), start_exclusive=0, end_inclusive=end, checkpoint=None)]
    parent = Path(parent).expanduser()
    if not parent.is_absolute():
        raise ValueError('resume_run must be absolute in a resolved configuration.')
    checkpoint = training.get('resume_checkpoint')
    if not checkpoint or Path(checkpoint).name != checkpoint:
        raise ValueError(f'An explicit resume_checkpoint is required to establish ancestry: {run}')
    stem = checkpoint.removesuffix('.zip')
    snapshot = parent/'models'/f'{stem}_curriculum_state.pkl'
    with snapshot.open('rb') as stream:
        state = pickle.load(stream)  # Trusted local training artifacts, as in load_resume.
    start = int(state['status']['step'])
    if not 0 <= start < end:
        raise ValueError(f'Invalid retained interval ({start}, {end}] in {run}')
    previous = read(parent/'config.resolved.yaml')
    for key in ['environment', 'ppo', 'vecnormalize']:
        if previous[key] != config[key]:
            raise ValueError(f'Cannot silently join a change in {key}: {run}')
    if previous['training'].get('seed') != training.get('seed'):
        raise ValueError('Cannot join independently seeded training runs.')
    return ancestry(parent, start, seen) + [dict(run=str(run), start_exclusive=start,
        end_inclusive=end, checkpoint=str(snapshot),
        progression=dict(early_phase_skipping=config.get('early_phase_skipping'),
                         plateau_early_stopping=config.get('plateau_early_stopping')))]


def build(run, output):
    run, output = Path(run).resolve(), Path(output).resolve()
    segments = ancestry(run)
    if output.exists():
        raise FileExistsError(f'Choose a new derived output directory: {output}')
    output.mkdir(parents=True)
    for folder in ['models', 'logs/curriculum_evaluations', 'logs/tensorboard/retained_history']:
        (output/folder).mkdir(parents=True, exist_ok=True)
    config = deepcopy(read(run/'config.resolved.yaml'))
    config['lineage_view'] = dict(source_run=str(run), segments=segments,
        resume_steps=[s['start_exclusive'] for s in segments[1:]],
        note='Derived plotting view, not an uninterrupted training execution or a model directory.')
    (output/'config.resolved.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    shutil.copy2(run/'models/curriculum_status.yaml', output/'models/curriculum_status.yaml')
    original_manifest = read(run/'manifest.yaml')
    (output/'manifest.yaml').write_text(yaml.safe_dump(dict(status=original_manifest['status'],
        kind='derived_training_lineage_view', source_run=str(run), seed=config['training'].get('seed')), sort_keys=False))
    rows, batches, hashes, counts = {}, {}, {}, []
    def digest(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for segment in segments:
        source = Path(segment['run'])
        low, high = segment['start_exclusive'], segment['end_inclusive']
        total, retained = 0, 0
        for relative in ['config.resolved.yaml', 'manifest.yaml', 'models/curriculum_status.yaml']:
            digest(source/relative)
        if segment['checkpoint']:
            digest(Path(segment['checkpoint']))
        files = sorted((source/'logs/tensorboard').rglob('events.out.tfevents.*'))
        if not files:
            raise FileNotFoundError(f'No TensorBoard events in {source}')
        for path in files:
            digest(path)
            accumulator = EventAccumulator(str(path), size_guidance={'scalars':0}, purge_orphaned_data=False)
            accumulator.Reload()
            for tag in accumulator.Tags().get('scalars', []):
                for event in accumulator.Scalars(tag):
                    total += 1
                    if low < event.step <= high and math.isfinite(event.value):
                        retained += 1
                        key = (tag, event.step)
                        if key not in rows or rows[key]['wall_time'] <= event.wall_time:
                            rows[key] = dict(tag=tag, step=int(event.step), value=float(event.value),
                                             wall_time=float(event.wall_time), source_run=str(source))
        kept_batches = 0
        for path in sorted((source/'logs/curriculum_evaluations').glob('*.yaml')):
            batch = read(path)
            if low < int(batch['step']) <= high:
                digest(path)
                key = (int(batch['phase']), int(batch['step']))
                if key in batches:
                    raise ValueError(f'Duplicate assessment at {key}')
                batches[key] = (path, batch)
                kept_batches += 1
        counts.append(dict(**segment, source_scalar_records=total,
            retained_scalar_records=retained, retained_assessment_batches=kept_batches))
    writer = EventFileWriter(str(output/'logs/tensorboard/retained_history'))
    try:
        for row in sorted(rows.values(), key=lambda r:(r['step'],r['wall_time'],r['tag'])):
            writer.add_event(Event(wall_time=row['wall_time'],step=row['step'],
                summary=Summary(value=[Summary.Value(tag=row['tag'],simple_value=row['value'])])))
        writer.flush()
    finally:
        writer.close()
    for (phase, step), (path, batch) in batches.items():
        shutil.copy2(path, output/'logs/curriculum_evaluations'/f'stage_{phase}_step_{step}.yaml')
    import csv
    with (output/'retained_scalars.csv').open('w',newline='') as stream:
        out = csv.DictWriter(stream,fieldnames=['tag','step','value','wall_time','source_run'])
        out.writeheader();out.writerows(sorted(rows.values(),key=lambda r:(r['tag'],r['step'])))
    position_steps = sorted(step for tag,step in rows if tag == 'rollout/position_error_mean_m')
    result = dict(kind='derived_training_lineage_view',source_run=str(run),segments=counts,
        retained_scalar_points=len(rows),retained_assessment_batches=len(batches),
        first_position_step=position_steps[0],last_position_step=position_steps[-1],
        largest_position_step_gap=max(b-a for a,b in zip(position_steps,position_steps[1:])),
        input_sha256=hashes,
        note='Retained checkpoint ancestry only. Restarts and changed progression criteria must be reported; no data interpolation or uninterrupted-execution claim.')
    (output/'lineage.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'README.md').write_text('# Reconstructed checkpoint history\n\nThis is a derived plotting view, not a new training run. Original records are unmodified.\nOnly intervals retained by the chosen resume checkpoints are included. Retakes after those checkpoints are excluded.\nSee lineage.json and retained_scalars.csv for source hashes and per-point provenance.\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plots',action='store_true',help='Generate training-thesis and standard learning figures.')
    args = parser.parse_args(argv)
    result = build(args.run,args.output)
    print(json.dumps({k:v for k,v in result.items() if k not in {'input_sha256'}},indent=2))
    if args.plots:
        from ..manager import PlotManager
        manager = PlotManager(args.output)
        manager.generate_suite('training_thesis')
        manager.generate_suite('training')
    print(f'Derived history: {args.output.resolve()}')


if __name__=='__main__':main()
