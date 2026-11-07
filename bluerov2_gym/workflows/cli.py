"""Single public entry point. Subcommands load only their own dependencies."""
import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from pathlib import Path
import sys
import yaml

from .configuration import resolve, read_yaml, evaluation_plan
from .artifacts import create_run, write_yaml, update_manifest


@contextmanager
def log_to(path):
    class Tee:
        def __init__(self, stream, log):self.stream,self.log=stream,log
        def write(self,value):self.log.write(value);self.log.flush();return self.stream.write(value)
        def flush(self):self.stream.flush();self.log.flush()
    with Path(path).open('w') as log:
        with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
            yield


def evaluate(args):
    requested,config=resolve(args.config)
    plan=evaluation_plan(config)
    print(yaml.safe_dump(plan,sort_keys=False))
    if plan['mode']!='stationkeeping_single' and not plan['runs']:
        raise ValueError('Evaluation has no enabled controllers/cases/scenarios.')
    for c in config.get('controllers',[]):
        if c.get('enabled',True) and c['name']=='ppo':
            for key in ['model_path','vecnormalize_path']:
                path=c.get('params',{}).get(key)
                if not path or not Path(path).is_file():raise FileNotFoundError(f'Explicit PPO {key} is missing: {path}')
    if plan['mode']=='stationkeeping_single':
        for key in ['model','normalizer']:
            if not config.get(key) or not Path(config[key]).is_file():raise FileNotFoundError(f'Missing {key}: {config.get(key)}')
    run=create_run('evaluation',requested,config,args.output)
    write_yaml(run/'plan.yaml',plan)
    try:
        with log_to(run/'logs/evaluation.log'):
            if plan['mode']=='stationkeeping_single':
                from .evaluation.single_policy import main as single
                if (run/'data').exists():
                    (run/'data').rmdir()  # single evaluator creates its exclusive data directory
                argv=['--model',config['model'],'--normalizer',config['normalizer'],
                      '--dynamics',config['dynamics'],'--output',str(run/'data'),
                      '--duration',str(config.get('duration',100)),'--seed',str(config.get('seed',42)),
                      '--observation-adapter',config.get('observation_adapter','legacy'),
                      '--target',*map(str,config.get('target',[0,0,-2])),
                      '--offset',*map(str,config.get('offset',[.2,-.15,.08]))]
                single(argv,make_plots=False)
            else:
                from .evaluation.runner import ExperimentRunner
                config['output']={'directory':str(run/'data')} if plan['mode']=='stationkeeping' else {
                    'directory_template':str(run/'data'/'{scenario}')}
                write_yaml(run/'config.resolved.yaml',config)
                runner=ExperimentRunner(config)
                runner.run()
                write_yaml(run/"logs/episodes.yaml",runner.records)
        update_manifest(run,status='completed')
    except BaseException as error:
        update_manifest(run,status='failed',error=str(error));raise
    print(f'Run: {run}')


def train(args):
    requested,config=resolve(args.config)
    seeds=args.seeds or [config.get('training',{}).get('seed')]
    if args.device is not None:
        config.setdefault('training', {})['device'] = args.device
    if len(set(seeds))!=len(seeds) or any(seed is not None and seed<0 for seed in seeds):raise ValueError('Seeds must be unique and non-negative.')
    task=config.get('training',{}).get('task','stationkeeping_curriculum')
    if args.resume_run is not None:
        if task != 'trajectory_curriculum' or args.seeds:
            raise ValueError('--resume-run supports one physics/wave curriculum run, without --seeds.')
        config['training']['resume_run'] = str(args.resume_run.expanduser().resolve())
    if task not in {'stationkeeping_curriculum','stationkeeping_no_curriculum','trajectory_tracking','trajectory_curriculum','trajectory_curriculum_legacy'}:
        raise ValueError(f'Unknown training task: {task}')
    if task!='stationkeeping_curriculum' and args.ablation:
        raise ValueError('Curriculum ablations require stationkeeping_curriculum.')
    from .training.curriculum import train_model_with_curriculum
    from .training.mastery import CurriculumEarlyStopped
    for seed in seeds:
        config['training']['seed']=seed
        output=Path(args.output)/f'seed_{seed}' if args.output and len(seeds)>1 else args.output
        run=create_run('training',requested,config,output)
        config['paths']['tensorboard_log']=str(run/'logs/tensorboard')
        config['paths']['model_dir']=str(run/'models')
        write_yaml(run/'config.resolved.yaml',config)
        try:
            with log_to(run/'logs/training.log'):
                if task=='trajectory_curriculum_legacy':
                    from .training.trajectory_curriculum_legacy import train_trajectory_curriculum
                    train_trajectory_curriculum(config)
                elif task=='trajectory_curriculum':
                    from .training.trajectory_curriculum import train_trajectory_curriculum
                    train_trajectory_curriculum(config)
                elif task=='trajectory_tracking':
                    from .training.trajectory import train_trajectory
                    train_trajectory(config)
                elif task=='stationkeeping_no_curriculum':
                    from .training.no_curriculum import train_without_curriculum
                    train_without_curriculum(config)
                else:
                    train_model_with_curriculum(str(run/'config.resolved.yaml'),seed=seed,
                        separate_run=bool(args.seeds),ablation=args.ablation)
            # Trainer is authoritative for merged defaults and effective ablations.
            manifests=list((run/'models').glob('*_config.yaml'))
            if manifests:
                effective=read_yaml(manifests[0])
                write_yaml(run/'config.resolved.yaml',effective['config'])
            update_manifest(run,status='completed',training_seed=seed,ablation=args.ablation)
        except CurriculumEarlyStopped as error:
            update_manifest(run,status='early_stopped',reason=str(error),training_seed=seed)
            print(f'Early stopping: {error}')
        except KeyboardInterrupt:
            update_manifest(run,status='interrupted',training_seed=seed)
            raise
        except BaseException as error:
            update_manifest(run,status='failed',error=str(error));raise
        print(f'Run: {run}')


def plot(args):
    from .plotting import PlotManager
    settings={}
    if args.config:
        path=Path(args.config).resolve();settings=read_yaml(path)
        def paths(value,key=''):
            if isinstance(value,dict):return {k:paths(v,k) for k,v in value.items()}
            if isinstance(value,str) and key in {'input_dir','output_dir','table_dir','log_dir','database','nominal','ftc','representative_output_dir'}:
                return str((path.parent/Path(value).expanduser()).resolve())
            return value
        settings=paths(settings)
        if settings.get('scenarios'):
            settings['scenarios']={k:str((path.parent/v).resolve()) for k,v in settings['scenarios'].items()}
    if args.scenario:settings['scenario']=args.scenario
    if args.repetition_id is not None:settings['repetition_id']=args.repetition_id
    manager=PlotManager(args.run,settings)
    result=manager.generate(args.plots) if args.plots else manager.generate_suite(args.suite or settings.get('suite','article'))
    print(f'Generated {len(result)} figure files.')


def main(argv=None):
    parser=argparse.ArgumentParser(prog='bluerov',description='Train, evaluate and plot BlueROV2 experiments.')
    commands=parser.add_subparsers(dest='command',required=True)
    for name in ['train','evaluate','inspect']:
        p=commands.add_parser(name)
        p.add_argument('--config',required=True,type=Path)
        if name!='inspect':p.add_argument('--output',type=Path,help='Exclusive run directory; defaults to runs/<kind>/<timestamp>.')
        if name=='train':
            p.add_argument('--seeds',type=int,nargs='+')
            p.add_argument('--ablation')
            p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default=None)
            p.add_argument('--resume-run', type=Path, help='Continue a stopped physics/wave curriculum in a new output directory.')
    for name in ['plot','plot-training']:
        p=commands.add_parser(name)
        p.add_argument('--run',type=Path)
        p.add_argument('--config',type=Path)
        p.add_argument('--suite',default='training' if name=='plot-training' else None)
        p.add_argument('--plots',nargs='+')
        p.add_argument('--scenario')
        p.add_argument('--repetition-id',type=int)
    checkpoints=commands.add_parser('evaluate-checkpoints')
    checkpoints.add_argument('--training-run',required=True,type=Path)
    checkpoints.add_argument('--steps',type=int,default=500)
    checkpoints.add_argument('--seed',type=int,default=42)
    checkpoints.add_argument('--stage',type=int,default=-1)
    checkpoints.add_argument('--output',type=Path)
    export_parser=commands.add_parser('export-figures')
    export_parser.add_argument('--run',required=True,type=Path)
    export_parser.add_argument('--destination',required=True,type=Path)
    export_parser.add_argument('--folder',help='Optional subfolder inside run/figures, e.g. article.')
    commands.add_parser('list-plots')
    args=parser.parse_args(argv)
    try:
        if args.command=='train':train(args)
        elif args.command=='evaluate':evaluate(args)
        elif args.command in {'plot','plot-training'}:
            if not args.run and not args.config:parser.error('plot requires --run or --config')
            plot(args)
        elif args.command=='evaluate-checkpoints':
            from .evaluation.checkpoints import evaluate_checkpoints
            evaluate_checkpoints(args.training_run,args.steps,args.seed,args.stage,args.output)
        elif args.command=='export-figures':
            import shutil
            source=(args.run/'figures').resolve()
            if args.folder:
                selection=(source/args.folder).resolve()
                if not selection.is_relative_to(source):raise ValueError('folder must be inside run/figures')
                source=selection
            files=[p for p in source.rglob('*') if p.is_file() and p.suffix in {'.pdf','.png'}]
            if not files:raise ValueError(f'No figures under {source}')
            destination=args.destination.expanduser().resolve()
            for file in files:
                target=destination/file.relative_to(source)
                if target.exists():raise FileExistsError(f'Export would overwrite {target}')
            for file in files:
                target=destination/file.relative_to(source)
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target)
            print(f'Exported {len(files)} figures to {destination}')
        elif args.command=='inspect':
            _,config=resolve(args.config)
            plan=evaluation_plan(config) if 'experiment' in config else config
            print(yaml.safe_dump(plan,sort_keys=False))
        else:
            from .plotting.registry import SUITES
            print(yaml.safe_dump({k:list(v) for k,v in SUITES.items()},sort_keys=False))
    except (ValueError,FileNotFoundError,FileExistsError,RuntimeError) as error:
        parser.exit(2,f'Error: {error}\n')

if __name__=='__main__':main()
