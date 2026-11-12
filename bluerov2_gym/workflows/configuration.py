"""Config-relative path resolution, with no simulator/model imports."""
from copy import deepcopy
from pathlib import Path
import yaml


def read_yaml(path):
    with Path(path).open(encoding='utf-8') as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    return value


def merge(base, overrides):
    result = deepcopy(base)
    for key,value in (overrides or {}).items():
        result[key] = merge(result[key],value) if isinstance(value,dict) and isinstance(result.get(key),dict) else deepcopy(value)
    return result


def resolve(path):
    path = Path(path).expanduser().resolve()
    requested = read_yaml(path)
    config = deepcopy(requested)
    def absolute(value):
        value = Path(value).expanduser()
        return str(value.resolve() if value.is_absolute() else (path.parent/value).resolve())
    if config.get('training', {}).get('task') == 'trajectory_finetune':
        source = Path(absolute(config['finetuning']['source_run']))
        inherited = read_yaml(source/'config.resolved.yaml')
        config = merge({key: inherited[key] for key in ['environment', 'ppo', 'vecnormalize', 'reference']
                        if key in inherited}, config)
        config['finetuning']['source_run'] = str(source)
    environment = config.setdefault('environment', {})
    for key in ['dynamics_config_file', 'env_config_file']:
        if environment.get(key):
            file = absolute(environment.pop(key))
            target = 'dynamics_config' if key == 'dynamics_config_file' else 'config'
            environment[target] = merge(read_yaml(file), environment.get(target,{}))
    if isinstance(environment.get('dynamics_config'), str):
        environment['dynamics_config'] = read_yaml(absolute(environment['dynamics_config']))
    if environment.get('training_run'):
        run = Path(absolute(environment['training_run']))
        trained = read_yaml(run/'config.resolved.yaml').get('environment', {})
        if not trained.get('dynamics_config'):
            raise ValueError(f'Training run has no resolved dynamics: {run}')
        inherited = merge(trained, environment)
        if inherited['dynamics_config'] != trained['dynamics_config']:
            raise ValueError('Evaluation dynamics differ from training_run. Remove training_run only for an intentional ablation.')
        if inherited.get('config', {}) != trained.get('config', {}):
            raise ValueError('Evaluation environment config differs from training_run.')
        inherited['training_run'] = str(run)
        config['environment'] = inherited
        if trained.get('yaw_tracking', {}).get('relative_observation', False):
            for controller in config.get('controllers', []):
                if controller.get('name') == 'ppo':
                    controller.setdefault('params', {})['observation_adapter'] = 'legacy_yaw_error'
    disturbance = config.get('disturbance', {})
    if disturbance.get('config_path'):
        external = read_yaml(absolute(disturbance.pop('config_path')))
        for key,value in external.items():disturbance.setdefault(key,value)
    for item in config.get('controllers', []):
        params = item.get('params',{})
        for key in ['model_path','vecnormalize_path','env_config_path','dynamics_config_path']:
            if params.get(key):params[key] = absolute(params[key])
    for key in ['model','normalizer','dynamics','log_dir','output_dir']:
        if config.get(key):config[key] = absolute(config[key])
    for key,value in config.get('paths',{}).items():
        if key in ['tensorboard_log','model_dir']:config['paths'][key] = absolute(value)
    return requested,config


def evaluation_plan(config):
    experiment = config.get('experiment',{})
    mode = experiment.get('mode','stationkeeping_single')
    controllers = [item for item in config.get('controllers',[]) if item.get('enabled',True)]
    disturbance = config.get('disturbance',{})
    scenarios = disturbance.get('scenario_names')
    if scenarios is None:
        default = disturbance.get('default_scenario')
        scenarios = [default] if default and mode != 'stationkeeping' else list(disturbance.get('scenarios',{}))
    if mode == 'stationkeeping':
        # Existing stationkeeping runner deliberately iterates every configured scenario.
        scenarios = list(disturbance.get('scenarios',{}))
        cases = config.get('stationkeeping',{}).get('targets',[])
    else:
        cases = [t for t in config.get('trajectories',[]) if t.get('enabled',True)]
    if mode == 'stationkeeping_single':
        from bluerov2_gym.envs.core.dynamics import Dynamics
        dyn=read_yaml(config['dynamics']) if config.get('dynamics') else {}
        return dict(mode=mode,model=config.get('model'),normalizer=config.get('normalizer'),
                    observation_adapter=config.get('observation_adapter','legacy'),
                    target=config.get('target',[0,0,-2]),offset=config.get('offset',[.2,-.15,.08]),
                    duration_s=config.get('duration',100),seed=config.get('seed',42),runs=1,
                    mechanics_model='corrected',dynamics=config.get('dynamics'),dt=dyn.get('dt',.1),
                    disturbance_model=Dynamics.resolve_disturbance_model(dyn.get('disturbance_model')))
    dyn=config.get('environment',{}).get('dynamics_config',{})
    from bluerov2_gym.envs.core.dynamics import Dynamics
    return dict(mode=mode,controllers=controllers,scenarios=scenarios,cases=cases,
                training_run=config.get('environment',{}).get('training_run'),
                repetitions=experiment.get('repetitions',5),seed=experiment.get('seed',42),
                dt=experiment.get('dt',.1),integration_dt=dyn.get('dt',.1),steps=experiment.get('steps',1000),
                scenario_seeds={name:disturbance.get('scenarios',{}).get(name,{}).get('seed',42) for name in scenarios},
                runs=len(controllers)*len(scenarios)*len(cases)*experiment.get('repetitions',5),
                mechanics_model=dyn.get('mechanics_model','corrected'),
                disturbance_model=Dynamics.resolve_disturbance_model(dyn.get('disturbance_model')))
