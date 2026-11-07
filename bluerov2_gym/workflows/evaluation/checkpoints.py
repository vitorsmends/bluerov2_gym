"""Retrospective fixed-condition evaluation of saved training checkpoints."""
import csv
import hashlib
from pathlib import Path
import re
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from ..configuration import read_yaml, merge
from ..artifacts import create_run, update_manifest, write_yaml
from .env_utils import make_env


def discover_pairs(run, config):
    folder=Path(run)/'models'
    pairs=[]
    for path in folder.glob('*_checkpoint_*_steps.zip'):
        match=re.fullmatch(r'(.*)_(\d+)_steps',path.stem)
        if not match:continue
        normalizer=folder/f'{match[1]}_vecnormalize_{match[2]}_steps.pkl'
        if not normalizer.is_file():raise FileNotFoundError(normalizer)
        pairs.append((int(match[2]),path,normalizer))
    paths=config['paths']
    model_base=paths['model_name']
    norm_base=Path(paths['vecnormalize_name']).stem
    for path in folder.glob(f'{model_base}*.zip'):
        if '_checkpoint_' in path.stem:continue
        normalizer=folder/(norm_base+path.stem[len(model_base):]+'.pkl')
        if not normalizer.is_file():raise FileNotFoundError(normalizer)
        pairs.append((float('inf'),path,normalizer))
    if not pairs:raise FileNotFoundError(f'No paired checkpoints in {folder}')
    return sorted(pairs,key=lambda item:item[0])


def evaluate_checkpoints(training_run, steps=500, seed=42, stage=-1, output=None):
    if steps<1:raise ValueError('steps must be positive')
    training_run=Path(training_run).expanduser().resolve()
    config=read_yaml(training_run/'config.resolved.yaml')
    if config.get('training', {}).get('task') in {'trajectory_tracking', 'trajectory_curriculum', 'trajectory_curriculum_legacy'}:
        raise ValueError('evaluate-checkpoints uses a stationkeeping curriculum protocol. '
                         'For trajectory-trained agents use the trajectory evaluation configuration.')
    lesson=config['curriculum'][stage]
    environment=config['environment']
    dynamics=environment['dynamics_config']
    sea=merge(config.get('jonswap_defaults',{}),lesson['jonswap'])
    sea['seed']=seed
    protocol=dict(training_run=str(training_run),steps=steps,seed=seed,
                  stage=stage,stage_name=lesson['name'],sea=sea,target=[0.,0.,0.],
                  initial_state='environment reset: all states zero',
                  deterministic=True,normalization_training=False,
                  interpretation='Retrospective checkpoint evaluation, not recorded training error',
                  environment=environment)
    pairs=discover_pairs(training_run,config)
    run=create_run('checkpoint_evaluation',protocol,protocol,output)
    data=run/'data';data.mkdir()
    metrics=[]
    print(f'Checkpoint evaluation: {len(pairs)} policies, {steps} steps each, fixed stage {lesson["name"]}',flush=True)
    try:
        for _,model_path,normalizer_path in pairs:
            env=make_env(dynamics_config=dynamics,env_config=environment.get('config'))
            dummy=DummyVecEnv([lambda:make_env(dynamics_config=dynamics,env_config=environment.get('config'))])
            normalizer=None
            try:
                normalizer=VecNormalize.load(str(normalizer_path),dummy)
                normalizer.training=False;normalizer.norm_reward=False
                model=PPO.load(str(model_path),device='cpu')
                obs,_=env.reset(seed=seed,options={'jonswap_params':sea})
                rows=[];terminated=truncated=False
                for i in range(steps):
                    action,_=model.predict(normalizer.normalize_obs(obs),deterministic=True)
                    obs,reward,terminated,truncated,info=env.step(np.asarray(action).reshape(6))
                    state=env.unwrapped.state
                    error=float(info['metrics/position_error_euclidean'])
                    rows.append([(i+1)*env.unwrapped.dynamics.dt,error,reward,
                                 *[state[key] for key in ['x','y','z','roll','pitch','yaw']]])
                    if terminated or truncated:break
                values=np.asarray(rows)
                completed_steps=int(model.num_timesteps)
                csv_path=data/f'checkpoint_{completed_steps}.csv'
                with csv_path.open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(['time','position_error_m','reward','x','y','z','roll','pitch','yaw']);writer.writerows(rows)
                radius=environment.get('config',{}).get('reward',{}).get('success',{}).get('radius',.2)
                record=dict(training_steps=completed_steps,position_error_mean_m=float(values[:,1].mean()),
                    position_error_rmse_m=float(np.sqrt(np.mean(values[:,1]**2))),
                    final_error_m=float(values[-1,1]),reward_mean_per_step=float(values[:,2].mean()),
                    return_raw=float(values[:,2].sum()),success_fraction=float(np.mean(values[:,1]<radius)),
                    evaluated_steps=len(rows),duration_s=float(values[-1,0]),
                    terminated=bool(terminated),truncated=bool(truncated),
                    model=str(model_path),normalizer=str(normalizer_path),
                    model_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                    normalizer_sha256=hashlib.sha256(normalizer_path.read_bytes()).hexdigest())
                metrics.append(record)
                print(f'{completed_steps:>8} steps | error RMSE {record["position_error_rmse_m"]:.4f} m | success {record["success_fraction"]:.1%} | {len(rows)}/{steps} evaluation steps',flush=True)
            finally:
                env.close()
                (normalizer if normalizer is not None else dummy).close()
        with (data/'checkpoint_metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
        write_yaml(run/'protocol.yaml',protocol)
        update_manifest(run,status='completed',checkpoint_count=len(metrics))
    except BaseException as error:
        update_manifest(run,status='failed',error=str(error));raise
    print(f'Run: {run}',flush=True)
    return run
