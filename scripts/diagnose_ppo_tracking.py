"""Controlled PPO tracking ablations; leaves production controllers unchanged."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import yaml

from bluerov2_gym.workflows.configuration import resolve, read_yaml, merge
from bluerov2_gym.workflows.paths import artifact_path
from bluerov2_gym.workflows.evaluation.env_utils import make_env, obs_to_state, wrap_angle, build_tracking_observation
from bluerov2_gym.workflows.evaluation.controllers.ppo import PPOController
from bluerov2_gym.workflows.evaluation.trajectories import create_trajectory


def rotation(phi,theta,psi):
    cp,sp=np.cos(phi),np.sin(phi)
    ct,st=np.cos(theta),np.sin(theta)
    cy,sy=np.cos(psi),np.sin(psi)
    return np.array([[cy*ct,cy*st*sp-sy*cp,cy*st*cp+sy*sp],
                     [sy*ct,sy*st*sp+cy*cp,sy*st*cp-cy*sp],[-st,ct*sp,ct*cp]])


def observation(obs,ref,variant,yaw_rate=0.):
    if variant=='production':return build_tracking_observation(obs,ref)
    virtual={k:np.asarray(v,dtype=np.float32).copy() for k,v in obs.items()}
    if variant=='raw':return virtual
    state=obs_to_state(obs).astype(float)
    error=state[:3]-ref[:3]
    if variant in ['body_relative','reference_pose','reference_pose_rate']:
        angle=state[5] if variant=='body_relative' else ref[5]
        error=rotation(0.,0.,angle).T@error
    for key,value in zip(['x','y','z'],error):virtual[key]=np.array([value],np.float32)
    if variant=='world_surge':
        virtual['u']=np.array([state[6]-ref[6]],np.float32)
    elif variant!='world_absolute':
        rot=rotation(*state[3:6])
        velocity=state[6:9]-rot.T@ref[6:9]
        for key,value in zip(['u','v','w'],velocity):virtual[key]=np.array([value],np.float32)
    if variant in ['world_relative_yaw','reference_pose','reference_pose_rate']:
        virtual['yaw']=np.array([wrap_angle(state[5]-ref[5])],np.float32)
    if variant=='reference_pose_rate':
        angular=state[9:12]-rotation(*state[3:6]).T@np.array([0.,0.,yaw_rate])
        for key,value in zip(['p','q','r'],angular):virtual[key]=np.array([value],np.float32)
    return virtual


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/evaluation/trajectory_comparison.yaml')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--campaign',action='store_true',help='Use the configured trajectories and paired repetition seeds')
    parser.add_argument('--adapters',nargs='+',choices=['production','world_surge','world_absolute','world_relative','body_relative','world_relative_yaw','reference_pose','reference_pose_rate'])
    args=parser.parse_args()
    _,config=resolve(args.config)
    ppo=next(c for c in config['controllers'] if c['name']=='ppo')
    policy=PPOController(**ppo['params'])
    environment=config['environment']
    trained=read_yaml(Path(environment['training_run'])/'config.resolved.yaml')
    final_sea=merge(trained['jonswap_defaults'],trained['curriculum'][-1]['jonswap'])
    sea_name=config['disturbance'].get('default_scenario','severe')
    eval_sea=config['disturbance']['scenarios'][sea_name]
    no_sea=dict(eval_sea,scale=0.,noise_std=0.,enable_wave_force=False)
    variants=['production','world_surge','world_absolute','world_relative','body_relative',
              'world_relative_yaw','reference_pose','reference_pose_rate']
    cases=[('circle',sea_name,v) for v in variants]
    cases += [('origin','training_final','raw'),('origin','training_final','production'),
              ('origin',sea_name,'raw'),('origin',sea_name,'production'),
              ('static',sea_name,'production'),('static',sea_name,'world_relative'),
              ('circle','training_final','production'),('circle','training_final','world_relative'),
              ('circle','no_waves','production'),('circle','no_waves','world_relative')]
    if args.campaign:
        cases=[(item['name'],sea_name,adapter,args.seed+rep)
               for item in config['trajectories'] if item.get('enabled',True)
               for rep in range(config['experiment'].get('repetitions',1))
               for adapter in (args.adapters or ['production','body_relative'])]
    else:
        cases=[(*case,args.seed) for case in cases]
    output=args.output or artifact_path('diagnostics','ppo_tracking',datetime.now(ZoneInfo('America/Bahia')).strftime('%Y%m%d-%H%M%S'))
    output=output.resolve();output.mkdir(parents=True,exist_ok=False)
    results=[]
    counts={k:v.count for k,v in policy.vec_env.obs_rms.items()}
    try:
        for task,scenario,variant,seed in cases:
            env=make_env(dynamics_config=environment['dynamics_config'],env_config=environment.get('config'))
            try:
                sea=dict({'training_final':final_sea,'no_waves':no_sea}.get(scenario,eval_sea),seed=seed)
                obs,_=env.reset(seed=seed,options={'jonswap_params':sea})
                params=next((item.get('params',{}) for item in config.get('trajectories',[]) if item['name']==task),{})
                trajectory=create_trajectory(task if task not in ['origin','static'] else 'circle',params)
                dt=env.unwrapped.dynamics.dt
                rows=[];clipping=[];terminated=truncated=False
                for i in range(args.steps):
                    t=i*dt
                    if task not in ['origin','static']:
                        ref=trajectory.get_reference(t)
                        yaw_rate=wrap_angle(trajectory.get_reference(t+.01)[5]-trajectory.get_reference(max(0.,t-.01))[5])/(.02 if t>0 else .01)
                    else:
                        ref=np.zeros(12);ref[:3]=[1.,0.,-.6] if task=='static' else [0.,0.,0.]
                        yaw_rate=0.
                    state=obs_to_state(obs).astype(float)
                    virtual=observation(obs,ref,variant,yaw_rate)
                    normalized=policy.vec_env.normalize_obs(virtual)
                    clipping.append(np.mean([abs(float(v[0]))>=policy.vec_env.clip_obs-.001 for v in normalized.values()]))
                    action,_=policy.model.predict(normalized,deterministic=True)
                    action=np.clip(np.asarray(action).reshape(6),-40,40)
                    error=np.linalg.norm(state[:3]-ref[:3])
                    velocity_error=np.linalg.norm(state[6:9]-rotation(*state[3:6]).T@ref[6:9])
                    rows.append([t,*state,*ref,error,wrap_angle(state[5]-ref[5]),velocity_error,*action])
                    obs,_,terminated,truncated,_=env.step(action)
                    if terminated or truncated:break
                data=np.asarray(rows)
                header=['time',*['x','y','z','roll','pitch','yaw','u','v','w','p','q','r'],
                        *[f'{s}_ref' for s in ['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']],
                        'position_error_m','yaw_error_rad','velocity_error_body',*[f'T{i}' for i in range(1,7)]]
                with (output/f'{task}_{scenario}_{variant}_seed{seed}.csv').open('w',newline='') as stream:
                    writer=csv.writer(stream);writer.writerow(header);writer.writerows(rows)
                metric=dict(task=task,scenario=scenario,adapter=variant,seed=seed,steps=len(data),
                    position_rmse_m=float(np.sqrt(np.mean(data[:,25]**2))),
                    tail_position_rmse_m=float(np.sqrt(np.mean(data[len(data)//2:,25]**2))),
                    yaw_rmse_deg=float(np.rad2deg(np.sqrt(np.mean(data[:,26]**2)))),
                    velocity_rmse_m_s=float(np.sqrt(np.mean(data[:,27]**2))),
                    normalized_clip_fraction=float(np.mean(clipping)),terminated=bool(terminated),truncated=bool(truncated))
                results.append(metric)
                print(f'{task:7s} {scenario:14s} {variant:20s} seed={seed} RMSE={metric["position_rmse_m"]:.4f}m yaw={metric["yaw_rmse_deg"]:.1f}deg clip={metric["normalized_clip_fraction"]:.1%}',flush=True)
            finally:env.close()
        with (output/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
        assert counts=={k:v.count for k,v in policy.vec_env.obs_rms.items()}
        protocol=dict(config=config,steps=args.steps,seed=args.seed,variants=variants,campaign=args.campaign,
                      normalizer_frozen=not policy.vec_env.training,
                      model_sha256=hashlib.sha256(policy.model_path.read_bytes()).hexdigest(),
                      normalizer_sha256=hashlib.sha256(policy.vecnormalize_path.read_bytes()).hexdigest(),
                      interpretation='Controlled single-seed ablations, not a replacement benchmark')
        (output/'protocol.yaml').write_text(yaml.safe_dump(protocol,sort_keys=False))
        print(f'Output: {output}',flush=True)
    finally:policy.vec_env.close()

if __name__=='__main__':main()
