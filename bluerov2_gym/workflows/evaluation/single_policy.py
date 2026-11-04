"""Evaluate an existing PPO at one fixed setpoint; never updates model or stats."""
import argparse
import csv
import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import yaml

from bluerov2_gym.workflows.evaluation.controllers.ppo import PPOController
from bluerov2_gym.workflows.evaluation.env_utils import make_env, obs_to_state, wrap_angle
from bluerov2_gym.envs.core.config_utils import load_yaml


def main(argv=None, make_plots=True):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='bluerov_ppo.zip')
    parser.add_argument('--normalizer', default='bluerov_vec_normalize.pkl')
    parser.add_argument('--dynamics', default='configs/dynamics/dynamics_config.yaml')
    parser.add_argument('--target', type=float, nargs=3, default=[0.,0.,-2.])
    parser.add_argument('--offset', type=float, nargs=3, default=[.2,-.15,.08])
    parser.add_argument('--duration', type=float, default=100.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='results/stationkeeping_single_corrected')
    args = parser.parse_args(argv)
    config = load_yaml(args.dynamics)
    config['mechanics_model'] = 'corrected'
    config.setdefault('jonswap', {})['seed'] = args.seed
    dt = float(config.get('dt', .1))
    if args.duration <= 0 or dt <= 0:
        parser.error('duration and dt must be positive')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    env = make_env(dynamics_config=config)
    controller = None
    rows = []
    target = np.array(args.target)
    reference = np.r_[target,np.zeros(9)]
    terminated = truncated = False
    try:
        controller = PPOController(model_path=args.model, vecnormalize_path=args.normalizer,
                                   dynamics_config_path=args.dynamics)
        obs, _ = env.reset(seed=args.seed)
        for key, value in zip(['x','y','z'], target+np.array(args.offset)):
            env.unwrapped.state[key] = float(value)
        env.unwrapped.state['yaw'] = float(np.deg2rad(10.))
        obs = env.unwrapped._get_obs()
        def record(t, action):
            state = obs_to_state(obs).astype(float)
            errors = state[:3]-target
            angles = np.array([wrap_angle(v) for v in state[3:6]])
            rows.append([t,*state,*errors,float(np.linalg.norm(errors)),*angles,*action])
        record(0., np.zeros(6))
        for step in range(round(args.duration/dt)):
            state = obs_to_state(obs)
            action = controller.get_action(obs,state,reference,step*dt)
            obs, _, terminated, truncated, _ = env.step(action)
            record((step+1)*dt, action)
            if terminated or truncated:
                break
        data = np.array(rows)
        error = data[:,13:16]
        norm = data[:,16]
        tail = data[:,0] >= max(0.,data[-1,0]-20.)
        summary = dict(duration_s=float(data[-1,0]), requested_duration_s=args.duration,
                       terminated=bool(terminated),truncated=bool(truncated),
                       position_rmse_m=float(np.sqrt(np.mean(norm**2))),
                       axis_rmse_m=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                       final_position_error_m=float(norm[-1]),
                       final_yaw_error_deg=float(np.rad2deg(data[-1,19])),
                       final_20s_position_rmse_m=float(np.sqrt(np.mean(norm[tail]**2))))
        fields = ['time','x','y','z','roll','pitch','yaw','u','v','w','p','q','r',
                  'error_x','error_y','error_z','position_error_norm',
                  'error_roll','error_pitch','error_yaw',*[f'T{i}' for i in range(1,7)]]
        with (output/'stationkeeping.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(fields);writer.writerows(rows)
        hashes = {name:dict(path=str(Path(path).resolve()),
                            sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                  for name,path in [('model',args.model),('normalizer',args.normalizer)]}
        manifest = dict(artifacts=hashes,target_m=target.tolist(),yaw_target_deg=0.,
                        initial_offset_m=args.offset,initial_yaw_deg=10.,seed=args.seed,
                        deterministic=True,normalization_training=False,
                        dynamics_config=config,
                        resolved_disturbance_model=env.unwrapped.dynamics.disturbance_model,
                        summary=summary)
        (output/'evaluation.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
        if make_plots:
            from ..plotting.stationkeeping.single_policy import plot_errors
            plot_errors(data, target, env.unwrapped.dynamics.disturbance_model['preset'],
                        terminated, truncated, output)
        print(yaml.safe_dump(summary,sort_keys=False))
        print(f'Results: {output.resolve()}')
    finally:
        env.close()
        if controller is not None:controller.vec_env.close()


if __name__=='__main__':
    main()
