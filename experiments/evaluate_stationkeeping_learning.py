"""Compare raw PPO training observations against SMC for origin stabilization."""
from pathlib import Path
import argparse
import hashlib
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from experiments.controllers.ppo import PPOController
from experiments.controllers.smc import SMCController
from experiments.env_utils import obs_to_state
from experiments.references import StationKeepingReference
from bluerov2_gym.envs.core.config_utils import load_yaml
from bluerov2_gym.envs.core.dynamics import Dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT / 'training_runs/ppo_0436ce3_seed42_ep5000')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    base = args.run.resolve()
    output = args.output or base / 'stationkeeping_learning'
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    output.mkdir(parents=True)
    model = base / 'models/bluerov_ppo_physics_v2_seed_42.zip'
    normalizer = base / 'models/bluerov_vecnormalize_physics_v2_seed_42.pkl'
    ppo = PPOController(model_path=str(model), vecnormalize_path=str(normalizer))
    physics = load_yaml(ROOT / 'config/dynamics_config.yaml')
    scenarios = load_yaml(ROOT / 'experiments/ocean_environment.yaml')['scenarios']
    ref = np.zeros(12)
    reference = StationKeepingReference(np.zeros(3), 0.)
    summaries, frames = [], []
    for initial in ('origin', 'offset'):
        for sea in ('moderate', 'severe'):
            for seed in (42, 142, 242):
                for name in ('ppo', 'smc'):
                    env = gym.make('BlueRov-v0', render_mode=None,
                                   env_config={'dt': .1, 'jonswap': dict(scenarios[sea], seed=seed)},
                                   dynamics_config=physics, max_episode_steps=1000)
                    try:
                        obs, _ = env.reset(seed=seed)
                        if initial == 'offset':
                            env.unwrapped.state.update(x=.2, y=-.15, z=.08, yaw=float(np.deg2rad(10)))
                            obs = env.unwrapped._get_obs()
                        smc = SMCController(dynamics=env.unwrapped.dynamics, dt=.1, reference_provider=reference)
                        smc.reset()
                        rows = []
                        for k in range(1000):
                            state = obs_to_state(obs)
                            # Raw observations reproduce the task actually trained.
                            # No tracking adapter, yaw rotation of position, or target substitution.
                            if name == 'ppo':
                                action, _ = ppo.model.predict(ppo.vec_env.normalize_obs(obs), deterministic=True)
                            else:
                                action = smc.get_action(obs=obs, state=state, reference=ref, t=k*.1)
                            action = np.clip(np.asarray(action).reshape(6), -40., 40.)
                            rows.append(dict(initial=initial, scenario=sea, seed=seed, controller=name,
                                             time=k*.1, x=state[0], y=state[1], z=state[2],
                                             yaw=state[5], error_m=np.linalg.norm(state[:3]),
                                             saturated=np.mean(np.abs(action) >=39.99)))
                            obs, _, terminated, truncated, _ = env.step(action)
                            if terminated or truncated:
                                break
                        frame = pd.DataFrame(rows)
                        frames.append(frame)
                        summaries.append(dict(initial=initial, scenario=sea, seed=seed, controller=name,
                                              steps=len(frame), rmse_m=np.sqrt(np.mean(frame.error_m**2)),
                                              final20s_mean_error_m=frame.loc[frame.time>=80, 'error_m'].mean(),
                                              within_20cm_pct=100*np.mean(frame.error_m<.2),
                                              yaw_rmse_deg=np.rad2deg(np.sqrt(np.mean(frame.yaw**2))),
                                              saturation_pct=100*frame.saturated.mean()))
                    finally:
                        env.close()
    ppo.vec_env.close()
    data = pd.concat(frames, ignore_index=True)
    summary = pd.DataFrame(summaries)
    data.to_csv(output / 'timeseries.csv', index=False)
    summary.to_csv(output / 'per_run.csv', index=False)
    metrics = ['rmse_m','final20s_mean_error_m','within_20cm_pct','yaw_rmse_deg','saturation_pct']
    aggregate = summary.groupby(['initial','scenario','controller'])[metrics].mean().reset_index()
    aggregate.to_csv(output / 'summary.csv', index=False)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True)
    for i, initial in enumerate(('origin','offset')):
        for j, sea in enumerate(('moderate','severe')):
            ax=axes[i,j]
            for name,color in [('ppo','tab:blue'),('smc','tab:orange')]:
                subset=data[(data.initial==initial)&(data.scenario==sea)&(data.controller==name)]
                for _,case in subset.groupby('seed'):
                    ax.plot(case.time,case.error_m,color=color,alpha=.15,lw=.8)
                mean=subset.groupby('time').error_m.mean()
                ax.plot(mean.index,mean.values,label=name.upper(),color=color,lw=1.6)
            ax.axhline(.2,color='gray',ls='--',lw=.8,label='0.20 m')
            ax.set_title(f'{sea.capitalize()} | {"Initial state zero" if initial=="origin" else "Initial position offset 0.262 m, yaw 10 deg"}')
            ax.set_ylabel('Position error (m)')
            ax.grid(alpha=.2)
            if i==1: ax.set_xlabel('Time (s)')
    axes[0,0].legend()
    fig.suptitle('Origin station keeping — physics v2, PPO 5000-step episodes\nMean of wave seeds 42, 142, 242; individual runs shown faintly')
    fig.tight_layout()
    fig.savefig(output/'comparison.png',dpi=180)
    fig.savefig(output/'comparison.pdf')
    manifest=dict(dynamics_version=Dynamics.MODEL_VERSION, physical_config=physics,
                  scenarios={k:scenarios[k] for k in ('moderate','severe')}, seeds=[42,142,242],
                  target_position=[0,0,0],target_yaw=0,dt=.1,steps=1000,
                  ppo_observations='Raw training observations; frozen normalization; deterministic actions',
                  limitations='SMC default parameters include its own internal limits; common thruster clip +/-40 N',
                  artifacts={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (model,normalizer)})
    (output/'manifest.yaml').write_text(yaml.safe_dump(manifest,sort_keys=False))
    print(aggregate.to_string(index=False))
    print(f'Artifacts: {output}')


if __name__=='__main__':
    main()
