"""Explore reference size with frozen article PPO; speed preserved by period scaling.

Single W1 realization, no retraining. Keeps article data and figures untouched.
"""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'artifacts/evaluation/article_ppo_three_paths/benchmark/config.resolved.yaml')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/trajectory_scale_exploration')
    parser.add_argument('--scales',type=float,nargs='+',default=[1,2,4])
    args=parser.parse_args()
    if any(s<=0 for s in args.scales) or len(set(args.scales))!=len(args.scales):parser.error('Scales must be positive and distinct')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    source=yaml.safe_load(args.source.read_text());records=[];frames={}
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
    lengths={'figure_eight_depth':'a','square':'side','circle':'radius'}
    labels={'figure_eight_depth':'Figure eight','square':'Square','circle':'Circle'}
    cases=[t for t in source['trajectories'] if t.get('enabled',True)]
    for scale in args.scales:
        cfg=copy.deepcopy(source)
        cfg['controllers']=[c for c in cfg['controllers'] if c['name']=='ppo']
        cfg['disturbance']['scenario_names']=['walker_w1'];cfg['disturbance']['default_scenario']='walker_w1'
        cfg['experiment']['repetitions']=1
        cfg['experiment']['steps']=round(source['experiment']['steps']*scale)
        for case in cfg['trajectories']:
            case['params'][lengths[case['name']]]*=scale
            case['params']['period']*=scale
        name=f'scale_{scale:g}';path=output/f'{name}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
        with (output/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable,'-m','bluerov2_gym.workflows.cli','evaluate','--config',str(path),'--output',str(output/name)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for case in cfg['trajectories']:
            df=pd.read_csv(output/name/'data/walker_w1'/f"ppo_{case['name']}.csv")
            frames[scale,case['name']]=df
            length=case['params'][lengths[case['name']]];error=df.tracking_error_m.to_numpy()
            post=df.time>=10*scale
            records.append(dict(scale=scale,trajectory=case['name'],length_m=length,period_s=case['params']['period'],duration_s=cfg['experiment']['steps']*cfg['experiment']['dt'],rmse_m=np.sqrt(np.mean(error**2)),rmse_over_length=np.sqrt(np.mean(error**2))/length,post_transient_rmse_m=np.sqrt(np.mean(error[post]**2)),yaw_mae_deg=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error)))).mean()))
        print(f'Completed {scale:g}x',flush=True)
    metrics=pd.DataFrame(records);metrics.to_csv(output/'metrics.csv',index=False)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        fig,axes=plt.subplots(len(args.scales),len(cases),figsize=(10,3.1*len(args.scales)),squeeze=False,layout='constrained')
        for i,scale in enumerate(args.scales):
            for j,case in enumerate(cases):
                name=case['name'];ax=axes[i,j];df=frames[scale,name]
                ax.plot(df.x_ref,df.y_ref,'k--',lw=1,label='Reference');ax.plot(df.x,df.y,color='#59A14F',lw=.9,label='PPO')
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2)
                row=metrics[(metrics.scale==scale)&(metrics.trajectory==name)].iloc[0]
                ax.set_title(f"{labels[name]} · {scale:g}×\nRMSE {row.rmse_m:.3f} m; RMSE/L {100*row.rmse_over_length:.1f}%",fontsize=11)
                if i==len(args.scales)-1:ax.set_xlabel('X [m]')
                if j==0:ax.set_ylabel('Y [m]')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=2,frameon=False)
        for ext in ['pdf','png']:fig.savefig(output/f'trajectory_scales_xy.{ext}',dpi=220)
        plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
        for case in cases:
            data=metrics[metrics.trajectory==case['name']]
            axes[0].plot(data.scale,data.rmse_m,'o-',label=labels[case['name']])
            axes[1].plot(data.scale,100*data.rmse_over_length,'o-',label=labels[case['name']])
        for ax in axes:ax.set_xlabel('Spatial / period scale');ax.set_xticks(args.scales);ax.grid(alpha=.2)
        axes[0].set_ylabel('3-D position RMSE [m]');axes[1].set_ylabel('Position RMSE / reference length [%]');axes[0].legend(frameon=False)
        for ext in ['pdf','png']:fig.savefig(output/f'trajectory_scale_errors.{ext}',dpi=220)
        plt.close(fig)
    (output/'protocol.json').write_text(json.dumps(dict(source=str(args.source.resolve()),scales=args.scales,sea='W1',seed=42,controller='Frozen article PPO',protocol='Horizontal dimensions, periods and duration multiplied by scale; nominal horizontal velocity and number of reference cycles preserved. Initial robot position remains fixed, so initial reference offsets grow for square and circle. Full-horizon metrics include acquisition; post-transient metric excludes first 10*scale seconds. RMSE is 3-D; L is eight amplitude a, square side, or circle radius. XY panels omit depth, so they cannot alone establish 3-D tracking improvement. One realization per scale, not a statistical generalization test.'),indent=2))
    print(metrics.to_string(index=False));print(output)

if __name__=='__main__':main()
