"""Evaluate NMPC/PID/SMC and compare against existing frozen-PPO scale runs."""
from pathlib import Path
import argparse,copy,json,os,subprocess,sys,hashlib
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
COLORS={'PPO':'#59A14F','NMPC':'#E15759','PID':'#4E79A7','SMC':'#B07AA1'}
LABELS={'figure_eight_depth':'Figure eight','square':'Square','circle':'Circle'}
LENGTHS={'figure_eight_depth':'a','square':'side','circle':'radius'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ppo-run',type=Path,default=ROOT/'artifacts/evaluation/trajectory_scale_exploration')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/trajectory_scale_controllers')
    p.add_argument('--plot-only',action='store_true')
    a=p.parse_args();source=a.ppo_run.resolve();out=a.output.resolve();scales=[1,2,4]
    if not a.plot_only:
        out.mkdir(parents=True,exist_ok=False)
        base=yaml.safe_load((ROOT/'artifacts/evaluation/article_ppo_three_paths/benchmark/config.resolved.yaml').read_text())
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MPLBACKEND='Agg')
        tasks=[]
        try:
            for scale in scales:
                cfg=yaml.safe_load((source/f'scale_{scale}.yaml').read_text())
                cfg['controllers']=[copy.deepcopy(c) for c in base['controllers'] if c['name'] in {'nmpc','pid','smc'}]
                path=out/f'scale_{scale}.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
                log=(out/f'scale_{scale}.log').open('w')
                process=subprocess.Popen([sys.executable,'-m','bluerov2_gym.workflows.cli','evaluate','--config',str(path),'--output',str(out/f'scale_{scale}')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                tasks.append((scale,process,log))
            print('Started paired W1 evaluations: 3 scales × 3 trajectories × NMPC/PID/SMC.',flush=True)
            for scale,process,log in tasks:
                code=process.wait();log.close()
                if code:raise RuntimeError(f'Scale {scale} failed; see its log')
                print(f'Completed scale {scale}x',flush=True)
        except BaseException:
            for _,process,log in tasks:
                if process.poll() is None:process.terminate()
                log.close()
            raise
    frames={};rows=[];hashes={}
    for scale in scales:
        cfg=yaml.safe_load((source/f'scale_{scale}.yaml').read_text())
        actual=yaml.safe_load((out/f'scale_{scale}/config.resolved.yaml').read_text())
        assert cfg['environment']==actual['environment'] and cfg['experiment']==actual['experiment'] and cfg['disturbance']==actual['disturbance'] and cfg['trajectories']==actual['trajectories']
        assert yaml.safe_load((out/f'scale_{scale}/manifest.yaml').read_text())['status']=='completed'
        for case in cfg['trajectories']:
            name=case['name'];length=case['params'][LENGTHS[name]];paired=None
            for ctrl in COLORS:
                folder=source if ctrl=='PPO' else out
                path=folder/f'scale_{scale}/data/walker_w1/{ctrl.lower()}_{name}.csv';df=pd.read_csv(path)
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                assert len(df)==cfg['experiment']['steps'] and set(df.seed)=={42} and df.repetition_id.nunique()==1
                columns=['time','seed','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','noise_std']+[c for c in df if c.endswith('_ref')]
                if paired is not None:
                    np.testing.assert_allclose(df[columns],paired[columns],rtol=0,atol=1e-10)
                    states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                    np.testing.assert_allclose(df[states].iloc[0],paired[states].iloc[0],rtol=0,atol=1e-10)
                paired=df;frames[scale,name,ctrl]=df
                err=df.tracking_error_m.to_numpy();post=df.time>=10*scale
                rows.append(dict(scale=scale,trajectory=name,controller=ctrl,length_m=length,rmse_m=np.sqrt(np.mean(err**2)),rmse_over_length=np.sqrt(np.mean(err**2))/length,post_transient_rmse_m=np.sqrt(np.mean(err[post]**2)),yaw_mae_deg=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error)))).mean()))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics.csv',index=False)
    with plt.rc_context({'font.family':'serif','font.size':11,'pdf.fonttype':42}):
        def save(fig,name):
            for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=220)
            plt.close(fig)
        for ctrls,suffix in [(list(COLORS),'all'),(['PPO','NMPC'],'ppo_nmpc')]:
            fig,axes=plt.subplots(3,3,figsize=(11,9),layout='constrained')
            for i,scale in enumerate(scales):
                for j,name in enumerate(LABELS):
                    ax=axes[i,j];ref=frames[scale,name,'PPO'];ax.plot(ref.x_ref,ref.y_ref,'k--',lw=1,label='Reference')
                    for ctrl in ctrls:
                        df=frames[scale,name,ctrl];ax.plot(df.x,df.y,color=COLORS[ctrl],lw=.9,label=ctrl)
                    ax.set_aspect('equal',adjustable='box');ax.set_title(f'{LABELS[name]} · {scale}×');ax.grid(alpha=.2)
                    if i==2:ax.set_xlabel('X [m]')
                    if j==0:ax.set_ylabel('Y [m]')
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=len(ctrls)+1,frameon=False)
            save(fig,'trajectory_scales_xy_'+suffix)
        for scale in scales:
            fig=plt.figure(figsize=(11,4.5))
            for j,name in enumerate(LABELS,1):
                ax=fig.add_subplot(1,3,j,projection='3d');ref=frames[scale,name,'PPO']
                ax.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1,label='Reference')
                for ctrl in ['PPO','NMPC']:
                    df=frames[scale,name,ctrl];ax.plot(df.x,df.y,df.z,color=COLORS[ctrl],lw=.9,label=ctrl)
                ax.set_title(LABELS[name],fontsize=11);ax.set_xlabel('X [m]',labelpad=0);ax.set_ylabel('Y [m]',labelpad=0);ax.set_zlabel('Z [m]',labelpad=0)
                ax.tick_params(labelsize=8,pad=0);ax.view_init(23,-55);ax.set_box_aspect((1,1,.7));ax.invert_zaxis()
            fig.legend(*ax.get_legend_handles_labels(),loc='upper center',ncol=3,frameon=False)
            fig.subplots_adjust(left=.01,right=.89,bottom=.08,top=.88,wspace=.20)
            save(fig,f'trajectory_scale_{scale}_3d_ppo_nmpc')
        for col,ylabel,file,mult in [('rmse_m','3-D position RMSE [m]','rmse',1),('rmse_over_length','RMSE / reference length [%]','relative_rmse',100),('yaw_mae_deg','Yaw MAE [deg]','yaw',1),('post_transient_rmse_m','Post-acquisition 3-D RMSE [m]','post_transient_rmse',1)]:
            fig,axes=plt.subplots(1,3,figsize=(11,3.5),layout='constrained')
            for ax,name in zip(axes,LABELS):
                for ctrl,color in COLORS.items():
                    data=metrics[(metrics.controller==ctrl)&(metrics.trajectory==name)]
                    ax.plot(data.scale,mult*data[col],'o-',color=color,label=ctrl)
                ax.set_title(LABELS[name]);ax.set_xticks(scales);ax.set_xlabel('Spatial / period scale');ax.grid(alpha=.2)
            axes[0].set_ylabel(ylabel);fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=4,frameon=False);save(fig,'scale_'+file)
    protocol=json.loads((source/'protocol.json').read_text());protocol.update(controllers=list(COLORS),data_sha256=hashes,paired_contract='Equal plant, references, seeds, sampling and initial states checked within each scale; original PPO CSVs reused. Controller definitions unchanged. Longer scaled runs share wave realization only over their overlapping time interval.')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    print(metrics.pivot_table(index=['trajectory','scale'],columns='controller',values='rmse_m').to_string());print('Figures:',out,flush=True)

if __name__=='__main__':main()
