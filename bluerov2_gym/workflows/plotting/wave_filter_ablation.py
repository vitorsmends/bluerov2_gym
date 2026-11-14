"""Compare fixed article PPO before/after disabling only the wave-velocity filter."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .finetune_controller_comparison import ROOT


def generate(run):
    run=Path(run);output=run/'comparison';output.mkdir(exist_ok=True)
    config=yaml.safe_load((run/'config.resolved.yaml').read_text())
    cases=['reference_889_figure_eight','star'];names=['Figure-eight','Star']
    frames={};records=[];hashes={}
    for case in cases:
        old=ROOT/'artifacts/evaluation'/('physics_wave_curriculum_yaw_star' if case=='star' else 'physics_wave_curriculum_yaw_sea_states')
        baseline=yaml.safe_load((old/'config.resolved.yaml').read_text())
        expected=baseline['environment']['dynamics_config'].copy()
        import copy
        expected=copy.deepcopy(expected);expected['disturbance_model']['wave_filter']=False
        if expected!=config['environment']['dynamics_config']:raise ValueError('Other dynamics changed')
        if config['experiment']!=baseline['experiment']:raise ValueError('Different experiment')
        if config['disturbance']['scenarios']['moderate']!=baseline['disturbance']['scenarios']['moderate']:raise ValueError('Different sea')
        for key in ['config','yaw_tracking']:
            if config['environment'].get(key)!=baseline['environment'].get(key):raise ValueError(f'Different {key}')
        policy=[c for c in baseline['controllers'] if c['name']=='ppo'][0]
        if policy!=config['controllers'][0]:raise ValueError('Different policy or normalizer')
        matched=None
        for label,folder in [('Filter on',old),('Filter off',run)]:
            path=folder/f'data/moderate/ppo_{case}.csv';hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            df=pd.read_csv(path).sort_values(['repetition_id','time']).reset_index(drop=True)
            columns=['seed','repetition_id','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']+[c for c in df if c.endswith('_ref')]
            if matched is not None:
                np.testing.assert_allclose(matched[columns],df[columns],rtol=0,atol=1e-10)
                states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                np.testing.assert_allclose(matched.groupby('repetition_id').first()[states],df.groupby('repetition_id').first()[states],rtol=0,atol=1e-10)
            matched=df
            df['yaw_abs_deg']=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error))))
            for rep,g in df.groupby('repetition_id'):
                if len(g)!=1000 or not np.allclose(g.time,np.arange(1000)*.1):raise ValueError('Incomplete horizon')
                post=g[g.time>=10]
                records.append(dict(trajectory=case,condition=label,repetition=int(rep),seed=int(g.seed.iloc[0]),
                    rmse_m=np.sqrt(np.mean(g.tracking_error_m**2)),yaw_mae_deg=g.yaw_abs_deg.mean(),
                    post10_rmse_m=np.sqrt(np.mean(post.tracking_error_m**2)),post10_yaw_mae_deg=post.yaw_abs_deg.mean(),
                    max_error_m=g.tracking_error_m.max(),last10_mean_error_m=g[g.time>=90].tracking_error_m.mean()))
            frames[case,label]=df
    metrics=pd.DataFrame(records);metrics.to_csv(output/'run_metrics.csv',index=False)
    summary=metrics.groupby(['trajectory','condition'])[['rmse_m','yaw_mae_deg','post10_rmse_m','post10_yaw_mae_deg','max_error_m','last10_mean_error_m']].mean()
    summary.to_csv(output/'summary.csv')
    colors={'Filter on':'#1678B5','Filter off':'#E69F00'}
    def save(fig,name):
        for ext in ['pdf','png']:fig.savefig(output/f'{name}.{ext}',dpi=250,bbox_inches='tight')
        plt.close(fig)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        fig,axes=plt.subplots(2,2,figsize=(9,6),sharex=True)
        for row,(case,name) in enumerate(zip(cases,names)):
            for col,(column,ylabel) in enumerate([('tracking_error_m','Position error [m]'),('yaw_abs_deg','Absolute yaw error [deg]')]):
                ax=axes[row,col]
                for label,color in colors.items():
                    stats=frames[case,label].groupby('time')[column].agg(['mean',lambda v:v.std(ddof=0)])
                    mean,sd=stats.iloc[:,0].to_numpy(),stats.iloc[:,1].to_numpy()
                    ax.plot(stats.index,mean,color=color,lw=1.3,label=label)
                    ax.fill_between(stats.index,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.15,lw=0)
                ax.set_title(name);ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.set_xlim(0,100)
                if row==1:ax.set_xlabel('Time [s]')
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
        fig.tight_layout(rect=(0,0,1,.94));save(fig,'errors_filter_on_off')
        fig=plt.figure(figsize=(9,4.4));handles=None
        for i,(case,name) in enumerate(zip(cases,names)):
            ax=fig.add_subplot(1,2,i+1,projection='3d');ref=frames[case,'Filter on'].query('repetition_id == 0')
            lines=ax.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1.5,label='Reference')
            for label,color in colors.items():
                df=frames[case,label].query('repetition_id == 0');lines+=ax.plot(df.x,df.y,df.z,color=color,lw=1.2,label=label)
            handles=lines;ax.set_title(name)
            ax.set_xlabel('X [m]',labelpad=1);ax.set_ylabel('Y [m]',labelpad=1);ax.set_zlabel('Z [m]',labelpad=2)
            ax.set_box_aspect((1,1,.85));ax.view_init(22,-60);ax.invert_zaxis();ax.tick_params(labelsize=9,pad=0)
        fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
        fig.subplots_adjust(top=.83,left=.02,right=.97,bottom=.08,wspace=.20);save(fig,'trajectories_filter_on_off')
    for key in ['model_path','vecnormalize_path']:
        p=Path(config['controllers'][0]['params'][key]);hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps({'sources_sha256':hashes,'change':{'wave_filter':False},'unchanged':'Original article PPO and normalizer; target mechanics; all other disturbance flags; Moderate; references; initial states; seeds 42,43,44','bands':'Population standard deviation across three repetitions; no independent policy retraining'},indent=2))
    print(summary.to_string())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,default=ROOT/'artifacts/evaluation/ppo_without_wave_filter');a=p.parse_args();generate(a.run)
