"""Compare archived article PPO, star fine-tuning and NMPC in Moderate seas."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patheffects

ROOT = Path(__file__).resolve().parents[3]
CASES = ['reference_889_figure_eight', 'square', 'circle', 'star']
LABELS = ['Figure-eight', 'Square', 'Circle', 'Star']
POLICIES = [('ppo_article', 'PPO', '#59A14F'),
            ('ft_star', 'FT', '#E69F00'), ('nmpc', 'NMPC', '#E15759')]


def generate(output, campaign):
    output.mkdir(parents=True, exist_ok=True)
    frames, hashes, metrics = {}, {}, []
    for case in CASES:
        baseline = ROOT/'artifacts/evaluation'/('physics_wave_curriculum_yaw_star' if case == 'star' else 'physics_wave_curriculum_yaw_sea_states')
        tuned = campaign/('combined_star' if case == 'star' else 'combined_benchmark')
        paths = {'ppo_article': baseline/f'data/moderate/ppo_{case}.csv',
                 'ft_star': tuned/f'data/moderate/ppo_{case}.csv',
                 'nmpc': baseline/f'data/moderate/nmpc_{case}.csv'}
        reference = None
        for policy, path in paths.items():
            df = pd.read_csv(path).sort_values(['repetition_id', 'time']).reset_index(drop=True)
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            pair_columns = ['repetition_id','seed','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std'] + [c for c in df if c.endswith('_ref')]
            if reference is not None:
                np.testing.assert_allclose(reference[pair_columns], df[pair_columns], rtol=0, atol=1e-10)
                states = ['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                np.testing.assert_allclose(reference.groupby('repetition_id').first()[states],df.groupby('repetition_id').first()[states],rtol=0,atol=1e-10)
            reference = df
            df['yaw_abs_deg'] = np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error), np.cos(df.yaw_error))))
            for rep, group in df.groupby('repetition_id'):
                if len(group) != 1000 or not np.allclose(group.time, np.arange(1000)*.1):
                    raise ValueError(f'Incomplete evaluation: {path}')
                metrics.append(dict(policy=policy,trajectory=case,repetition=int(rep),seed=int(group.seed.iloc[0]),
                                    position_rmse_m=float(np.sqrt(np.mean(group.tracking_error_m**2))),yaw_mae_deg=float(group.yaw_abs_deg.mean())))
            if df.repetition_id.nunique()!=3: raise ValueError('Expected three paired repetitions')
            frames[case,policy] = df
    table = pd.DataFrame(metrics)
    table.to_csv(output/'run_metrics.csv',index=False)
    summary = table.groupby(['policy','trajectory'])[['position_rmse_m','yaw_mae_deg']].mean().reset_index()
    overall = table.groupby('policy')[['position_rmse_m','yaw_mae_deg']].mean().reset_index().assign(trajectory='overall')
    pd.concat([summary,overall]).to_csv(output/'summary.csv',index=False)
    def save(fig,name):
        for ext in ['pdf','png']: fig.savefig(output/f'{name}.{ext}',dpi=250,bbox_inches='tight')
        plt.close(fig)
    with plt.rc_context({'font.family':'serif','font.size':11,'pdf.fonttype':42}):
        for column, ylabel, name in [('tracking_error_m','Position error [m]','position_error'),('yaw_abs_deg','Absolute yaw error [deg]','yaw_error')]:
            fig,axes=plt.subplots(2,2,figsize=(8,6),sharex=True)
            handles=[]
            for ax,case,label in zip(axes.flat,CASES,LABELS):
                for policy,title,color in POLICIES:
                    stats=frames[case,policy].groupby('time')[column].agg(['mean',lambda v:v.std(ddof=0)])
                    mean,sd=stats.iloc[:,0].to_numpy(),stats.iloc[:,1].to_numpy()
                    line,=ax.plot(stats.index,mean,color=color,lw=1.3,label=title)
                    ax.fill_between(stats.index,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.13,lw=0)
                    if case==CASES[0]:handles.append(line)
                ax.set_title(label);ax.set_ylabel(ylabel);ax.grid(alpha=.2)
                ax.set_xlim(0,100)
            for ax in axes[1]:ax.set_xlabel('Time [s]')
            fig.legend(handles=handles,loc='upper center',ncol=3,frameon=False)
            fig.tight_layout(rect=(0,0,1,.94));save(fig,name)
        fig=plt.figure(figsize=(8,6.4));handles=[]
        for i,(case,label) in enumerate(zip(CASES,LABELS)):
            ax=fig.add_subplot(2,2,i+1,projection='3d')
            ref=frames[case,'ppo_article'].query('repetition_id == 0')
            line,=ax.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1.5,label='Reference',path_effects=[patheffects.Stroke(linewidth=2.7,foreground='white'),patheffects.Normal()])
            lines=[line]
            for policy,title,color in POLICIES:
                df=frames[case,policy].query('repetition_id == 0')
                lines.extend(ax.plot(df.x,df.y,df.z,color=color,lw=1.1,label=title))
            if i==0:handles=lines
            ax.set_xlabel('X [m]',labelpad=0);ax.set_ylabel('Y [m]',labelpad=0);ax.set_zlabel('Z [m]',labelpad=2)
            ax.set_title(label);ax.set_box_aspect((1,1,.85));ax.set_proj_type('ortho');ax.view_init(20,-60);ax.invert_zaxis()
            ax.tick_params(labelsize=8,pad=0)
        fig.legend(handles=handles,loc='upper center',ncol=4,frameon=False)
        fig.subplots_adjust(top=.91,bottom=.06,left=.02,right=.96,hspace=.20,wspace=.12);save(fig,'trajectories_3d')
    (output/'provenance.json').write_text(json.dumps({'sea':'moderate','repetitions':3,'source_sha256':hashes,'overall_weighting':'Equal weight per geometry and repetition; four geometries including star','bands':'Population standard deviation of three paired disturbance repetitions'},indent=2))
    print(overall.to_string(index=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=ROOT/'artifacts/evaluation/star_finetune_validation')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/star_finetune_validation/controller_comparison')
    args=parser.parse_args();generate(args.output,args.campaign)
