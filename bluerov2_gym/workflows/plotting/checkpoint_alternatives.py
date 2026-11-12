"""Alternative displays of saved checkpoint metrics; no new evaluations.

python -m bluerov2_gym.workflows.plotting.checkpoint_alternatives --run artifacts/evaluation/finetune_article_campaign
"""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MODELS=['stage0_no_waves','preparation_waves','curriculum_eight','star_finetune']
NAMES=['A: no waves','B: preparation waves','C4: figure-eight','FT: star']
CASES=['reference_889_figure_eight','square','circle','star']
CASE_NAMES=['Figure-eight','Square','Circle','Star']
COLORS=['#8c8c8c','#e69f00','#0072b2','#009e73']
MARKERS=['o','s','^','D']


def generate(run):
    run=Path(run);source=run/'checkpoint_comparison/metrics.csv'
    NAMES=list(globals()['NAMES'])
    incomplete=False
    if (run/'campaign.yaml').exists():
        from ..configuration import read_yaml
        incomplete=read_yaml(run/'campaign.yaml')['spec'].get('finetuned_training_status')=='early_stopped'
        if incomplete:NAMES[-1]='FT: star'
    data=pd.read_csv(source)
    if data.duplicated(['model','trajectory']).any():raise ValueError('Duplicate checkpoint/trajectory metrics.')
    expected={(m,c) for m in MODELS for c in CASES}
    if set(zip(data.model,data.trajectory))!=expected:raise ValueError('Expected exactly four checkpoints and four trajectories.')
    # Never turn partial-horizon errors into a successful full-horizon metric.
    invalid=~data.completed.astype(bool)
    data.loc[invalid,['rmse_m','yaw_mae_deg']]=np.nan
    output=run/'checkpoint_comparison/alternatives';output.mkdir(parents=True,exist_ok=True)
    saved=[]
    def save(fig,name):
        for ext in ['pdf','png']:
            path=output/f'{name}.{ext}';fig.savefig(path,dpi=250,bbox_inches='tight',pad_inches=.07);saved.append(path)
        plt.close(fig)
    def values(model,key):
        return data[data.model==model].set_index('trajectory').reindex(CASES)[key].to_numpy(float)
    metrics=[('rmse_m','Position RMSE [m]'),('yaw_mae_deg','Yaw MAE [deg]')]
    with plt.rc_context({'font.family':'serif','font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False}):
        # A: same four policies, grouped by trajectory, with explicit logarithmic axes.
        fig,axes=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
        offsets=np.linspace(-.25,.25,4)
        for ax,(key,label) in zip(axes,metrics):
            for j,(model,name,color) in enumerate(zip(MODELS,NAMES,COLORS)):
                ax.scatter(values(model,key),np.arange(4)+offsets[j],color=color,marker=MARKERS[j],s=43,label=name,zorder=3)
            ax.set_yticks(range(4),CASE_NAMES)
            ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel(label+' (log scale)')
            ax.grid(axis='x',which='both',ls=':',alpha=.3)
        handles=[Line2D([],[],color=c,marker=m,ls='',label=n) for c,m,n in zip(COLORS,MARKERS,NAMES)]
        fig.legend(handles=handles,loc='outside upper center',ncol=2,frameon=False)
        save(fig,'A_grouped_dots')

        # B: checkpoint progression; edges are visual guides, not learning curves.
        fig,axes=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
        traj_colors=['#0072b2','#d55e00','#009e73','#cc79a7']
        for ax,(key,label) in zip(axes,metrics):
            for case,label_case,color,marker in zip(CASES,CASE_NAMES,traj_colors,MARKERS):
                vals=data[data.trajectory==case].set_index('model').reindex(MODELS)[key]
                ax.plot(range(4),vals,color=color,marker=marker,lw=1.4,ms=5,label=label_case)
            ax.set_xticks(range(4),['A','B','C4','FT'])
            ax.set_xlabel('Saved checkpoint');ax.set_ylabel(label);ax.set_yscale('log')
            ax.grid(which='both',ls=':',alpha=.3)
        fig.legend(handles=axes[0].get_legend_handles_labels()[0],labels=CASE_NAMES,
                   loc='outside upper center',ncol=4,frameon=False)
        save(fig,'B_checkpoint_progression')

        # C: only the last two models, on a common percent-change scale.
        fig,ax=plt.subplots(figsize=(6.8,3.6),layout='constrained')
        changes=[]
        for j,(key,label,color,marker) in enumerate([('rmse_m','Position RMSE','#0072b2','o'),('yaw_mae_deg','Yaw MAE','#d55e00','D')]):
            base=values('curriculum_eight',key);fine=values('star_finetune',key)
            change=100*(base-fine)/base
            y=np.arange(4)+(j-.5)*.22
            ax.hlines(y,0,change,color=color,lw=1.8)
            ax.scatter(change,y,color=color,marker=marker,s=42,label=label,zorder=3)
            for i,value in enumerate(change):
                if not np.isfinite(value):continue
                ax.annotate(f'{value:+.1f}%',(value,y[i]),xytext=(6 if value>=0 else -6,0),
                            textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=10)
                changes.append(dict(trajectory=CASES[i],metric=key,reduction_percent=float(value)))
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(4),CASE_NAMES);ax.invert_yaxis()
        ax.set_xlabel('Error reduction vs. C4 [%]   (positive = lower error)')
        if incomplete:ax.set_title('Star fine-tuning vs. C4',fontsize=11)
        ax.margins(x=.22);ax.grid(axis='x',ls=':',alpha=.3)
        ax.legend(loc='lower left',frameon=False)
        save(fig,'C_finetuning_change')
        pd.DataFrame(changes).to_csv(output/'finetuning_change.csv',index=False)

        # D: all sixteen observations on a joint accuracy plane, not a fitted Pareto front.
        fig,ax=plt.subplots(figsize=(6.8,4.5),layout='constrained')
        for model,color in zip(MODELS,COLORS):
            for case,marker in zip(CASES,MARKERS):
                row=data[(data.model==model)&(data.trajectory==case)].iloc[0]
                if row.completed:ax.scatter(row.rmse_m,row.yaw_mae_deg,color=color,marker=marker,s=65,edgecolor='white',lw=.4,zorder=3)
        ax.set_xscale('log');ax.set_yscale('log')
        ax.set(xlabel='Position RMSE [m] (log scale)',ylabel='Yaw MAE [deg] (log scale)')
        ax.grid(which='both',ls=':',alpha=.3)
        first=ax.legend(handles=[Line2D([],[],color=c,marker='o',ls='',label=n) for c,n in zip(COLORS,NAMES)],
                        loc='upper left',bbox_to_anchor=(1.02,1),frameon=False,title='Checkpoint',fontsize=10)
        ax.add_artist(first)
        ax.legend(handles=[Line2D([],[],color='black',marker=m,ls='',label=n) for m,n in zip(MARKERS,CASE_NAMES)],
                  loc='lower left',bbox_to_anchor=(1.02,0),frameon=False,title='Trajectory',fontsize=10)
        save(fig,'D_position_heading')
    provenance=dict(input=str(source),input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    figures=[p.name for p in saved],wave_seeds=sorted(map(int,data.seed.unique())),
                    note='One paired realization per cell; no confidence intervals or across-training-seed variability. B connects saved checkpoints, not training epochs. C compares only C4 and FT; positive values mean error reduction.',
                    incomplete_episodes=int(invalid.sum()))
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'README.md').write_text('''# Checkpoint plot alternatives\n\nA: grouped points, all four checkpoints, logarithmic axes.\nB: four trajectory lines across saved checkpoints; not epoch learning curves.\nC: position/yaw percent error reduction from C4 to FT only; negative = regression.\nD: joint position/yaw accuracy, all four checkpoints, log/log axes.\n\nAll figures use the same existing metrics and one paired wave seed; no new\nevaluations were run. The original heatmap was not replaced. Partial episodes\nare excluded from full-horizon metrics, not assigned a low partial error.\n''')
    print(output)
    return saved


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/evaluation/star_finetune_validation'))
    generate(p.parse_args().run)
