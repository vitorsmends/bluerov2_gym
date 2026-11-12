"""Additional checkpoint plot layouts, generated exclusively from saved CSVs."""
from pathlib import Path
import argparse
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from .checkpoint_alternatives import MODELS,NAMES,CASES,CASE_NAMES,COLORS


def generate(run):
    run=Path(run)
    output=run/'checkpoint_comparison/alternatives';output.mkdir(parents=True,exist_ok=True)
    source=run/'checkpoint_comparison/metrics.csv';table=pd.read_csv(source)
    if not table.completed.all():raise ValueError('These additional layouts require complete episodes; use the stopped-episode display otherwise.')
    frames={};hashes={str(source):hashlib.sha256(source.read_bytes()).hexdigest()}
    for model in MODELS:
        frames[model]={}
        for case in CASES:
            file=run/f'checkpoints/{model}/data/moderate/ppo_{case}.csv'
            frames[model][case]=pd.read_csv(file)
            hashes[str(file)]=hashlib.sha256(file.read_bytes()).hexdigest()
    emitted=[]
    def save(fig,name):
        for ext in ['pdf','png']:
            p=output/f'{name}.{ext}';fig.savefig(p,dpi=250,bbox_inches='tight',pad_inches=.07);emitted.append(p)
        plt.close(fig)
    def values(model,key):
        return table[table.model==model].set_index('trajectory').reindex(CASES)[key].to_numpy()
    with plt.rc_context({'font.family':'serif','font.size':11,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        # E: traditional grouped bars; two metrics kept separate.
        fig,axes=plt.subplots(1,2,figsize=(8,3.8),layout='constrained')
        x=np.arange(4);width=.18
        for ax,key,ylabel in zip(axes,['rmse_m','yaw_mae_deg'],['Position RMSE [m]','Yaw MAE [deg]']):
            for i,(model,color,name) in enumerate(zip(MODELS,COLORS,NAMES)):
                ax.bar(x+(i-1.5)*width,values(model,key),width,color=color,label=name,edgecolor='white',linewidth=.4)
            ax.set_xticks(x,CASE_NAMES,rotation=25,ha='right')
            ax.set_ylabel(ylabel);ax.set_yscale('log');ax.grid(axis='y',which='both',ls=':',alpha=.25);ax.set_axisbelow(True)
        fig.legend(handles=[Patch(facecolor=c,label=n) for c,n in zip(COLORS,NAMES)],loc='outside upper center',ncol=4,frameon=False,fontsize=10)
        save(fig,'E_grouped_bars')

        # F: same single realization, temporal dynamics rather than summary metrics.
        fig,axes=plt.subplots(2,2,figsize=(7.2,5.4),sharex=True,sharey=True,layout='constrained')
        for ax,case,label in zip(axes.flat,CASES,CASE_NAMES):
            for model,color in zip(MODELS,COLORS):
                df=frames[model][case];error=df.tracking_error_m.to_numpy()
                ax.plot(df.time,np.where(error>0,error,np.nan),color=color,lw=1.25,label=NAMES[MODELS.index(model)])
            ax.set_title(label,fontsize=11);ax.set_yscale('log');ax.set_xlim(0,100);ax.grid(which='both',ls=':',alpha=.25)
        for ax in axes[-1]:ax.set_xlabel('Time [s]')
        for ax in axes[:,0]:ax.set_ylabel('Position error [m]')
        fig.legend(handles=axes[0,0].get_legend_handles_labels()[0],labels=NAMES,
                   loc='outside upper center',ncol=2,frameon=False,fontsize=10)
        save(fig,'F_temporal_errors')

        # G: paired last-checkpoint comparison, linear scales and explicit values.
        fig,axes=plt.subplots(1,2,figsize=(8,3.5),sharey=True,layout='constrained')
        for ax,key,xlabel,digits in zip(axes,['rmse_m','yaw_mae_deg'],['Position RMSE [m]','Yaw MAE [deg]'],[3,1]):
            before=values('curriculum_eight',key);after=values('star_finetune',key)
            ax.hlines(np.arange(4),np.minimum(before,after),np.maximum(before,after),color='.7',lw=2,zorder=1)
            ax.scatter(before,range(4),color=COLORS[2],s=48,marker='o',zorder=3)
            ax.scatter(after,range(4),color=COLORS[3],s=48,marker='D',zorder=3)
            for j,(a,b) in enumerate(zip(before,after)):
                ax.annotate(f'{a:.{digits}f}',(a,j),xytext=(0,9),textcoords='offset points',ha='center',color=COLORS[2],fontsize=9)
                ax.annotate(f'{b:.{digits}f}',(b,j),xytext=(0,-15),textcoords='offset points',ha='center',color=COLORS[3],fontsize=9)
            ax.set_yticks(range(4),CASE_NAMES);ax.set_xlabel(xlabel);ax.margins(x=.2,y=.2);ax.grid(axis='x',ls=':',alpha=.3)
        axes[0].invert_yaxis()
        fig.legend(handles=[Line2D([],[],color=COLORS[i],marker=m,ls='',label=NAMES[i]) for i,m in [(2,'o'),(3,'D')]],
                   loc='outside upper center',ncol=2,frameon=False)
        save(fig,'G_before_after_dumbbell')

        # H: temporal sample distributions, not independent seeds or trials.
        fig,axes=plt.subplots(1,2,figsize=(8,4),layout='constrained')
        traj_colors=['#0072b2','#d55e00','#009e73','#cc79a7']
        for ax,key,label in zip(axes,['tracking_error_m','yaw_error'],['Instantaneous position error [m]','Absolute yaw error [deg]']):
            for j,(case,color) in enumerate(zip(CASES,traj_colors)):
                arrays=[]
                for model in MODELS:
                    values_time=frames[model][case][key].to_numpy()
                    if key=='yaw_error':values_time=np.degrees(np.abs(np.arctan2(np.sin(values_time),np.cos(values_time))))
                    arrays.append(values_time)
                boxes=ax.boxplot(arrays,positions=np.arange(4)+(j-1.5)*.19,widths=.15,
                                 patch_artist=True,showfliers=False,manage_ticks=False)
                for patch in boxes['boxes']:patch.set(facecolor=color,alpha=.65)
                for median in boxes['medians']:median.set(color='black',linewidth=1)
            ax.set_xticks(range(4),['A','B','C4','FT']);ax.set_xlabel('Saved checkpoint');ax.set_ylabel(label)
            ax.set_yscale('log');ax.grid(axis='y',which='both',ls=':',alpha=.25);ax.set_axisbelow(True)
        fig.legend(handles=[Patch(facecolor=c,label=n,alpha=.65) for c,n in zip(traj_colors,CASE_NAMES)],
                   loc='outside upper center',ncol=4,frameon=False,fontsize=10)
        save(fig,'H_temporal_boxplots')

        # Compact visual menu; final figures remain individually exportable.
        fig,axes=plt.subplots(2,2,figsize=(13,9.5),layout='constrained')
        for ax,name,title in zip(axes.flat,['E_grouped_bars','F_temporal_errors','G_before_after_dumbbell','H_temporal_boxplots'],
                                ['E — Grouped bars: all four checkpoints','F — Error histories: all four checkpoints',
                                 'G — Before/after: C4 and star fine-tuning','H — Temporal distributions: all four checkpoints']):
            ax.imshow(plt.imread(output/f'{name}.png'));ax.set_axis_off();ax.set_title(title,fontsize=12)
        fig.savefig(output/'alternatives_E_H_overview.png',dpi=140,bbox_inches='tight');plt.close(fig)
    (output/'additional_provenance.json').write_text(json.dumps(dict(input_sha256=hashes,
        note='Same saved data; one paired wave seed per geometry. No new evaluations. F omits exact zero values on its logarithmic axis. H summarizes autocorrelated time samples, not independent seeds; fliers hidden for readability.',
        figure_files=[p.name for p in emitted]),indent=2)+'\n')
    (output/'additional_options.md').write_text('''E: grouped RMSE/yaw bars, all checkpoints, log scales.\nF: position-error histories, four trajectory panels, log scale.\nG: C4 versus FT only, paired points with numeric values on linear scales.\nH: distributions of errors over time (not seed distributions); fliers omitted.\n\nThe training stopped by its configured plateau criterion. FT denotes that\nterminal policy. Stage mastery and the stopping criterion are distinct.\nThe original plots and manuscript have not been replaced.\n''')
    print(output/'alternatives_E_H_overview.png')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,default=Path('artifacts/evaluation/star_finetune_validation'))
    generate(p.parse_args().run)
