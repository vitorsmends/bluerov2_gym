"""Compact thesis-only versions of the archived five-controller summary figures."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from .finetune_controller_comparison import ROOT


def generate():
    source=ROOT/'artifacts/evaluation/star_finetune_validation/article_all_controllers'
    out=ROOT/'documents/msc_thesis/images/path_tracking/current/compact_summary';out.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(source/'tables/run_metrics.csv')
    colors={'PID':'#4E79A7','SMC':'#B07AA1','NMPC':'#E15759','PPO':'#59A14F','FT':'#E69F00'}
    curve=pd.read_csv(ROOT/'bluerov2_gym/workflows/plotting/assets/t200_performance_16v.csv').sort_values('force_n')
    def save(fig,name):
        fig.tight_layout(pad=.6)
        for ext in ['pdf','png']:fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.03)
        plt.close(fig)
    with plt.rc_context({'font.family':'serif','font.size':14,'axes.labelsize':14,'xtick.labelsize':12,'ytick.labelsize':12,'legend.fontsize':10,'pdf.fonttype':42}):
        fig,ax=plt.subplots(figsize=(4.4,3.5))
        for j,sea in enumerate(['calm','moderate','rough']):
            values=[metrics[(metrics.controller==c)&(metrics.sea==sea)].rmse_m for c in colors]
            boxes=ax.boxplot(values,positions=np.arange(5)+(j-1)*.24,widths=.19,patch_artist=True,manage_ticks=False,showmeans=True,
                             meanprops=dict(marker='o',markerfacecolor='white',markeredgecolor='black',markersize=3),flierprops=dict(markersize=3))
            for box,color in zip(boxes['boxes'],colors.values()):box.set_facecolor(color);box.set_alpha(.55);box.set_hatch(['','///','xx'][j])
        ax.set_xticks(np.arange(5),list(colors));ax.set_ylabel('RMSE [m]');ax.grid(axis='y',alpha=.25)
        ax.legend(handles=[Patch(facecolor='white',edgecolor='black',hatch=h,label=s) for h,s in zip(['','///','xx'],['Calm','Moderate','Rough'])],loc='upper right')
        save(fig,'boxplot_rmse_error')
        histories={}
        for ctrl in colors:
            energies=[];cpus=[]
            for path in sorted((source/'data').glob(f'*/{ctrl.lower()}_*.csv')):
                df=pd.read_csv(path)
                for _,g in df.groupby('repetition_id'):
                    g=g.sort_values('time')
                    power=np.interp(g[[f'T{i}' for i in range(1,7)]],curve.force_n,curve.power_w).sum(axis=1)
                    energies.append(np.cumsum(power)*.1);cpus.append(g.controller_cpu_time_s.to_numpy()*1000)
            if len(energies)!=27:raise ValueError(f'Expected 27 runs for {ctrl}')
            histories[ctrl]=(np.array(energies),np.array(cpus))
        for index,ylabel,name in [(0,'Cumulative energy [J]','cumulative_energy_mean_std'),(1,'CPU time [ms]','fig_path_tracking_cpu_time_mean_std')]:
            fig,ax=plt.subplots(figsize=(4.4,3.5))
            for ctrl,color in colors.items():
                values=histories[ctrl][index];mean,sd=values.mean(axis=0),values.std(axis=0)
                ax.plot(np.arange(1000)*.1,mean,color=color,label=ctrl,lw=1.3)
                ax.fill_between(np.arange(1000)*.1,np.maximum(0,mean-sd),mean+sd,color=color,alpha=.14,lw=0)
            ax.set_xlabel('Time [s]');ax.set_ylabel(ylabel);ax.set_xlim(0,100);ax.grid(alpha=.25)
            ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.25),columnspacing=.8,handlelength=1.4)
            save(fig,name)
        fig,ax=plt.subplots(figsize=(4.4,3.5))
        for ctrl,color in colors.items():
            g=metrics[metrics.controller==ctrl]
            ax.errorbar(g.cpu_ms.mean(),g.rmse_m.mean(),xerr=g.cpu_ms.std(ddof=0),yerr=g.rmse_m.std(ddof=0),fmt='o',color=color,capsize=3,label=ctrl)
        ax.set_xscale('log');ax.set_xlabel('Mean CPU time [ms]');ax.set_ylabel('RMSE [m]');ax.grid(alpha=.25)
        ax.legend(loc='upper right',ncol=2,columnspacing=.6,handlelength=1)
        save(fig,'pareto_rmse_vs_cpu_time_mean_std')
    print(out)

if __name__=='__main__':generate()
