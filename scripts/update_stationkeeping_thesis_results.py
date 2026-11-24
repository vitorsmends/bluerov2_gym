"""Publication stationkeeping comparisons from completed CSVs; no new simulations."""
from pathlib import Path
import hashlib,json,shutil,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np,pandas as pd,yaml

COLORS={'PID':'#4E79A7','SMC':'#B07AA1','NMPC':'#E15759','PPO':'#59A14F'}
ORDER=['PID','SMC','NMPC','PPO'];SEAS=['walker_w1','walker_w2','walker_w3']
RUN=ROOT/'artifacts/evaluation/stationkeeping_physics_curriculum_fast_resume'
OUT=RUN/'figures/thesis_current';TABLES=RUN/'tables/thesis_current'
DEST=ROOT/'documents/msc_thesis/images/stationkeeping/validation_current'
TDEST=ROOT/'documents/msc_thesis/tables/stationkeeping/validation_current'


def main():
    manifest=yaml.safe_load((RUN/'manifest.yaml').read_text())
    if manifest.get('status')!='completed':raise ValueError('Evaluation has not completed.')
    for p in [OUT,TABLES,DEST,TDEST]:p.mkdir(parents=True,exist_ok=True)
    cases=pd.read_csv(RUN/'tables/stationkeeping/stationkeeping_metrics_per_case.csv')
    summary=pd.read_csv(RUN/'tables/stationkeeping/stationkeeping_summary_by_controller.csv').set_index('controller')
    seas=pd.read_csv(RUN/'tables/stationkeeping/stationkeeping_summary_by_controller_and_sea.csv')
    if len(cases)!=180 or not cases.n_steps.eq(1000).all():raise ValueError('Expected 180 complete 1000-step episodes.')
    inputs=[];frames=[]
    for ctrl in ORDER:
        path=RUN/f'data/{ctrl.lower()}_stationkeeping.csv';inputs.append(path)
        d=pd.read_csv(path,usecols=['sea_scenario','target_id','repetition_id','seed','time','position_error_m','yaw_error','control_effort'])
        if len(d)!=45000:raise ValueError(f'Unexpected row count: {path}')
        d['controller']=ctrl;d['yaw_abs_deg']=np.rad2deg(np.abs(np.arctan2(np.sin(d.yaw_error),np.cos(d.yaw_error))))
        frames.append(d)
    data=pd.concat(frames,ignore_index=True)
    def save(fig,name):
        for ext in ['pdf','png']:
            path=OUT/f'{name}.{ext}';fig.savefig(path,dpi=300,bbox_inches='tight',pad_inches=.05);shutil.copy2(path,DEST/path.name)
        plt.close(fig)
    def clean(ax):
        ax.grid(alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    with plt.rc_context({'font.family':'serif','font.size':13,'axes.labelsize':14,'pdf.fonttype':42}):
        exported=[]
        for sea in SEAS:
            fig,axes=plt.subplots(2,1,figsize=(6.8,5.2),sharex=True,layout='constrained')
            for ctrl in ORDER:
                d=data[(data.sea_scenario==sea)&(data.controller==ctrl)]
                for ax,col in zip(axes,['position_error_m','yaw_abs_deg']):
                    stats=d.groupby('time')[col].agg(['mean',lambda v:v.std(ddof=0)])
                    stats.columns=['mean','std'];x=stats.index.to_numpy();mean=stats['mean'].to_numpy();std=stats['std'].to_numpy()
                    ax.plot(x,mean,color=COLORS[ctrl],lw=1.35,label=ctrl)
                    ax.fill_between(x,np.maximum(0,mean-std),mean+std,color=COLORS[ctrl],alpha=.14,lw=0)
                    exported.extend(dict(sea=sea,controller=ctrl,metric=col,time_s=float(t),mean=float(m),std=float(s)) for t,m,s in zip(x,mean,std))
            axes[0].set(ylabel='Position error (m)',ylim=(0,None));axes[1].set(ylabel='Absolute yaw error (deg)',xlabel='Time (s)',ylim=(0,None))
            for ax in axes:clean(ax);ax.set_xlim(0,100)
            axes[0].legend(ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=11)
            save(fig,sea.replace('walker_','stationkeeping_')+'_response')
        pd.DataFrame(exported).to_csv(TABLES/'response_curves.csv',index=False)
        fig,axes=plt.subplots(2,1,figsize=(6.8,5.2),sharex=True,layout='constrained')
        for ax,metric,label in zip(axes,['rmse_position_error_m','yaw_mae_deg'],['Position RMSE (m)','Yaw MAE (deg)']):
            for j,ctrl in enumerate(ORDER):
                values=[cases[(cases.controller==ctrl)&(cases.sea_scenario==sea)][metric].to_numpy() for sea in SEAS]
                pos=np.arange(3)+(j-1.5)*.18
                boxes=ax.boxplot(values,positions=pos,widths=.15,patch_artist=True,manage_ticks=False,showfliers=True,
                    medianprops={'color':'black','linewidth':1.1},flierprops={'marker':'.','markersize':2})
                for box in boxes['boxes']:box.set(facecolor=COLORS[ctrl],alpha=.55)
            clean(ax);ax.set_ylabel(label)
        axes[1].set_xticks(range(3),['W1','W2','W3']);axes[0].legend(handles=[Patch(facecolor=COLORS[c],alpha=.55,label=c) for c in ORDER],ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=11)
        save(fig,'stationkeeping_error_boxplots')
        fig,axes=plt.subplots(3,1,figsize=(6.8,6),sharex=True,layout='constrained')
        for ax,sea in zip(axes,SEAS):
            for ctrl in ORDER:
                d=data[(data.controller==ctrl)&(data.sea_scenario==sea)]
                g=d.groupby('time').control_effort;mean=g.mean();std=g.std(ddof=0)
                ax.plot(mean.index,mean.values,color=COLORS[ctrl],lw=1.2,label=ctrl)
                ax.fill_between(mean.index,np.maximum(0,mean-std),mean+std,color=COLORS[ctrl],alpha=.12,lw=0)
            clean(ax);ax.set_ylabel(sea.replace('walker_','').upper()+r': $\|\mathbf{f}\|^2$ (N$^2$)');ax.set_ylim(bottom=0);ax.set_xlim(0,100)
        axes[0].legend(ncol=4,frameon=False,loc='lower center',bbox_to_anchor=(.5,1.01),fontsize=11);axes[-1].set_xlabel('Time (s)')
        save(fig,'stationkeeping_effort')
        fig,ax=plt.subplots(figsize=(6.2,3.8),layout='constrained');x=np.arange(4)
        ax.bar(x,[summary.loc[c,'total_energy_J_mean']/1000 for c in ORDER],yerr=[summary.loc[c,'total_energy_J_std']/1000 for c in ORDER],color=[COLORS[c] for c in ORDER],capsize=4,alpha=.8)
        ax.set_xticks(x,ORDER);ax.set_ylabel('Estimated energy per episode (kJ)');clean(ax);save(fig,'stationkeeping_energy')
        fig,axes=plt.subplots(1,2,figsize=(8.4,3.6),layout='constrained')
        for ax,metric,ylabel in zip(axes,['rmse_position_error_m','yaw_mae_deg'],['Position RMSE (m)','Yaw MAE (deg)']):
            for ctrl in ORDER:
                row=summary.loc[ctrl];cx=row['mean_cpu_time_ms_mean'];cy=row[metric+'_mean']
                ax.errorbar(cx,cy,xerr=row['mean_cpu_time_ms_std'],yerr=row[metric+'_std'],fmt='o',color=COLORS[ctrl],capsize=3,ms=5)
                ax.annotate(ctrl,(cx,cy),xytext=(5,5),textcoords='offset points',fontsize=10)
            clean(ax);ax.set_xscale('log');ax.set_xlabel('Mean CPU time (ms/action)');ax.set_ylabel(ylabel);ax.margins(x=.2,y=.2);ax.set_ylim(bottom=0)
        save(fig,'stationkeeping_accuracy_cpu')
    def pm(row,key,scale=1,dec=3):return f"{row[key+'_mean']/scale:.{dec}f} $\\pm$ {row[key+'_std']/scale:.{dec}f}"
    lines=[r'\begin{table}[tbp]',r'\centering',r'\caption[Station-keeping comparison in W1--W3.]{Full-horizon station-keeping metrics (mean $\pm$ population standard deviation across five translated targets and three paired wave seeds per sea). Each controller has 15 cases per sea; these are not 15 independent wave realizations.}',r'\label{tab:stationkeeping_by_controller_and_sea}',r'\begingroup\setlength{\tabcolsep}{3pt}\small',r'\begin{tabular}{llcccc}',r'\toprule',r'Sea & Controller & RMSE (m) & Yaw MAE ($^\circ$) & CPU (ms) & Energy (kJ) \\',r'\midrule']
    for sea in SEAS:
        for ctrl in ORDER:
            row=seas[(seas.controller==ctrl)&(seas.sea_scenario==sea)].iloc[0]
            lines.append(sea.replace('walker_','').upper()+' & '+ctrl+' & '+pm(row,'rmse_position_error_m')+' & '+pm(row,'yaw_mae_deg')+' & '+pm(row,'mean_cpu_time_ms',dec=4)+' & '+pm(row,'total_energy_J',1000,2)+r' \\')
        if sea!=SEAS[-1]:lines.append(r'\midrule')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\endgroup',r'\end{table}'])
    (TABLES/'summary.tex').write_text('\n'.join(lines)+'\n');shutil.copy2(TABLES/'summary.tex',TDEST/'summary.tex')
    for source in ['stationkeeping_summary_by_controller.csv','stationkeeping_summary_by_controller_and_sea.csv','stationkeeping_metrics_per_case.csv']:
        shutil.copy2(RUN/'tables/stationkeeping'/source,TDEST/source)
    inputs += [RUN/'config.resolved.yaml',RUN/'manifest.yaml',RUN/'tables/stationkeeping/stationkeeping_metrics_per_case.csv']
    (TDEST/'provenance.json').write_text(json.dumps(dict(run=str(RUN),episodes=180,horizon_steps=1000,dt_s=.1,
        dispersion='Population SD over targets and wave realizations; translated targets are not independent waves.',
        input_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    print('Exported seven comparison figures and the updated table to',DEST)

if __name__=='__main__':main()
