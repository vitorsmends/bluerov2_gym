"""Compact absolute/relative tracking summaries from existing campaign metrics."""
from pathlib import Path
import argparse,json,shutil
import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .article.config import CONTROLLERS


def generate(campaign,paper=None):
    campaign=Path(campaign);tables=campaign/'tables';figures=campaign/'figures/article'
    cfg=yaml.safe_load((campaign/'benchmark/config.resolved.yaml').read_text())
    data=pd.read_csv(tables/'run_metrics.csv')
    occupancy=pd.read_csv(tables/'time_within_tolerance.csv')
    data=data.merge(occupancy[['sea','trajectory','controller','tracking_time_percent']],on=['sea','trajectory','controller'],how='left',validate='one_to_one')
    if data.tracking_time_percent.isna().any():raise ValueError('Missing tracking-occupancy records')
    keys={'figure_eight_depth':'a','square':'side','circle':'radius','star':'radius_outer'}
    lengths={item['name']:float(item['params'][keys[item['name']]]) for item in cfg['trajectories'] if item.get('enabled',True)}
    data['reference_length_m']=data.trajectory.map(lengths)
    if data.reference_length_m.isna().any() or (data.reference_length_m<=0).any():raise ValueError('Positive reference dimensions required')
    data['relative_rmse_percent']=100*data.rmse_m/data.reference_length_m
    data.to_csv(tables/'relative_tracking_metrics.csv',index=False)
    controllers=list(data.controller.unique());seas=list(data.sea.unique())
    styles=dict(CONTROLLERS,FT={'color':'#E69F00'})
    figures.mkdir(parents=True,exist_ok=True)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
        for ax,col,label in zip(axes,['rmse_m','relative_rmse_percent'],['Position RMSE [m]','Relative position RMSE [%]']):
            for offset,ctrl in zip(np.linspace(-.30,.30,len(controllers)),controllers):
                groups=[data.loc[(data.sea==sea)&(data.controller==ctrl),col] for sea in seas]
                color=styles[ctrl]['color']
                ax.boxplot(groups,positions=np.arange(len(seas))+offset,widths=.60/len(controllers)*.8,manage_ticks=False,patch_artist=True,boxprops={'facecolor':color,'alpha':.65},medianprops={'color':'black'},flierprops={'marker':'.','markersize':3})
                ax.plot([],[],color=color,lw=4,label=ctrl)
            ax.set_xticks(range(len(seas)),[s.replace('walker_','').upper() for s in seas]);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=len(controllers),frameon=False)
        for ext in ['png','pdf']:fig.savefig(figures/f'absolute_relative_rmse.{ext}',dpi=250)
        plt.close(fig)
    lines=[r'\begin{table*}[tbp]',r'\centering\small',r'\begingroup\setlength{\tabcolsep}{4pt}',
        r'\caption{Primary tracking occupancy followed by error and computation: mean $\pm$ population standard deviation across three geometries per sea (one paired realization each, seed 42). $P_{\epsilon}^{\mathrm{track}}$ is time within 0.20 m of the time-indexed reference. Relative RMSE is $100\,\mathrm{RMSE}/L$, using eight amplitude, square side, or circle radius for $L$. Dispersion is across geometries, not independent-seed uncertainty; CPU timing is execution-dependent.}',
        r'\label{tab:wave_benchmark_results}',r'\begin{tabular}{llrrrrr}',r'\toprule',
        r'Case & Controller & $P_{\epsilon}^{\mathrm{track}}$ (\%) & RMSE (m) & $\mathrm{RMSE}_{\mathrm{rel}}$ (\%) & Yaw MAE ($^\circ$) & CPU (ms) \\',r'\midrule']
    for sea in seas:
        for ctrl in controllers:
            g=data[(data.sea==sea)&(data.controller==ctrl)];cells=[sea.replace('walker_','').upper(),ctrl]
            for col,digits in [('tracking_time_percent',1),('rmse_m',3),('relative_rmse_percent',1),('yaw_mae_deg',1),('cpu_ms',4)]:
                cells.append(f'${g[col].mean():.{digits}f} \\pm {g[col].std(ddof=0):.{digits}f}$')
            lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular}',r'\endgroup',r'\end{table*}'])
    (tables/'wave_benchmark_results.tex').write_text('\n'.join(lines)+'\n')
    summary=data.groupby('controller')[['tracking_time_percent','rmse_m','relative_rmse_percent','yaw_mae_deg','cpu_ms']].mean()
    summary.to_csv(tables/'tracking_computation_summary.csv')
    (tables/'relative_tracking_protocol.json').write_text(json.dumps(dict(reference_lengths_m=lengths,definition='100 * 3-D synchronous position RMSE / geometry-specific horizontal reference length',pooling='Mean of per-record normalized errors, nine equally weighted sea/geometry cases per controller',caveat='Geometry-specific size convention, not distance travelled or percentage of route completed. Fixed depth is not scaled. Retrospective reference-scale selection; not an independent-seed uncertainty estimate.'),indent=2))
    if paper:
        paper=Path(paper);dest=paper/'figs/current/wave_article_scale4_all/article';dest.mkdir(parents=True,exist_ok=True)
        for ext in ['png','pdf']:shutil.copy2(figures/f'absolute_relative_rmse.{ext}',dest/f'absolute_relative_rmse.{ext}')
        dest=paper/'tables/wave_article_scale4_all';dest.mkdir(parents=True,exist_ok=True)
        for name in ['wave_benchmark_results.tex','relative_tracking_metrics.csv','tracking_computation_summary.csv','relative_tracking_protocol.json']:shutil.copy2(tables/name,dest/name)
    print(summary.to_string())
    return data

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--campaign',type=Path,required=True);parser.add_argument('--paper',type=Path)
    args=parser.parse_args();generate(args.campaign,args.paper)
