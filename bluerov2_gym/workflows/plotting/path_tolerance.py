"""Sampled time within a fixed tracking/path tolerance; no new simulation."""
from pathlib import Path
import argparse,json,hashlib,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..configuration import read_yaml
from ..evaluation.path_metrics import geometric_path_distance,time_within_tolerance
from .wave_article_results import CONTROLLERS,SEAS,CASES
from .article.config import CONTROLLERS as STYLES


def plot_tolerance(data,figs,tolerance=.2,controllers=None):
    """Render the recorded occupancy summaries without changing any metrics."""
    figs=Path(figs);figs.mkdir(parents=True,exist_ok=True)
    controllers=controllers or [c for c in CONTROLLERS if c in set(data.controller)]
    styles=dict(STYLES,FT={'color':'#E69F00','prefix':'ft_'})
    patterns={'PPO':'','NMPC':'///','PID':'xx','SMC':'..','FT':'++'}
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42,'axes.linewidth':.75,'mathtext.fontset':'cm'}):
        fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained');x=np.arange(3)
        for ax,col,title in [(axes[0],'tracking_time_percent',r'$P_{\epsilon}^{\mathrm{track}}$ [%]'),(axes[1],'path_time_percent',r'$P_{\epsilon}^{\mathrm{path}}$ [%]')]:
            for offset,ctrl in zip(np.linspace(-.30,.30,len(controllers)),controllers):
                means=[data.loc[(data.sea==sea)&(data.controller==ctrl),col].mean() for sea in SEAS]
                bars=ax.bar(x+offset,means,width=.60/len(controllers)*.85,color=styles[ctrl]['color'],label=ctrl,edgecolor='.2',linewidth=.65,hatch=patterns[ctrl])
            ax.set_ylabel(title,fontsize=13,labelpad=8)
            ax.set_xticks(x,['W1','W2','W3']);ax.set_ylim(0,100)
            ax.grid(axis='y',color='.86',linestyle='--',linewidth=.65);ax.set_axisbelow(True)
            ax.spines[['top','right']].set_visible(False)
            for spine in ax.spines.values():spine.set_color('.35')
            ax.tick_params(axis='both',labelsize=11)
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside upper center',ncol=len(controllers),frameon=False,fontsize=12)
        for ext in ['png','pdf']:fig.savefig(figs/f'time_within_tolerance.{ext}',dpi=250)
        plt.close(fig)


def generate(campaign,tolerance=.2,paper=None,include_ft=False,selected_controllers=None):
    campaign=Path(campaign).resolve();run=campaign/'benchmark';cfg=read_yaml(run/'config.resolved.yaml')
    specs={x['name']:x['params'] for x in cfg['trajectories'] if x.get('enabled',True)};cases=list(specs);figs=campaign/'figures';tables=campaign/'tables';tables.mkdir(exist_ok=True);figs.mkdir(exist_ok=True)
    controllers=selected_controllers or (['PPO','FT','NMPC','PID','SMC'] if include_ft else CONTROLLERS);styles=dict(STYLES,FT={'color':'#E69F00','prefix':'ft_'})
    rows=[];points=[];hashes={}
    for sea in SEAS:
        for case in cases:
            for ctrl in controllers:
                path=run/f'data/{sea}/{ctrl.lower()}_{case}.csv';df=pd.read_csv(path);hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                positions=df[['x','y','z']].to_numpy();cross=geometric_path_distance(positions,case,specs[case]);tracking=df.tracking_error_m.to_numpy()
                synchronous=np.linalg.norm(positions-df[['x_ref','y_ref','z_ref']].to_numpy(),axis=1)
                np.testing.assert_allclose(tracking,synchronous,rtol=1e-9,atol=1e-9)
                post=df.time.to_numpy()>=10
                rows.append(dict(sea=sea,trajectory=case,controller=ctrl,tolerance_m=tolerance,
                    tracking_time_percent=time_within_tolerance(tracking,tolerance),path_time_percent=time_within_tolerance(cross,tolerance),
                    tracking_post10_percent=time_within_tolerance(tracking[post],tolerance),path_post10_percent=time_within_tolerance(cross[post],tolerance),
                    path_distance_rmse_m=np.sqrt(np.mean(cross**2))))
                points.append(pd.DataFrame(dict(sea=sea,trajectory=case,controller=ctrl,time=df.time,tracking_error_m=tracking,path_distance_m=cross)))
    data=pd.DataFrame(rows);data.to_csv(tables/'time_within_tolerance.csv',index=False);pd.concat(points,ignore_index=True).to_csv(tables/'path_distance_timeseries.csv',index=False)
    data.groupby(['sea','controller'])[['tracking_time_percent','path_time_percent']].mean().to_csv(tables/'tolerance_summary.csv')
    plot_tolerance(data,figs,tolerance,controllers)
    (tables/'path_tolerance_provenance.json').write_text(json.dumps(dict(data_sha256=hashes,tolerance_m=tolerance,
        window_seconds=[0,float(cfg['experiment']['steps'])*float(cfg['experiment']['dt'])],dimensions='3-D, includes depth error',geometric_definition='Minimum distance to the complete closed reference path: exact segment projection for polygons, analytic circle distance, 4096-segment sine eight approximation (bound < 0.7*a micrometres, where a is the amplitude in metres).',
        caveat='Geometric occupancy alone does not enforce timing or progress; a stationary point on the path may score 100%. Synchronous tracking occupancy is reported alongside it. Means across the selected geometries per sea, one seed per condition, no independent-seed confidence interval.',
        reference='https://www.mathworks.com/help/uav/ref/waypointfollower.html'),indent=2))
    if paper:
        paper=Path(paper);f=paper/('figs/current/wave_article_ft' if include_ft else 'figs/current/wave_article');t=paper/('tables/wave_article_ft' if include_ft else 'tables/wave_article');f.mkdir(exist_ok=True,parents=True);t.mkdir(exist_ok=True,parents=True)
        for ext in ['png','pdf']:shutil.copy2(figs/f'time_within_tolerance.{ext}',f/f'time_within_tolerance.{ext}')
        for file in ['time_within_tolerance.csv','tolerance_summary.csv','path_tolerance_provenance.json']:shutil.copy2(tables/file,t/file)
    print(data.groupby('controller')[['tracking_time_percent','path_time_percent']].mean().to_string())
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--tolerance',type=float,default=.2);p.add_argument('--paper',type=Path)
    p.add_argument('--include-ft',action='store_true')
    p.add_argument('--controllers',nargs='+',choices=CONTROLLERS)
    a=p.parse_args();generate(a.campaign,a.tolerance,a.paper,a.include_ft,a.controllers)
