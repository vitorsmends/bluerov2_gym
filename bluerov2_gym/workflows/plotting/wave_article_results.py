"""Generate a complete, separately staged W1--W3 article figure/table package."""
from pathlib import Path
import argparse,hashlib,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ..configuration import read_yaml
from .article import config,plot_rmse,plot_energy,plot_cpu_time,plot_pareto,plot_tracking_error,plot_yaw_error

ROOT=Path(__file__).resolve().parents[3]
SEAS=['walker_w1','walker_w2','walker_w3']
CASES=['figure_eight_depth','square','circle','star']
LABELS={'figure_eight_depth':'Figure eight','square':'Square','circle':'Circle','star':'Star'}
CONTROLLERS=['PPO','NMPC','PID','SMC']


def generate(campaign,paper=None,include_ft=False,selected_controllers=None):
    campaign=Path(campaign).resolve();run=campaign/'benchmark'
    if read_yaml(run/'manifest.yaml')['status']!='completed':raise ValueError('Campaign is not complete')
    resolved=read_yaml(run/'config.resolved.yaml')
    cases=[x['name'] for x in resolved['trajectories'] if x.get('enabled',True)]
    if not cases or set(cases)-set(CASES):raise ValueError('Unsupported/empty trajectory selection')
    out=campaign/'figures';tables=campaign/'tables';out.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    controllers = selected_controllers or (['PPO','FT','NMPC','PID','SMC'] if include_ft else CONTROLLERS)
    expected_steps=int(resolved['experiment']['steps']);dt=float(resolved['experiment']['dt']);duration=expected_steps*dt
    definitions={c:(dict(prefix='ft_',color='#E69F00') if c=='FT' else config.CONTROLLERS[c]) for c in controllers}
    curves=pd.read_csv(ROOT/'bluerov2_gym/workflows/plotting/assets/t200_performance_16v.csv').sort_values('force_n')
    frames={};rows=[];hashes={}
    for sea in SEAS:
        for case in cases:
            for ctrl in controllers:
                path=run/f'data/{sea}/{ctrl.lower()}_{case}.csv';df=pd.read_csv(path)
                assert len(df)==expected_steps and df.repetition_id.nunique()==1
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                df['yaw_abs_deg']=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error))))
                frames[sea,case,ctrl]=df
                power=np.interp(df[[f'T{i}' for i in range(1,7)]],curves.force_n,curves.power_w).sum(axis=1)
                rows.append(dict(sea=sea,trajectory=case,controller=ctrl,seed=int(df.seed.iloc[0]),
                    rmse_m=np.sqrt(np.mean(df.tracking_error_m**2)),
                    post10_rmse_m=np.sqrt(np.mean(df.loc[df.time>=10,'tracking_error_m']**2)),
                    yaw_mae_deg=df.yaw_abs_deg.mean(),post10_yaw_mae_deg=df.loc[df.time>=10,'yaw_abs_deg'].mean(),
                    cpu_ms=df.controller_cpu_time_s.mean()*1000,energy_J=power.sum()*dt,
                    saturation_fraction=np.mean(np.abs(df[[f'T{i}' for i in range(1,7)]].to_numpy())>=39.999)))
    metrics=pd.DataFrame(rows);metrics.to_csv(tables/'run_metrics.csv',index=False)
    values=['rmse_m','yaw_mae_deg','cpu_ms','energy_J']
    metrics.groupby(['sea','controller'])[values].agg(['mean',lambda v:v.std(ddof=0)]).rename(columns={'<lambda_0>':'std'}).to_csv(tables/'sea_summary.csv')
    metrics.groupby('controller')[values].mean().to_csv(tables/'pooled_summary.csv')
    settings=dict(controller_definitions=definitions,scenarios={f'W{i}':run/'data'/s for i,s in enumerate(SEAS,1)},
        output_dir=out/'article',table_dir=tables/'article',trajectory_names=cases,pool_scenarios=True,aggregate_scenarios=True)
    for module in [plot_rmse,plot_yaw_error,plot_tracking_error,plot_cpu_time,plot_pareto]:module.main(settings)
    # Select the displayed sea by the PPO mean RMSE, retaining all seas in summaries.
    representative=metrics[metrics.controller=='PPO'].groupby('sea').rmse_m.mean().idxmin()
    sea_label=representative.replace('walker_', '').upper()
    main=dict(settings,scenarios={sea_label:run/'data'/representative},output_dir=out/('main_'+sea_label.lower()),table_dir=tables/('main_'+sea_label.lower()))
    for module in [plot_tracking_error,plot_yaw_error]:module.main(main)
    def save(fig,name):
        for ext in ['png','pdf']:
            fig.savefig(out/f'{name}.{ext}',dpi=250,**({} if name.endswith('_3d') else {'bbox_inches':'tight'}))
        plt.close(fig)
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42}):
        # Summary distributions are across four geometries per sea, NOT seeds.
        fig,axes=plt.subplots(1,2,figsize=(9,4),layout='constrained')
        for ax,col,label in [(axes[0],'rmse_m','Position RMSE [m]'),(axes[1],'yaw_mae_deg','Yaw MAE [deg]')]:
            for offset,ctrl in zip(np.linspace(-.32,.32,len(controllers)),controllers):
                color=definitions[ctrl]['color'];boxes=[metrics.loc[(metrics.sea==sea)&(metrics.controller==ctrl),col] for sea in SEAS]
                ax.boxplot(boxes,positions=np.arange(3)+offset,widths=.64/len(controllers)*.8,manage_ticks=False,patch_artist=True,boxprops={'facecolor':color,'alpha':.65},medianprops={'color':'black'},flierprops={'marker':'.','markersize':3})
                ax.plot([],[],lw=5,color=color,label=ctrl)
            ax.set_xticks(range(3),['W1','W2','W3']);ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        axes[0].legend(frameon=False,fontsize=10,ncol=2);save(fig,'wave_boxplots_position_yaw')
        for sea in SEAS:
            spatial_controllers = [c for c in controllers if c in {'PPO','FT','NMPC'}]
            # Compact XY view is appropriate for a constant-depth reference.
            fig,axes=plt.subplots(1,3,figsize=(11,4.3)) if len(cases)==3 else plt.subplots(2,2,figsize=(8,6.5));handles=[]
            for ax,case in zip(axes.flat,cases):
                ref=frames[sea,case,'PPO'];handles=ax.plot(ref.x_ref,ref.y_ref,'k--',lw=1.2,label='Reference')
                for ctrl in controllers:
                    df=frames[sea,case,ctrl];handles+=ax.plot(df.x,df.y,lw=1.1,color=definitions[ctrl]['color'],label=ctrl)
                ax.set_aspect('equal',adjustable='box');ax.set_title(LABELS[case],fontsize=12);ax.grid(alpha=.2);ax.set_xlabel('X [m]');ax.set_ylabel('Y [m]');ax.tick_params(labelsize=10)
            fig.legend(handles=handles,loc='upper center',ncol=len(controllers)+1,frameon=False,fontsize=10);fig.tight_layout(rect=[0,0,1,.92]);save(fig,sea+'_trajectories_xy')
            fig=plt.figure(figsize=(11,4.6) if len(cases)==3 else (9,7));handles=[]
            for i,case in enumerate(cases,1):
                ax=fig.add_subplot(1,3,i,projection='3d') if len(cases)==3 else fig.add_subplot(2,2,i,projection='3d');ref=frames[sea,case,'PPO'];handles=ax.plot(ref.x_ref,ref.y_ref,ref.z_ref,'k--',lw=1,label='Reference')
                for ctrl in spatial_controllers:
                    df=frames[sea,case,ctrl];handles+=ax.plot(df.x,df.y,df.z,lw=1,color=definitions[ctrl]['color'],label=ctrl)
                ax.set_title(LABELS[case],fontsize=12,pad=0);ax.set_xlabel('X [m]',labelpad=0);ax.set_ylabel('Y [m]',labelpad=0);ax.set_zlabel('Z [m]',labelpad=0);ax.view_init(23,-55);ax.set_box_aspect((1,1,.7));ax.tick_params(labelsize=8,pad=0);ax.invert_zaxis()
            fig.legend(handles=handles,loc='upper center',ncol=len(spatial_controllers)+1,frameon=False,fontsize=10);fig.subplots_adjust(left=.01,right=.88,bottom=.08,top=.90,hspace=.12,wspace=.20);save(fig,sea+'_trajectories_3d')
            # Full time histories including the initial transient, per geometry.
            fig,axes=plt.subplots(len(cases),2,figsize=(10,2.5*len(cases)),sharex=True,squeeze=False)
            for i,case in enumerate(cases):
                for ctrl in controllers:
                    df=frames[sea,case,ctrl];color=definitions[ctrl]['color'];axes[i,0].plot(df.time,df.tracking_error_m,color=color,lw=1,label=ctrl);axes[i,1].plot(df.time,df.yaw_abs_deg,color=color,lw=1,label=ctrl)
                axes[i,0].set_ylabel(LABELS[case]+'\nPosition error [m]');axes[i,1].set_ylabel('Yaw error [deg]')
                for ax in axes[i]:ax.grid(alpha=.2);ax.set_xlim(0,duration);ax.tick_params(labelsize=10)
            for ax in axes[-1]:ax.set_xlabel('Time [s]')
            fig.legend(*axes[0,0].get_legend_handles_labels(),loc='upper center',ncol=len(controllers),frameon=False);fig.tight_layout(rect=[0,0,1,.95]);save(fig,sea+'_errors_by_trajectory')
    lines=[r'\begin{table*}[tbp]',r'\centering\small',
        fr'\caption{{Full-horizon metrics: mean $\pm$ population standard deviation across {len(cases)} geometries per sea. Each condition uses one paired wave realization (seed 42); dispersion does not estimate independent-seed uncertainty. Energy is interpolation-based; CPU measurements are execution-dependent.}}',
        r'\label{tab:wave_benchmark_results}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Case & Controller & RMSE (m) & Yaw MAE ($^\circ$) & CPU (ms) & Energy (J) \\',r'\midrule']
    for i,sea in enumerate(SEAS,1):
        for ctrl in controllers:
            g=metrics[(metrics.sea==sea)&(metrics.controller==ctrl)];cells=[f'W{i}',ctrl]
            for col,digits in [('rmse_m',3),('yaw_mae_deg',1),('cpu_ms',3),('energy_J',0)]:cells.append(f'${g[col].mean():.{digits}f} \\pm {g[col].std(ddof=0):.{digits}f}$')
            lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular}',r'\end{table*}']);(tables/'wave_benchmark_results.tex').write_text('\n'.join(lines)+'\n')
    # Compact publication table and absolute/relative errors; raw energy data remain archived.
    from .relative_tracking import generate as relative_summary
    relative_summary(campaign)
    provenance=dict(data_sha256=hashes,records=len(metrics),geometries=cases,seed=42,duration_s=duration,controllers=controllers,
        representative_sea=representative, representative_selection='Lowest PPO mean full-horizon position RMSE across selected geometries; descriptive, retrospective selection.',
        statistical_scope='One realization per condition; summary SD is across geometries. Temporal curves pooled across conditions are not independent-seed uncertainty.',
        energy='Estimated electrical energy by interpolation of the archived T200 performance curve; not measured consumption.',
        timing='Concurrent workers; process CPU times are execution-dependent; NMPC eight uses archived measurements.',
        training_status=read_yaml(campaign/'inputs/training_status.yaml')['status'])
    (tables/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if paper:
        paper=Path(paper);dest=paper/('figs/current/wave_article_ft' if include_ft else 'figs/current/wave_article');dest.mkdir(parents=True,exist_ok=True)
        for p in out.rglob('*'):
            if p.suffix not in {'.png','.pdf'}:continue
            target=dest/p.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
        dest=paper/('tables/wave_article_ft' if include_ft else 'tables/wave_article');dest.mkdir(parents=True,exist_ok=True)
        for p in tables.rglob('*'):
            if not p.is_file():continue
            target=dest/p.relative_to(tables);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    print(metrics.groupby(['sea','controller'])[['rmse_m','yaw_mae_deg']].mean().to_string())
    print('Figures:',out,'Tables:',tables)
    return metrics


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--paper',type=Path);p.add_argument('--include-ft',action='store_true')
    p.add_argument('--controllers',nargs='+',choices=CONTROLLERS)
    a=p.parse_args();generate(a.campaign,a.paper,a.include_ft,a.controllers)
