"""Regenerate article comparisons with original PPO and FT as distinct policies."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from .article import config, plot_rmse, plot_energy, plot_cpu_time, plot_pareto, plot_tracking_error, plot_yaw_error
from ..evaluation.finetune_campaign import compatible_subset, checked_frame
from ..configuration import read_yaml
ROOT=Path(__file__).resolve().parents[3]
CASES=['reference_889_figure_eight','square','circle']


def generate(campaign, output, paper):
    baseline=ROOT/'artifacts/evaluation/physics_wave_curriculum_yaw_sea_states'
    tuned=campaign/'ppo_benchmark'
    old,new=[read_yaml(p/'config.resolved.yaml') for p in [baseline,tuned]]
    compatible_subset(old,new)
    output.mkdir(parents=True,exist_ok=True)
    hashes={};rows=[]
    curve=pd.read_csv(ROOT/'bluerov2_gym/workflows/plotting/assets/t200_performance_16v.csv').sort_values('force_n')
    for sea in ['calm','moderate','rough']:
        folder=output/'data'/sea;folder.mkdir(parents=True,exist_ok=True)
        for case in CASES:
            reference=None
            for ctrl in ['pid','smc','nmpc','ppo','ft']:
                path=(tuned if ctrl=='ft' else baseline)/f'data/{sea}/{"ppo" if ctrl=="ft" else ctrl}_{case}.csv'
                df=checked_frame(path,new if ctrl=='ft' else old,case).sort_values(['repetition_id','time']).reset_index(drop=True)
                cols=['seed','repetition_id','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']+[c for c in df if c.endswith('_ref')]
                if reference is not None:np.testing.assert_allclose(reference[cols],df[cols],atol=1e-10,rtol=0)
                else:reference=df
                hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                shutil.copy2(path,folder/f'{ctrl}_{case}.csv')
                for rep,g in df.groupby('repetition_id'):
                    yaw=np.degrees(np.abs(np.arctan2(np.sin(g.yaw_error),np.cos(g.yaw_error))))
                    rows.append(dict(sea=sea,trajectory=case,controller=ctrl.upper(),repetition_id=rep,
                        rmse_m=np.sqrt(np.mean(g.tracking_error_m**2)),yaw_mae_deg=yaw.mean(),
                        cpu_ms=g.controller_cpu_time_s.mean()*1000,
                        energy_J=np.interp(g[[f'T{i}' for i in range(1,7)]],curve.force_n,curve.power_w).sum()*old['experiment']['dt']))
    table=pd.DataFrame(rows);tables=output/'tables';tables.mkdir(exist_ok=True)
    table.to_csv(tables/'run_metrics.csv',index=False)
    table.groupby('controller')[['rmse_m','yaw_mae_deg','cpu_ms','energy_J']].mean().to_csv(tables/'pooled_summary.csv')
    defs=dict(config.CONTROLLERS);defs['FT']={'prefix':'ft_','color':'#E69F00'}
    settings=dict(controller_definitions=defs,scenarios={s.capitalize():output/'data'/s for s in ['calm','moderate','rough']},
                  output_dir=output/'figures',table_dir=tables,trajectory_names=CASES,pool_scenarios=True,aggregate_scenarios=True)
    for module in [plot_rmse,plot_energy,plot_cpu_time,plot_pareto,plot_tracking_error,plot_yaw_error]:module.main(settings)
    for name in ['boxplot_rmse_error','cumulative_energy_mean_std','fig_path_tracking_cpu_time_mean_std','pareto_rmse_vs_cpu_time_mean_std']:
        for ext in ['pdf','png']:
            dest=paper/'figs/current/article'/f'{name}.{ext}';shutil.copy2(output/'figures'/f'{name}.{ext}',dest)
    for name in ['path_tracking_error_mean_std','yaw_error_mean_std']:
        for ext in ['pdf','png']:shutil.copy2(output/'figures'/f'{name}.{ext}',paper/'figs/current/main_moderate'/f'{name}.{ext}')
    lines=[r'\begin{table*}[pos=tbp]',r'\centering\small',r'\caption{PPO, FT, and reference controllers over nine trajectory--repetition pairs per sea: mean $\pm$ population standard deviation. FT is evaluated in a separate execution; CPU timings do not constitute a controlled cross-campaign speed comparison. Energy is an interpolation-based estimate.}',r'\label{tab:benchmark_results}',r'\begin{tabular}{llrrrr}',r'\toprule',r'Sea & Controller & Position RMSE (m) & Yaw MAE ($^\circ$) & CPU (ms) & Energy (J) \\',r'\midrule']
    for sea in ['calm','moderate','rough']:
        for ctrl in ['PID','SMC','NMPC','PPO','FT']:
            g=table[(table.sea==sea)&(table.controller==ctrl)];cells=[sea.capitalize(),ctrl]
            for col,digits in [('rmse_m',3),('yaw_mae_deg',1),('cpu_ms',4),('energy_J',0)]:
                cells.append(f'${g[col].mean():.{digits}f} \\pm {g[col].std(ddof=0):.{digits}f}$')
            lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular}',r'\end{table*}'])
    (tables/'benchmark_results.tex').write_text('\n'.join(lines)+'\n')
    shutil.copy2(tables/'benchmark_results.tex',paper/'tables/benchmark_results.tex')
    (output/'provenance.json').write_text(json.dumps({'sha256':hashes,'episodes_per_controller':27,'trajectories':CASES,'cpu_limitation':'FT evaluated in a separate execution; no controlled hardware timing comparison'},indent=2))
    print(table.groupby('controller')[['rmse_m','yaw_mae_deg','energy_J','cpu_ms']].mean())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,default=ROOT/'artifacts/evaluation/star_finetune_validation')
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/star_finetune_validation/article_all_controllers')
    p.add_argument('--paper',type=Path,default=ROOT/'documents/oceans2026')
    a=p.parse_args();generate(a.campaign,a.output,a.paper)
