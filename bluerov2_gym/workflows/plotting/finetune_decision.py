"""Paired before/after fine-tuning trade-offs using existing held-out wave CSVs."""
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ..configuration import read_yaml

ROOT=Path(__file__).resolve().parents[3]
CASES=['reference_889_figure_eight','square','circle','star']
NAMES=['Figure-eight','Square','Circle','Star','Overall (equal weights)']


def generate(before, before_star, after, output):
    before,before_star,after,output=map(Path,[before,before_star,after,output])
    output.mkdir(parents=True,exist_ok=True)
    after_cfg=read_yaml(after/'config.resolved.yaml')
    rows=[];hashes={};common_seeds=None
    for case in CASES:
        source=before_star if case=='star' else before
        for run in [source,after]:
            if read_yaml(run/'manifest.yaml').get('status')!='completed':
                raise ValueError(f'Evaluation not completed: {run}')
        old_cfg=read_yaml(source/'config.resolved.yaml')
        for key in ['experiment']:
            if old_cfg[key]!=after_cfg[key]:raise ValueError(f'Different paired {key}.')
        for key in ['config','dynamics_config','yaw_tracking']:
            if old_cfg['environment'].get(key)!=after_cfg['environment'].get(key):
                raise ValueError(f'Different paired simulator setting: {key}')
        if old_cfg['disturbance']['scenarios']['moderate']!=after_cfg['disturbance']['scenarios']['moderate']:
            raise ValueError('Different Moderate disturbance parameters.')
        paths=[source/f'data/moderate/ppo_{case}.csv',after/f'data/moderate/ppo_{case}.csv']
        old,new=[pd.read_csv(p).sort_values(['seed','time']).reset_index(drop=True) for p in paths]
        for p in paths:hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        columns=['seed','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']
        columns += [c for c in old.columns if c.endswith('_ref')]
        np.testing.assert_allclose(old[columns],new[columns],rtol=0,atol=1e-10)
        seeds=sorted(map(int,old.seed.unique()))
        if len(seeds)!=20 or (common_seeds is not None and seeds!=common_seeds):
            raise ValueError('Expected the same 20 paired wave seeds in all four geometries.')
        states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
        np.testing.assert_allclose(old.groupby('seed').first()[states],new.groupby('seed').first()[states],rtol=0,atol=1e-10)
        common_seeds=seeds
        for name,frame in [('before',old),('after',new)]:
            for seed,group in frame.groupby('seed'):
                if len(group)!=1000 or not np.allclose(group.time,np.arange(1000)*.1,atol=1e-9):
                    raise ValueError('Incomplete evaluation horizon.')
                error=group.tracking_error_m.to_numpy()
                yaw=np.degrees(np.abs(np.arctan2(np.sin(group.yaw_error),np.cos(group.yaw_error))))
                if not np.isfinite(error).all() or not np.isfinite(yaw).all():raise ValueError('Non-finite errors.')
                rows.append(dict(policy=name,trajectory=case,seed=int(seed),
                                 rmse_m=float(np.sqrt(np.mean(error**2))),yaw_mae_deg=float(yaw.mean())))
    per_seed=pd.DataFrame(rows);per_seed.to_csv(output/'paired_seed_metrics.csv',index=False)
    metrics=['rmse_m','yaw_mae_deg'];values={}
    for name in ['before','after']:
        values[name]=np.stack([per_seed[(per_seed.policy==name)&(per_seed.trajectory==case)]
                              .set_index('seed').reindex(common_seeds)[metrics].to_numpy() for case in CASES],axis=1)
    # Resample entire seed blocks, preserving before/after pairing and dependence
    # across the four geometries that share the same wave realization.
    rng=np.random.default_rng(4242);indices=rng.integers(0,len(common_seeds),size=(10000,len(common_seeds)))
    summaries=[]
    for index,name in enumerate(CASES+['overall']):
        b=values['before'][:,index,:] if index<4 else values['before'].mean(axis=1)
        a=values['after'][:,index,:] if index<4 else values['after'].mean(axis=1)
        bm,am=b.mean(axis=0),a.mean(axis=0)
        reduction=100*(bm-am)/bm
        boot_b,boot_a=b[indices].mean(axis=1),a[indices].mean(axis=1)
        bootstrap=100*(boot_b-boot_a)/boot_b
        lo,hi=np.quantile(bootstrap,[.025,.975],axis=0)
        for j,key in enumerate(metrics):
            summaries.append(dict(trajectory=name,metric=key,before_mean=float(bm[j]),after_mean=float(am[j]),
                                  reduction_percent=float(reduction[j]),ci95_low=float(lo[j]),ci95_high=float(hi[j]),
                                  wave_seeds=len(common_seeds)))
    summary=pd.DataFrame(summaries);summary.to_csv(output/'summary.csv',index=False)
    with plt.rc_context({'font.family':'serif','font.size':11,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(8.6,4.3),sharey=True,layout='constrained')
        for ax,key,title,unit,digits in zip(axes,metrics,['Position RMSE','Yaw MAE'],['m','deg'],[3,1]):
            df=summary[summary.metric==key].set_index('trajectory').reindex(CASES+['overall'])
            points=df.reduction_percent.to_numpy();low=df.ci95_low.to_numpy();high=df.ci95_high.to_numpy()
            colors=np.where(points>=0,'#009e73','#d55e00')
            y=np.arange(5)
            ax.axhspan(3.55,4.45,color='.93',zorder=0)
            ax.barh(y,points,height=.5,color=colors,alpha=.8,zorder=2)
            for i,(point,l,h) in enumerate(zip(points,low,high)):
                ax.plot([l,h],[i,i],color='black',lw=1.2,zorder=3)
                ax.plot([l,l],[i-.07,i+.07],color='black',lw=1,zorder=3)
                ax.plot([h,h],[i-.07,i+.07],color='black',lw=1,zorder=3)
                ax.annotate(f'{point:+.1f}%',(max(point,h) if point>=0 else min(point,l),i),
                            xytext=(5 if point>=0 else -5,0),textcoords='offset points',
                            va='center',ha='left' if point>=0 else 'right',fontsize=10,
                            fontweight='bold' if i==4 else 'normal')
            ax.axvline(0,color='black',lw=.8);ax.axhline(3.5,color='.6',lw=.7)
            ax.set_title(title,fontsize=12)
            ax.set_yticks(y,NAMES);ax.grid(axis='x',ls=':',alpha=.3);ax.set_axisbelow(True)
            ax.margins(x=.25)
        axes[0].invert_yaxis()
        for ax in axes:
            for tick in ax.get_yticklabels():
                if tick.get_text().startswith('Overall'):tick.set_fontweight('bold')
        fig.supxlabel('Error reduction [%]   (+ improvement / − degradation)',fontsize=11)
        for ext in ['pdf','png']:fig.savefig(output/f'finetuning_decision.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    # The controller benchmark excludes star and includes three prescribed seas.
    # Its weights differ from the four-geometry Moderate diagnostic above.
    campaign_file=after.parent/'campaign.yaml'
    if campaign_file.exists():
        from ..evaluation.finetune_campaign import benchmark_summary
        campaign=read_yaml(campaign_file)
        baseline_run=Path(campaign['spec']['baseline'])
        benchmark_summary(baseline_run,output/'baseline_benchmark')
        benchmark_summary(after.parent/'combined_benchmark',output/'after_benchmark')
        benchmark_before=output/'baseline_benchmark/benchmark_per_trajectory.csv'
        benchmark_after=output/'after_benchmark/benchmark_per_trajectory.csv'
        old_b=pd.read_csv(benchmark_before).rename(columns={'scenario':'sea'})
        new_b=pd.read_csv(benchmark_after)
        old_b=old_b[old_b.controller.str.lower()=='ppo']
        new_b=new_b[new_b.controller.str.lower()=='ppo']
        keys=['sea','trajectory','repetition_id']
        merged=old_b.merge(new_b,on=keys,suffixes=('_before','_after'),validate='one_to_one')
        if len(merged)!=27:raise ValueError('Expected 27 paired article PPO runs.')
        benchmark_rows=[]
        for key,label,unit in [('rmse_m','Position RMSE','m'),('yaw_mae_deg','Yaw MAE','deg'),('energy_J','Estimated energy','J')]:
            b=float(merged[key+'_before'].mean());a=float(merged[key+'_after'].mean())
            benchmark_rows.append(dict(metric=key,label=label,unit=unit,before=b,after=a,reduction_percent=100*(b-a)/b))
        result=pd.DataFrame(benchmark_rows);result.to_csv(output/'article_benchmark_summary.csv',index=False)
        with plt.rc_context({'font.family':'serif','font.size':11,'pdf.fonttype':42,
                             'axes.spines.top':False,'axes.spines.right':False}):
            fig,ax=plt.subplots(figsize=(7.2,3.2),layout='constrained')
            reductions=result.reduction_percent.to_numpy();y=np.arange(len(result))
            ax.barh(y,reductions,color=np.where(reductions>=0,'#009e73','#d55e00'),alpha=.8,height=.5)
            ax.axvline(0,color='black',lw=.8)
            for i,row in enumerate(result.itertuples()):
                sign=1 if row.reduction_percent>=0 else -1
                ax.annotate(f'{row.reduction_percent:+.1f}%',(row.reduction_percent,i),
                            xytext=(sign*6,0),textcoords='offset points',ha='left' if sign>0 else 'right',va='center',fontweight='bold')
            labels=[f'{r.label}\n{r.before:.3f} → {r.after:.3f} {r.unit}' if r.unit!='J'
                    else f'{r.label}\n{r.before:.0f} → {r.after:.0f} J' for r in result.itertuples()]
            ax.set_yticks(y,labels);ax.invert_yaxis();ax.margins(x=.25)
            ax.set_xlabel('Reduction vs. C4 [%]   (+ improvement / − degradation)')
            ax.grid(axis='x',ls=':',alpha=.3);ax.set_axisbelow(True)
            for ext in ['pdf','png']:fig.savefig(output/f'article_benchmark_decision.{ext}',dpi=300,bbox_inches='tight')
            plt.close(fig)
        hashes[str(benchmark_before)]=hashlib.sha256(benchmark_before.read_bytes()).hexdigest()
        hashes[str(benchmark_after)]=hashlib.sha256(benchmark_after.read_bytes()).hexdigest()
    overall=summary[summary.trajectory=='overall']
    report=['# Fine-tuning decision summary','',
            'Twenty paired wave realizations per geometry in Moderate conditions; same plant, references and initial states.',
            'Positive percentages mean lower error after fine-tuning. The overall means weight each geometry equally.',
            'Position and yaw are kept separate; metres and degrees are never combined into an arbitrary score.','',
            '| Metric | Before | After | Error reduction |','|---|---:|---:|---:|']
    for row in overall.itertuples():
        report.append(f'| {row.metric} | {row.before_mean:.6f} | {row.after_mean:.6f} | {row.reduction_percent:+.2f}% |')
    report += ['', 'Whiskers are 95% percentile intervals from 10,000 paired-bootstrap seed-block resamples.',
               'They describe environmental-seed uncertainty for these fixed policies, not independently trained agents.',
               'The separate article_benchmark_summary.csv/figure uses the exact controller benchmark: three trajectories (eight, square, circle), three seas and three repetitions. Star is excluded there. Estimated energy is not measured electrical consumption.',
               'The star training ended by the configured RMSE-plateau criterion during FT1; FT2 was not reached.',
               'Choosing a checkpoint on these results makes this a model-selection set; a subsequent independent test can assess the chosen policy.']
    (output/'README.md').write_text('\n'.join(report)+'\n')
    (output/'provenance.json').write_text(json.dumps(dict(csv_sha256=hashes,seeds=common_seeds,
        weighting='Equal trajectories; equal wave seeds; mean of full-horizon run-level RMSE values.',
        bootstrap='Paired seed blocks, 10000 resamples, RNG 4242, percentile CI95.',independent_training_runs=False),indent=2)+'\n')
    print(summary.to_string(index=False))
    return output/'finetuning_decision.png'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before',type=Path,default=ROOT/'artifacts/evaluation/physics_wave_curriculum_yaw_seed_robustness')
    parser.add_argument('--before-star',type=Path,default=ROOT/'artifacts/evaluation/physics_wave_curriculum_yaw_seed_star')
    parser.add_argument('--after',type=Path,default=ROOT/'artifacts/evaluation/star_finetune_validation/seed_robustness')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluation/star_finetune_validation/finetuning_decision')
    args=parser.parse_args();print(generate(args.before,args.before_star,args.after,args.output))
