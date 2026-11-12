"""Evaluate curriculum checkpoints, replace only PPO CSVs, and stage article figures.

python -m bluerov2_gym.workflows.evaluation.finetune_campaign --phase plan
See docs/finetune_evaluation.md. No manuscript is overwritten by this workflow.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ..configuration import read_yaml, resolve, evaluation_plan
from ..artifacts import create_run, update_manifest, write_yaml

ROOT = Path(__file__).resolve().parents[3]
LABELS = {'stage0_no_waves': 'A: no waves', 'preparation_waves': 'B: preparation waves',
          'curriculum_eight': 'C4: figure-eight', 'star_finetune': 'FT: star'}
CASES = {'reference_889_figure_eight':'Figure-eight','square':'Square','circle':'Circle','star':'Star'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'))


def completed(run):
    if read_yaml(Path(run)/'manifest.yaml').get('status') != 'completed':
        raise ValueError(f'Run is not completed: {run}')


def pair(run, suffix=''):
    cfg = read_yaml(run/'config.resolved.yaml')
    model = run/'models'/f"{cfg['paths']['model_name']}{suffix}.zip"
    norm = Path(cfg['paths']['vecnormalize_name'])
    normalizer = run/'models'/f'{norm.stem}{suffix}{norm.suffix}'
    for file in [model, normalizer]:
        if not file.is_file(): raise FileNotFoundError(file)
    return dict(model_path=str(model), vecnormalize_path=str(normalizer))


def scenarios(cfg):
    d = cfg['disturbance']
    return d.get('scenario_names') or [d['default_scenario']]


def cases(cfg):
    return {t['name']: t.get('params', {}) for t in cfg['trajectories'] if t.get('enabled', True)}


def simulation_contract(cfg):
    env = cfg['environment']
    return dict(experiment=cfg['experiment'],
                environment={k:v for k,v in env.items() if k != 'training_run'},
                scenarios={s:cfg['disturbance']['scenarios'][s] for s in scenarios(cfg)},
                trajectories=cases(cfg))


def compatible_subset(baseline, new):
    """Exact configured plant/protocol, allowing selection of a baseline subset."""
    a,b = simulation_contract(baseline), simulation_contract(new)
    for key in ['experiment', 'environment']:
        if a[key] != b[key]: raise ValueError(f'Cannot reuse baselines: different {key}.')
    for key in ['scenarios', 'trajectories']:
        if any(a[key].get(name) != value for name,value in b[key].items()):
            raise ValueError(f'Cannot reuse baselines: different {key}.')


def prepare(path, output=None):
    spec = read_yaml(path)
    for key in ['source_training','finetuned_training','baseline','baseline_star','output']:
        p = Path(spec[key]).expanduser()
        spec[key] = str(p.resolve() if p.is_absolute() else (Path(path).resolve().parent/p).resolve())
    if output: spec['output'] = str(Path(output).resolve())
    source, final = Path(spec['source_training']), Path(spec['finetuned_training'])
    for run in [source, Path(spec['baseline']), Path(spec['baseline_star'])]: completed(run)
    fine_status = read_yaml(final/'manifest.yaml').get('status')
    if fine_status != 'completed':
        if not (spec.get('allow_early_stopped_finetune', spec.get('allow_incomplete_finetune', False)) is True and fine_status == 'early_stopped'
                and spec.get('finetuned_checkpoint_suffix') == '_incomplete'):
            raise ValueError(f'Fine-tuning is {fine_status}; selecting an early-stopped terminal checkpoint requires an explicit checkpoint suffix.')
    spec['finetuned_training_status'] = fine_status
    fine = read_yaml(final/'config.resolved.yaml')
    if fine.get('finetuning', {}).get('trajectory', {}).get('name') != 'star':
        raise ValueError('Expected the star-only fine-tuning run, not the discarded mixed run.')
    lineage = read_yaml(final/'models/initialization.yaml')
    source_pair = pair(source)
    for value in source_pair.values():
        if lineage['sha256'].get(Path(value).name) != sha(value):
            raise ValueError('Fine-tuning was not initialized from the selected source policy pair.')
    status = read_yaml(source/'models/curriculum_status.yaml')
    history = status['history']
    if len(history)<2 or history[0]['mechanics_model']!='std' or history[1]['mechanics_model']!='std':
        raise ValueError('Expected preparation checkpoints A and B in this source curriculum.')
    if history[0]['effective_context']['jonswap'].get('enable_wave_force', True):
        raise ValueError('First checkpoint is not the no-wave preparation stage.')
    policies = {
        'stage0_no_waves': pair(source,'_stage_1'),
        'preparation_waves': pair(source,'_stage_2'),
        'curriculum_eight': source_pair,
        'star_finetune': pair(final, spec.get('finetuned_checkpoint_suffix','')),
    }
    for name,policy in policies.items():
        policy['sha256'] = {Path(p).name:sha(p) for p in list(policy.values())}
    base = read_yaml(Path(spec['baseline'])/'config.resolved.yaml')
    # The original models may have been copied from a different machine; use
    # resolved contents, not their old absolute paths, for the simulator.
    env = deepcopy(fine['environment'])
    env['training_run'] = str(final)
    if {k:v for k,v in base['environment'].items() if k!='training_run'} != fine['environment']:
        raise ValueError('Fine-tuning base simulator differs from the archived article simulator.')
    for key in ['config','dynamics_config']:
        if fine['environment'][key] != read_yaml(source/'config.resolved.yaml')['environment'][key]:
            raise ValueError('Checkpoint comparison would change the simulation contract.')
    def configuration(template, policy, names=None, selected=None, repetitions=None, seed=None):
        cfg = deepcopy(template)
        cfg.pop('output',None)
        cfg['environment'] = deepcopy(env)
        ctrl = deepcopy(next(c for c in base['controllers'] if c['name']=='ppo'))
        ctrl['params'].update({k:policy[k] for k in ['model_path','vecnormalize_path']})
        ctrl['params'].update(observation_adapter='legacy_yaw_error',
            env_config_path=str(ROOT/'configs/dynamics/blue_rov_config.yaml'),
            dynamics_config_path=str(ROOT/'configs/dynamics/dynamics_config.yaml'))
        cfg['controllers'] = [ctrl]
        if names is not None: cfg['trajectories']=[dict(name=n,enabled=True,params={}) for n in names]
        if selected is not None:
            cfg['disturbance']['scenario_names']=selected
            cfg['disturbance']['default_scenario']=selected[0]
        if repetitions is not None: cfg['experiment']['repetitions']=repetitions
        if seed is not None:
            cfg['experiment']['seed']=seed
            for sea in scenarios(cfg): cfg['disturbance']['scenarios'][sea]['seed']=seed
        return cfg
    configs = {}
    check=spec['checkpoint_comparison']
    for name,policy in policies.items():
        configs[f'checkpoints/{name}']=configuration(base,policy,check['trajectories'],[check['scenario']],1,check['seed'])
    configs['ppo_benchmark']=configuration(base,policies['star_finetune'])
    star_base = read_yaml(Path(spec['baseline_star'])/'config.resolved.yaml')
    configs['ppo_star']=configuration(star_base,policies['star_finetune'],['star'],['moderate'])
    seed_spec=spec['seed_assessment']
    if seed_spec['first_seed']<2**31 or seed_spec['first_seed']+seed_spec['repetitions']>2**32:
        raise ValueError('Held-out wave seeds must fit in the upper uint32 half.')
    used=set()
    for run in [source,final]:
        for log in (run/'logs/curriculum_evaluations').glob('*.yaml'):
            for episode in read_yaml(log).get('episodes',[]):
                used.add(episode['context']['jonswap']['seed'])
    if used.intersection(range(seed_spec['first_seed'],seed_spec['first_seed']+seed_spec['repetitions'])):
        raise ValueError('Requested held-out seeds overlap curriculum assessments.')
    configs['seed_robustness']=configuration(base,policies['star_finetune'],seed_spec['trajectories'],
        [seed_spec['scenario']],seed_spec['repetitions'],seed_spec['first_seed'])
    root=Path(spec['output'])
    provenance=dict(spec=spec,policies=policies,configs=configs,
                    baseline_manifests={k:sha(Path(spec[k])/'manifest.yaml') for k in ['baseline','baseline_star']},
                    contract_source_hash=sha(Path(spec['baseline'])/'config.resolved.yaml'))
    if (root/'campaign.yaml').exists() and read_yaml(root/'campaign.yaml')!=provenance:
        raise ValueError('Campaign inputs changed. Use a new --output directory; existing results are preserved.')
    root.mkdir(parents=True,exist_ok=True)
    if fine_status != 'completed':
        (root/'RESULT_STATUS.md').write_text('Training ended by the configured validation-RMSE plateau criterion. This evaluates the terminal checkpoint; FT1 was executed and FT2 was not reached.\n')
    write_yaml(root/'campaign.yaml',provenance)
    for name,cfg in configs.items():
        file=root/'configs'/f'{name}.yaml';file.parent.mkdir(parents=True,exist_ok=True);write_yaml(file,cfg)
    return root,provenance


def run_evaluation(root,name):
    from ..cli import evaluate
    cfg_file=root/'configs'/f'{name}.yaml';run=root/name
    if run.exists():
        completed(run)
        expected=read_yaml(cfg_file)
        actual=read_yaml(run/'config.resolved.yaml');actual.pop('output',None)
        if actual!=expected:raise ValueError(f'Existing evaluation uses different inputs: {run}')
        manifest=read_yaml(run/'manifest.yaml')
        if any(sha(p)!=h for p,h in manifest['artifact_sha256'].items()):
            raise ValueError(f'Policy artifacts changed since evaluation: {run}')
        print(f'[REUSE] Completed {run}')
        return
    evaluate(SimpleNamespace(config=cfg_file,output=run))


def checked_frame(path,cfg,case):
    df=pd.read_csv(path)
    if not set(['seed','repetition_id','time','tracking_error_m','yaw_error']).issubset(df):
        raise ValueError(f'Invalid evaluation CSV: {path}')
    if set(df.repetition_id)!=set(range(cfg['experiment']['repetitions'])):
        raise ValueError(f'Missing repetitions: {path}')
    for rep,g in df.groupby('repetition_id'):
        if len(g)!=cfg['experiment']['steps'] or not np.allclose(g.time,np.arange(len(g))*cfg['experiment']['dt'],rtol=0,atol=1e-9):
            raise ValueError(f'Incomplete horizon; will not merge a truncated evaluation: {path}')
        if not (g.seed==cfg['experiment']['seed']+rep).all():raise ValueError(f'Seed mismatch: {path}')
    if not (df.trajectory==case).all():raise ValueError(f'Trajectory mismatch: {path}')
    if not np.isfinite(df.select_dtypes(include='number').to_numpy()).all():
        raise ValueError(f'Non-finite numeric data: {path}')
    return df


def merge_baselines(baseline,ppo_run,output):
    """Copy existing PID/SMC/NMPC CSVs without simulating them again."""
    baseline,ppo_run,output=map(Path,[baseline,ppo_run,output])
    for run in [baseline,ppo_run]:completed(run)
    old,new=read_yaml(baseline/'config.resolved.yaml'),read_yaml(ppo_run/'config.resolved.yaml')
    compatible_subset(old,new)
    controllers={c['name']:deepcopy(c) for c in old['controllers'] if c.get('enabled',True)}
    if set(controllers)!={'pid','smc','nmpc','ppo'}:raise ValueError('Expected exactly the four article controllers.')
    controllers['ppo']=deepcopy(new['controllers'][0])
    combined=deepcopy(new);combined['controllers']=[controllers[n] for n in ['pid','ppo','smc','nmpc']]
    combined['output']={'directory_template':str(output/'data/{scenario}')}
    inputs={};copies=[]
    for sea in scenarios(new):
        for case in cases(new):
            ppo=ppo_run/f'data/{sea}/ppo_{case}.csv';reference=checked_frame(ppo,new,case)
            copies.append((ppo,Path(f'data/{sea}/{ppo.name}')))
            for ctrl in ['pid','smc','nmpc']:
                src=baseline/f'data/{sea}/{ctrl}_{case}.csv';cached=checked_frame(src,new,case)
                # Repetition/seed/time, prescribed wave settings, references, and
                # initial states must agree; controller outputs are never compared.
                cols=['repetition_id','seed','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']
                cols += [c for c in reference if c.endswith('_ref')]
                np.testing.assert_allclose(cached[cols],reference[cols],rtol=0,atol=1e-10,
                    err_msg=f'Baseline reference/seed/disturbance mismatch: {src}')
                states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                np.testing.assert_allclose(cached.groupby('repetition_id').first()[states],
                                           reference.groupby('repetition_id').first()[states],rtol=0,atol=1e-10)
                copies.append((src,Path(f'data/{sea}/{src.name}')))
    inputs={str(p):sha(p) for p,_ in copies}
    if output.exists():
        completed(output)
        recorded=read_yaml(output/'reuse_provenance.yaml')
        if recorded['csv_sha256']!=inputs:raise ValueError('Cached inputs changed; refusing to overwrite merged results.')
        for src,rel in copies:
            if sha(output/rel)!=inputs[str(src)]:raise ValueError('Merged CSV was changed after creation.')
        print(f'[REUSE] Verified merged results {output}');return
    create_run('evaluation',combined,combined,output)
    for src,rel in copies:
        dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
    write_yaml(output/'plan.yaml',evaluation_plan(combined))
    write_yaml(output/'reuse_provenance.yaml',dict(baseline=str(baseline),ppo_run=str(ppo_run),csv_sha256=inputs,
        compatible_configuration=True,reused_controllers=['pid','smc','nmpc'],
        timing_note='Baseline CPU timings are archival; PPO timing is from a new campaign. No shared-hardware timing claim is made.',
        runtime_note='Configuration, CSV reference/seed/initial-state compatibility checked; original runtime-source hashes were not archived.'))
    records=[]
    for source,names in [(baseline,{'pid','smc','nmpc'}),(ppo_run,{'ppo'})]:
        file=source/'logs/episodes.yaml'
        if file.exists():
            records.extend(e for e in read_episode_list(file) if e['controller'] in names and
                           e['scenario'] in scenarios(new) and e['trajectory'] in cases(new))
    write_yaml(output/'logs/episodes.yaml',records)
    update_manifest(output,status='completed',reused_controllers=['pid','smc','nmpc'],csv_sha256=inputs)


def read_episode_list(file):
    import yaml
    return yaml.safe_load(Path(file).read_text()) or []


def checkpoint_plot(root,plan):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    labels=dict(LABELS)
    if plan['spec'].get('finetuned_training_status') != 'completed':
        labels['star_finetune']='FT: star'
    scenario=plan['spec']['checkpoint_comparison']['scenario'];rows=[]
    for model in LABELS:
        run=root/'checkpoints'/model;completed(run)
        cfg=read_yaml(run/'config.resolved.yaml')
        for case in cases(cfg):
            df=pd.read_csv(run/f'data/{scenario}/ppo_{case}.csv')
            records=[e for e in read_episode_list(run/'logs/episodes.yaml') if e['trajectory']==case]
            good=(len(df)==cfg['experiment']['steps'] and np.isfinite(df[['tracking_error_m','yaw_error']]).all().all() and records and
                  not records[0]['terminated'] and not records[0]['truncated'])
            errors=df.tracking_error_m.to_numpy();yaw=np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error),np.cos(df.yaw_error))))
            rows.append(dict(model=model,trajectory=case,completed=bool(good),duration_s=len(df)*cfg['experiment']['dt'],
                rmse_m=float(np.sqrt(np.mean(errors**2))) if good else None,
                yaw_mae_deg=float(yaw.mean()) if good else None,
                observed_rmse_m=float(np.sqrt(np.mean(errors**2))),seed=int(df.seed.iloc[0])))
    table=pd.DataFrame(rows);out=root/'checkpoint_comparison';out.mkdir(exist_ok=True)
    table.to_csv(out/'metrics.csv',index=False)
    trajectories=plan['spec']['checkpoint_comparison']['trajectories']
    with plt.rc_context({'font.family':'serif','font.size':10,'pdf.fonttype':42}):
        fig,axes=plt.subplots(1,2,figsize=(7.2,3.1),layout='constrained')
        for ax,key,title in zip(axes,['rmse_m','yaw_mae_deg'],['Position RMSE [m]','Yaw MAE [deg]']):
            values=table.pivot(index='model',columns='trajectory',values=key).reindex(index=LABELS,columns=trajectories).to_numpy(dtype=float)
            valid=values[np.isfinite(values)]
            if not len(valid):raise ValueError('No completed checkpoint episodes; inspect the raw results.')
            lower=max(float(valid.min()),1e-6);upper=max(float(valid.max()),lower*1.01)
            image=ax.imshow(np.ma.masked_invalid(values),cmap='YlGnBu',norm=LogNorm(vmin=lower,vmax=upper),aspect='auto')
            ax.set_xticks(range(len(trajectories)),[CASES[c] for c in trajectories],rotation=30,ha='right')
            ax.set_yticks(range(len(LABELS)),list(labels.values()) if ax is axes[0] else [])
            ax.set_title(title,fontsize=11)
            for i,name in enumerate(LABELS):
                for j,case in enumerate(trajectories):
                    row=table[(table.model==name)&(table.trajectory==case)].iloc[0]
                    value=values[i,j]
                    label=f'{value:.3f}' if key=='rmse_m' else f'{value:.1f}'
                    if not row.completed:label=f'Stop\n{row.duration_s:.0f} s'
                    ax.text(j,i,label,ha='center',va='center',fontsize=9,
                            color='white' if np.isfinite(value) and image.norm(value)>.55 else 'black')
            fig.colorbar(image,ax=ax,shrink=.8,pad=.02)
        for ext in ['pdf','png']:fig.savefig(out/f'checkpoint_comparison.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
    write_yaml(out/'provenance.yaml',dict(policies=plan['policies'],simulation_contract='All checkpoints evaluated on the same target plant and selected sea.',
        seed=plan['spec']['checkpoint_comparison']['seed'],
        interpretation='Single paired wave realization; descriptive checkpoint comparison, not a causal curriculum ablation or multi-training-seed uncertainty.',
        incomplete_episodes='Shown as Stop with duration; partial RMSE never substituted for full-horizon RMSE.'))


def plot_results(root,plan):
    from ..plotting.manager import PlotManager
    for name in ['combined_benchmark','combined_star']:
        PlotManager(run_directory=root/name).generate_suite('article')
        PlotManager(run_directory=root/name,config={'scenario':'moderate'}).generate_suite('trajectories')
    checkpoint_plot(root,plan)


def benchmark_summary(run, output):
    """Same run weighting and 16 V thrust/power interpolation as the article."""
    run,output=Path(run),Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=read_yaml(run/'config.resolved.yaml');dt=cfg['experiment']['dt']
    curve=pd.read_csv(ROOT/'bluerov2_gym/workflows/plotting/assets/t200_performance_16v.csv').sort_values('force_n')
    rows=[]
    for sea in scenarios(cfg):
        for case in cases(cfg):
            for ctrl in ['pid','ppo','smc','nmpc']:
                frame=checked_frame(run/f'data/{sea}/{ctrl}_{case}.csv',cfg,case)
                for rep,g in frame.groupby('repetition_id'):
                    yaw=np.degrees(np.abs(np.arctan2(np.sin(g.yaw_error),np.cos(g.yaw_error))))
                    thrust=g[[f'T{i}' for i in range(1,7)]].to_numpy()
                    power=np.interp(thrust,curve.force_n,curve.power_w)
                    rows.append(dict(sea=sea,trajectory=case,controller=ctrl.upper(),repetition_id=int(rep),
                        rmse_m=float(np.sqrt(np.mean(g.tracking_error_m**2))),yaw_mae_deg=float(yaw.mean()),
                        yaw_rmse_deg=float(np.sqrt(np.mean(yaw**2))),cpu_ms=float(g.controller_cpu_time_s.mean()*1000),
                        energy_J=float(power.sum()*dt)))
    frame=pd.DataFrame(rows);frame.to_csv(output/'benchmark_per_trajectory.csv',index=False)
    metrics=['rmse_m','yaw_mae_deg','yaw_rmse_deg','cpu_ms','energy_J']
    def summarize(groups):
        return frame.groupby(groups)[metrics].agg(['mean',lambda x:np.std(x,ddof=0)]).rename(columns={'<lambda_0>':'std'})
    summary=summarize(['sea','controller']);summary.to_csv(output/'benchmark_summary.csv')
    summarize(['controller']).to_csv(output/'benchmark_pooled.csv')
    lines=[r'\begin{table*}[pos=tbp]',r'\centering\small',
        r'\caption{Controller comparison after star fine-tuning: mean $\pm$ population standard deviation over nine trajectory--repetition pairs per sea. PID/SMC/NMPC data and CPU timings are retained from the archived campaign; PPO is evaluated anew. Energy is an interpolation-based estimate.}',
        r'\label{tab:benchmark_results}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Sea & Controller & Position RMSE (m) & Yaw MAE ($^\circ$) & CPU (ms) & Energy (J) \\',r'\midrule']
    for sea in scenarios(cfg):
        for ctrl in ['PID','PPO','SMC','NMPC']:
            row=summary.loc[(sea,ctrl)];cells=[sea.capitalize(),ctrl]
            for key,digits in [('rmse_m',3),('yaw_mae_deg',1),('cpu_ms',4),('energy_J',0)]:
                cells.append(f'${row[(key,"mean")]:.{digits}f} \\pm {row[(key,"std")]:.{digits}f}$')
            lines.append(' & '.join(cells)+r' \\')
        lines.append(r'\midrule')
    lines[-1]=r'\bottomrule';lines.extend([r'\end{tabular}',r'\end{table*}'])
    (output/'benchmark_results.tex').write_text('\n'.join(lines)+'\n')


def article_bundle(root,plan):
    """Stage a reviewable candidate; do not overwrite article text/current assets."""
    paper=ROOT/'documents/oceans2026'
    sys.path.insert(0,str(paper))
    from evaluation_figures import write_evaluation_figures
    from training_seed_validation import write_seed_validation
    candidate=root/'article_candidate';candidate.mkdir(exist_ok=True)
    for path in (root/'combined_benchmark/figures').rglob('*'):
        if path.is_file() and path.suffix in {'.pdf','.png'}:
            dest=candidate/'figs/current'/path.relative_to(root/'combined_benchmark/figures')
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    benchmark_summary(root/'combined_benchmark',candidate/'tables')
    write_evaluation_figures(root/'combined_benchmark',candidate,root/'combined_star')
    write_seed_validation(root/'seed_robustness',candidate,root/'combined_benchmark')
    check_dir=candidate/'figs/current/checkpoint_comparison';check_dir.mkdir(parents=True,exist_ok=True)
    for path in (root/'checkpoint_comparison').glob('*'):
        if path.is_file():shutil.copy2(path,check_dir/path.name)
    fragment=r'''% Insert before the held-out-wave-seed assessment, after importing candidate figures.
\paragraph{Checkpoint comparison under a common evaluation model.}
Figure~\ref{fig:checkpoint_comparison} compares four saved policies: no-wave
preparation (A), preparation with waves (B), the final figure-eight curriculum,
and star-only fine-tuning. All are tested with the same target plant,
Moderate condition, trajectories, initial states, and wave seed 42. This is a
single-realization descriptive comparison; training exposure and training
mechanics differ across checkpoints, so it is not an isolated causal ablation.
Stopped episodes are identified by duration and excluded from full-horizon RMSE.
\begin{figure*}[pos=tbp]
\centering
\includegraphics[width=\textwidth]{figs/current/checkpoint_comparison/checkpoint_comparison.pdf}
\caption{Checkpoint comparison with a shared evaluation plant and paired wave
realization. Cells report full-horizon position RMSE and yaw MAE; color scales
are logarithmic. Stopped episodes are labeled with their observed duration.}
\label{fig:checkpoint_comparison}
\end{figure*}
'''
    if plan['spec'].get('finetuned_training_status') != 'completed':
        fragment = r'\textbf{Training ended by its configured validation-RMSE plateau criterion during FT1; FT2 was not reached.}'+'\n'+fragment
    (candidate/'checkpoint_comparison.tex').write_text(fragment)
    (candidate/'README.md').write_text('''# Fine-tuned article candidate\n\nThe original manuscript and its figures were not overwritten. This folder contains\nnew checkpoint-comparison, final-policy seed assessment, and controller-comparison\nfigures. Place the checkpoint-comparison fragment before the multi-seed diagnostic.\nUpdate the main text, checkpoint hashes, training/fine-tuning counts and numeric\nresults together when incorporating these figures. Star is now a trained\ngeometry; circle and square remain transfer tests. PID/SMC/NMPC CSVs were reused,\nwhile PPO was rerun. CPU measurements combine archival and new executions;\nconsult combined_benchmark/reuse_provenance.yaml before interpreting CPU rankings.\n\nDo not run the old article updater against a new checkpoint while retaining the\nold checkpoint's numeric statements. That updater is not a manuscript rewrite.\n''')
    if plan['spec'].get('finetuned_training_status') != 'completed':
        readme=candidate/'README.md'
        readme.write_text('**Training ended by the configured validation-RMSE plateau criterion during FT1; FT2 was not reached.**\n\n'+readme.read_text())
    write_yaml(candidate/'sources.yaml',dict(campaign=str(root),policies=plan['policies'],
        reuse_provenance=read_yaml(root/'combined_benchmark/reuse_provenance.yaml')))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'configs/evaluation/finetune_article_campaign.yaml')
    p.add_argument('--output',type=Path)
    p.add_argument('--phase',choices=['plan','checkpoints','controllers','seeds','plots','article','all'],default='plan')
    args=p.parse_args(argv)
    root,plan=prepare(args.config,args.output)
    if args.phase=='plan':
        print(f'Prepared {root}: 16 checkpoint episodes, 27 new PPO benchmark episodes, 3 new PPO star episodes, 80 final-policy seed episodes. Baselines reused.');return
    if args.phase in {'checkpoints','all'}:
        for name in LABELS:run_evaluation(root,f'checkpoints/{name}')
        checkpoint_plot(root,plan)
    if args.phase in {'controllers','all'}:
        for name in ['ppo_benchmark','ppo_star']:run_evaluation(root,name)
        merge_baselines(plan['spec']['baseline'],root/'ppo_benchmark',root/'combined_benchmark')
        merge_baselines(plan['spec']['baseline_star'],root/'ppo_star',root/'combined_star')
    if args.phase in {'seeds','all'}:run_evaluation(root,'seed_robustness')
    if args.phase in {'plots','all'}:plot_results(root,plan)
    if args.phase in {'article','all'}:article_bundle(root,plan)
    print(f'Outputs: {root}')


if __name__=='__main__':main()
