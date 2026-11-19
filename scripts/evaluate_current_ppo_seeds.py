"""Focused fixed-policy wave-seed diagnostic for the current article checkpoint.

Twenty unseen disturbance seeds, W2 only, three 4x paths, 400 s each.
No extra reference-controller evaluations or optimization updates.
"""
from pathlib import Path
import hashlib,json,os,re,subprocess,sys
import numpy as np,pandas as pd,yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def plot_seed_metrics(metrics,out):
    # Fixed offsets are deterministic display positions, not confidence limits.
    cases=list(metrics.trajectory.unique());titles={'figure_eight_depth':'Figure eight','square':'Square','circle':'Circle'}
    with plt.rc_context({'font.family':'serif','font.size':12,'pdf.fonttype':42,'mathtext.fontset':'cm'}):
        fig,axes=plt.subplots(1,3,figsize=(10,3.5),layout='constrained')
        for ax,col,label in zip(axes,['tracking_time_percent','rmse_m','yaw_mae_deg'],[r'$P_{\epsilon}^{\mathrm{track}}$ [%]',r'$\mathrm{RMSE}$ [m]',r'$\mathrm{MAE}_{\psi}$ [deg]']):
            for i,case in enumerate(cases):
                v=metrics[metrics.trajectory==case][col].to_numpy();offset=np.linspace(-.15,.15,len(v))
                ax.scatter(i+offset,v,color='#59A14F',s=18,alpha=.70,edgecolors='none')
                ax.errorbar(i,v.mean(),yerr=v.std(ddof=0),fmt='D',color='.15',capsize=4,ms=4,lw=1.1,zorder=5)
            ax.set_xticks(range(3),[titles[c] for c in cases]);ax.set_ylabel(label);ax.grid(axis='y',color='.86',ls='--',lw=.65);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
            ax.tick_params(labelsize=10)
            if col=='tracking_time_percent':ax.set_ylim(0,100)
        for ext in ['pdf','png']:fig.savefig(out/f'wave_seed_sensitivity.{ext}',dpi=250)
        plt.close(fig)


def main():
    out=ROOT/'artifacts/evaluation/article_current_ppo_wave_seeds'
    if '--plot-only' in sys.argv:
        plot_seed_metrics(pd.read_csv(out/'per_seed_metrics.csv'),out)
        return
    out.mkdir(parents=True,exist_ok=False)
    benchmark=ROOT/'artifacts/evaluation/article_scale4_four_controllers/benchmark'
    cfg=yaml.safe_load((benchmark/'config.resolved.yaml').read_text())
    cfg['controllers']=[c for c in cfg['controllers'] if c['name']=='ppo']
    known=set()
    for path in (ROOT/'artifacts/training/progressive_wave_exposure/logs/curriculum_evaluations').glob('*.yaml'):
        known.update(map(int,re.findall(r'^\s*seed:\s*(\d+)\s*$',path.read_text(),re.M)))
    seed=3_000_000_000
    while known.intersection(range(seed,seed+20)):seed+=20
    cfg['experiment'].update(repetitions=20,seed=seed)
    cfg['disturbance']['scenario_names']=['walker_w2'];cfg['disturbance']['default_scenario']='walker_w2'
    cfg['disturbance']['scenarios']['walker_w2']['seed']=seed
    path=out/'evaluation.yaml';path.write_text(yaml.safe_dump(cfg,sort_keys=False))
    models={key:Path(cfg['controllers'][0]['params'][key]) for key in ['model_path','vecnormalize_path']}
    hashes={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in models.items()}
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    print(f'Started 60 PPO-only records: W2, 3 paths, seeds {seed}--{seed+19}.',flush=True)
    with (out/'evaluation.log').open('w') as log:
        subprocess.run([sys.executable,'-m','bluerov2_gym.workflows.cli','evaluate','--config',str(path),'--output',str(out/'benchmark')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[];data_hashes={}
    for case in cfg['trajectories']:
        source=out/f"benchmark/data/walker_w2/ppo_{case['name']}.csv";df=pd.read_csv(source)
        assert set(df.seed.astype(int))==set(range(seed,seed+20))
        data_hashes[str(source)]=hashlib.sha256(source.read_bytes()).hexdigest()
        for rep,g in df.groupby('repetition_id'):
            assert len(g)==4000 and g.seed.nunique()==1
            np.testing.assert_allclose(g.time,np.arange(4000)*.1,rtol=0,atol=1e-10)
            rows.append(dict(trajectory=case['name'],seed=int(g.seed.iloc[0]),rmse_m=np.sqrt(np.mean(g.tracking_error_m**2)),yaw_mae_deg=np.degrees(np.abs(np.arctan2(np.sin(g.yaw_error),np.cos(g.yaw_error)))).mean(),tracking_time_percent=100*np.mean(g.tracking_error_m<=.2)))
    metrics=pd.DataFrame(rows);metrics.to_csv(out/'per_seed_metrics.csv',index=False)
    summary=metrics.groupby('trajectory')[['rmse_m','yaw_mae_deg','tracking_time_percent']].agg(['mean',lambda v:v.std(ddof=0),'min','max']).rename(columns={'<lambda_0>':'std'})
    summary.to_csv(out/'seed_summary.csv')
    plot_seed_metrics(metrics,out)
    assert hashes=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in models.items()}
    provenance=dict(model_sha256=hashes,source_benchmark=str(benchmark),sea='W2',seeds=list(range(seed,seed+20)),seed_overlap_with_archived_assessments=False,checked_assessment_seed_count=len(known),records=60,steps_per_episode=4000,training_seed=42,scope='Fixed final source PPO and frozen normalizer; wave realizations vary. Three 4x paths, W2 only. Full 400 s metrics; population SD across 20 environmental seeds, not across independent training runs. Records excluded from controller-comparison metrics.',data_sha256=data_hashes)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(summary.to_string());print('Completed:',out,flush=True)
if __name__=='__main__':main()
