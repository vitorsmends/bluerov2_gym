"""Paired 4x article benchmark with PPO, NMPC, PID and SMC in W1--W3.

Reuse matching PPO/NMPC and W1 baseline records; evaluate missing PID/SMC cases.
Only the 3-D spatial article figures omit PID/SMC, for readability.
"""
from pathlib import Path
import copy,json,os,shutil,subprocess,sys,hashlib
import numpy as np
import pandas as pd
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bluerov2_gym.workflows.artifacts import create_run,update_manifest,write_yaml

def main():
    out=ROOT/'artifacts/evaluation/article_scale4_four_controllers';out.mkdir(parents=True,exist_ok=False)
    paired=ROOT/'artifacts/evaluation/article_scale4_ppo_nmpc/benchmark'
    baseline=ROOT/'artifacts/evaluation/trajectory_scale_controllers/scale_4'
    cfg=yaml.safe_load((paired/'config.resolved.yaml').read_text())
    other=yaml.safe_load((baseline/'config.resolved.yaml').read_text())
    for key in ['environment','experiment','trajectories']:assert cfg[key]==other[key]
    assert cfg['disturbance']['scenarios']==other['disturbance']['scenarios']
    for source in [paired,baseline]:assert yaml.safe_load((source/'manifest.yaml').read_text())['status']=='completed'
    cfg['controllers'] += [c for c in other['controllers'] if c['name'] in {'pid','smc'}]
    write_yaml(out/'evaluation.yaml',cfg)
    canonical=create_run('evaluation',cfg,cfg,out/'benchmark')
    (out/'inputs').mkdir();shutil.copy2(ROOT/'artifacts/training/progressive_wave_exposure/manifest.yaml',out/'inputs/training_status.yaml')
    worker=copy.deepcopy(cfg);worker['controllers']=[c for c in cfg['controllers'] if c['name'] in {'pid','smc'}]
    worker['disturbance']['scenario_names']=['walker_w2','walker_w3'];worker['disturbance']['default_scenario']='walker_w2'
    path=out/'pid_smc.yaml';write_yaml(path,worker)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    try:
        print('Evaluating missing PID/SMC cases in W2/W3; retaining all matching 4x records.',flush=True)
        with (out/'pid_smc.log').open('w') as log:
            subprocess.run([sys.executable,'-m','bluerov2_gym.workflows.cli','evaluate','--config',str(path),'--output',str(out/'workers/pid_smc')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        actual=yaml.safe_load((out/'workers/pid_smc/config.resolved.yaml').read_text())
        for key in ['environment','experiment','trajectories']:assert actual[key]==cfg[key]
        assert actual['disturbance']['scenarios']==cfg['disturbance']['scenarios']
        hashes={}
        for sea in cfg['disturbance']['scenario_names']:
            for case in cfg['trajectories']:
                previous=None
                for ctrl in ['ppo','nmpc','pid','smc']:
                    source=paired if ctrl in {'ppo','nmpc'} else (baseline if sea=='walker_w1' else out/'workers/pid_smc')
                    original=source/f"data/{sea}/{ctrl}_{case['name']}.csv";df=pd.read_csv(original)
                    assert len(df)==4000 and set(df.seed)=={42} and df.repetition_id.nunique()==1
                    if previous is not None:
                        cols=['time','seed','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','noise_std']+[c for c in df if c.endswith('_ref')]
                        np.testing.assert_allclose(previous[cols],df[cols],rtol=0,atol=1e-10)
                        states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                        np.testing.assert_allclose(previous[states].iloc[0],df[states].iloc[0],rtol=0,atol=1e-10)
                    previous=df;target=canonical/f'data/{sea}'/original.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,target)
                    hashes[str(original)]=hashlib.sha256(original.read_bytes()).hexdigest()
        (out/'provenance.json').write_text(json.dumps(dict(data_sha256=hashes,records=36,seed=42,spatial_scale=4,period_scale=4,duration_s=400,controllers=['PPO','NMPC','PID','SMC'],display_scope='All four controllers in metrics and discussions. Only 3-D trajectory figures omit PID/SMC for readability because their errors obscure the main trajectories. No controller is excluded from the benchmark.',selection='Reference scale chosen retrospectively after W1 1x/2x/4x exploration. Matching source records reused; missing PID/SMC cases newly simulated. Training unchanged.'),indent=2))
        update_manifest(canonical,status='completed',expected_runs=36)
    except BaseException as error:
        update_manifest(canonical,status='failed',error=str(error));raise
    print('Validated 36 paired records:',canonical,flush=True)
if __name__=='__main__':main()
