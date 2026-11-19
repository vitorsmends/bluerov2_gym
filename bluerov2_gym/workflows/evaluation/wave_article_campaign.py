"""Paired four-controller/three-geometry W1--W3 article campaign.

One paired realization per case by default. Cached NMPC figure-eight records
are reused only after exact physical-contract checks. Other cases are new.
"""
from pathlib import Path
from copy import deepcopy
import argparse,hashlib,json,os,shutil,subprocess,sys
import numpy as np
import pandas as pd
from ..configuration import read_yaml,evaluation_plan
from ..artifacts import create_run,update_manifest,write_yaml

ROOT=Path(__file__).resolve().parents[3]
SEAS=['walker_w1','walker_w2','walker_w3']
CASES=['figure_eight_depth','square','circle']
CONTROLLERS=['ppo','nmpc','pid','smc']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compatible(master,worker,sea,case):
    assert master['environment']==worker['environment'], 'Different physical environment'
    a,b=[{k:v for k,v in c['experiment'].items() if k!='repetitions'} for c in [master,worker]]
    assert a==b,'Different initial state or protocol'
    assert master['disturbance']['scenarios'][sea]==worker['disturbance']['scenarios'][sea]
    cases=lambda c:{t['name']:t.get('params',{}) for t in c['trajectories'] if t.get('enabled',True)}
    assert cases(master)[case]==cases(worker)[case], 'Different reference'


def run(training,output):
    training=Path(training).resolve();output=Path(output).resolve()
    cfg=read_yaml(training/'config.resolved.yaml');state=read_yaml(training/'models/curriculum_status.yaml')
    if state['status'] not in {'completed','exposure_completed'}:
        raise ValueError('Select a terminal exposure/completed model; no live checkpoint guessing.')
    output.mkdir(parents=True,exist_ok=False);inputs=output/'inputs';inputs.mkdir()
    pair={}
    for key,p in [('model_path',training/'models'/(cfg['paths']['model_name']+'.zip')),
                  ('vecnormalize_path',training/'models'/cfg['paths']['vecnormalize_name'])]:
        shutil.copy2(p,inputs/p.name);pair[key]=str(inputs/p.name)
    shutil.copy2(training/'config.resolved.yaml',inputs/'training_config.yaml')
    shutil.copy2(training/'models/curriculum_status.yaml',inputs/'training_status.yaml')
    master=read_yaml(ROOT/'configs/evaluation/walker_eight/baseline.yaml')
    master['experiment']['repetitions']=1
    ppo=dict(observation_adapter='legacy_yaw_error',action_control=cfg['environment']['action_control'],
             env_config_path=str(ROOT/'configs/dynamics/blue_rov_config.yaml'),
             dynamics_config_path=str(ROOT/'configs/dynamics/dynamics_config.yaml'),**pair)
    master['controllers']=[dict(name='ppo',enabled=True,params=ppo),
                           dict(name='nmpc',enabled=True,params=dict(horizon=10,control_blocks=5)),
                           dict(name='pid',enabled=True,params={}),dict(name='smc',enabled=True,params={})]
    master['trajectories']=[master['trajectories'][0],
        dict(name='square',enabled=True,params=dict(side=1.6,depth=-5.,period=80.)),
        dict(name='circle',enabled=True,params=dict(radius=1.,depth=-5.,period=50.))]
    write_yaml(output/'evaluation.yaml',master)
    canonical=create_run('evaluation',master,master,output/'benchmark')
    write_yaml(canonical/'plan.yaml',evaluation_plan(master))
    jobs=output/'jobs';jobs.mkdir();tasks=[];commands=[]
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')
    def launch(name,worker):
        path=jobs/(name+'.yaml');write_yaml(path,worker);dest=output/'workers'/name
        command=[sys.executable,'-m','bluerov2_gym.workflows.cli','evaluate','--config',str(path),'--output',str(dest)]
        log=(jobs/(name+'.log')).open('w')
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        tasks.append((name,child,log,dest));commands.append(command)
    cached=ROOT/'artifacts/evaluation/walker_eight/baseline'
    cached_cfg=read_yaml(cached/'config.resolved.yaml')
    if read_yaml(cached/'manifest.yaml')['status']!='completed':raise ValueError('Cached NMPC run is not complete')
    for sea in SEAS:
        compatible(master,cached_cfg,sea,'figure_eight_depth')
        if not (cached/f'data/{sea}/nmpc_figure_eight_depth.csv').is_file():
            raise FileNotFoundError(f'Missing cached NMPC eight: {sea}')
    fast=deepcopy(master);fast['controllers']=[x for x in fast['controllers'] if x['name']!='nmpc'];launch('ppo_pid_smc',fast)
    for sea in SEAS:
        worker=deepcopy(master);worker['controllers']=[x for x in worker['controllers'] if x['name']=='nmpc']
        worker['disturbance']['scenario_names']=[sea];worker['disturbance']['default_scenario']=sea
        worker['trajectories']=[x for x in worker['trajectories'] if x['name']!='figure_eight_depth'];launch('nmpc_'+sea,worker)
    provenance=dict(training_run=str(training),training_status=state['status'],checkpoint_steps=state['step'],
        unmastered_stages=state.get('unmastered_stages',[]),inputs_sha256={p.name:sha(p) for p in inputs.iterdir()},
        commands=commands,repetitions=1,seed=42,
        note='Single paired realization per condition, not independent training seeds. Concurrent workers; CPU times are execution-dependent. Cached NMPC eight timing is from an earlier execution.')
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print('Started four workers; 36 paired records on eight, square and circle.',flush=True)
    try:
        failed=[]
        for name,child,log,dest in tasks:
            code=child.wait();log.close();print(name,'finished',code,flush=True)
            if code:failed.append(name)
        if failed:raise RuntimeError('Worker failure: '+str(failed))
        hashes={}
        for _,_,_,dest in tasks:
            worker=read_yaml(dest/'config.resolved.yaml')
            assert read_yaml(dest/'manifest.yaml')['status']=='completed'
            for sea in worker['disturbance']['scenario_names']:
                for case in worker['trajectories']:
                    compatible(master,worker,sea,case['name'])
                    for controller in worker['controllers']:
                        path=dest/f"data/{sea}/{controller['name']}_{case['name']}.csv"
                        target=canonical/f'data/{sea}'/path.name;target.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copy2(path,target);hashes[str(path)]=sha(path)
        for sea in SEAS:
            path=cached/f'data/{sea}/nmpc_figure_eight_depth.csv';df=pd.read_csv(path)
            df=df.loc[df.repetition_id==0].copy();df.to_csv(canonical/f'data/{sea}/nmpc_figure_eight_depth.csv',index=False);hashes[str(path)]=sha(path)
        for sea in SEAS:
            for case in CASES:
                previous=None
                for ctrl in CONTROLLERS:
                    path=canonical/f'data/{sea}/{ctrl}_{case}.csv';df=pd.read_csv(path)
                    assert len(df)==1000 and df.repetition_id.nunique()==1
                    np.testing.assert_allclose(df.time,np.arange(1000)*.1,rtol=0,atol=1e-10)
                    assert set(df.seed)=={42}
                    if previous is not None:
                        columns=['seed','time','Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std']+[k for k in df if k.endswith('_ref')]
                        np.testing.assert_allclose(previous[columns],df[columns],rtol=0,atol=1e-10)
                        states=['x','y','z','roll','pitch','yaw','u','v','w','p','q','r']
                        np.testing.assert_allclose(previous[states].iloc[0],df[states].iloc[0],rtol=0,atol=1e-10)
                    previous=df
        assert len(list((canonical/'data').rglob('*.csv')))==len(SEAS)*len(CASES)*len(CONTROLLERS)
        provenance['data_sources_sha256']=hashes
        (output/'provenance.json').write_text(json.dumps(provenance,indent=2))
        update_manifest(canonical,status='completed',assembly='Validated controller workers and paired cached NMPC eight records',expected_runs=36)
    except BaseException as error:
        for _,child,_,_ in tasks:
            if child.poll() is None:child.terminate()
        update_manifest(canonical,status='failed',error=str(error));raise
    print('All 36 records validated:',canonical,flush=True)
    return canonical


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.training_run,args.output)
