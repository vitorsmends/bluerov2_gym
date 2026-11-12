"""Small synthetic checks for reuse integrity; no RL training or simulations."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bluerov2_gym.workflows.artifacts import write_yaml
from bluerov2_gym.workflows.evaluation.finetune_campaign import (
    merge_baselines, checkpoint_plot, LABELS, sha,
)


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.cfg=dict(experiment=dict(mode='trajectory_suite',dt=.1,steps=2,repetitions=1,seed=42),
            environment=dict(config={'dt':.1},dynamics_config={'mechanics_model':'corrected'}),
            controllers=[dict(name=n,params={}) for n in ['pid','ppo','smc','nmpc']],
            trajectories=[dict(name='square',params={})],
            disturbance=dict(default_scenario='moderate',scenario_names=['moderate'],
                             scenarios={'moderate':{'Hs':2.5,'seed':42}}))
        self.old=self.root/'old';self.new=self.root/'new'
        for run in [self.old,self.new]:
            (run/'data/moderate').mkdir(parents=True)
            write_yaml(run/'manifest.yaml',{'status':'completed'})
        write_yaml(self.old/'config.resolved.yaml',self.cfg)
        cfg=deepcopy(self.cfg);cfg['controllers']=[dict(name='ppo',params={})]
        write_yaml(self.new/'config.resolved.yaml',cfg)
        for name in ['pid','ppo','smc','nmpc']:
            self.frame(name,.1).to_csv(self.old/f'data/moderate/{name}_square.csv',index=False)
        self.frame('ppo',.05).to_csv(self.new/'data/moderate/ppo_square.csv',index=False)

    def frame(self,name,error):
        frame=pd.DataFrame({'repetition_id':[0,0],'seed':[42,42],'time':[0.,.1],
                           'trajectory':['square']*2,'controller':[name]*2,
                           'tracking_error_m':[error]*2,'yaw_error':[.1]*2})
        for key in ['Hs','Tp','gamma','wave_dir_x','wave_dir_y','scale','max_current','noise_std',
                    'x','y','z','roll','pitch','yaw','u','v','w','p','q','r',
                    'x_ref','y_ref','z_ref','yaw_ref']:
            frame[key]=0.
        return frame

    def test_replaces_only_ppo_and_detects_tampering(self):
        output=self.root/'combined'
        merge_baselines(self.old,self.new,output)
        for name in ['pid','smc','nmpc']:
            self.assertEqual(sha(self.old/f'data/moderate/{name}_square.csv'),sha(output/f'data/moderate/{name}_square.csv'))
        ppo=output/'data/moderate/ppo_square.csv'
        np.testing.assert_allclose(pd.read_csv(ppo).tracking_error_m,.05)
        merge_baselines(self.old,self.new,output)  # idempotent and verified
        ppo.write_text('tampered')
        with self.assertRaises(ValueError):merge_baselines(self.old,self.new,output)

    def test_refuses_incompatible_sea(self):
        cfg=deepcopy(self.cfg);cfg['disturbance']['scenarios']['moderate']['Hs']=4.
        write_yaml(self.new/'config.resolved.yaml',cfg)
        with self.assertRaises(ValueError):merge_baselines(self.old,self.new,self.root/'combined')

    def test_refuses_different_initial_state(self):
        path=self.old/'data/moderate/pid_square.csv';df=pd.read_csv(path);df.loc[0,'x']=1.;df.to_csv(path,index=False)
        with self.assertRaises(AssertionError):merge_baselines(self.old,self.new,self.root/'combined')

    def test_stopped_policy_is_not_assigned_full_horizon_rmse(self):
        for index,name in enumerate(LABELS):
            run=self.root/'checkpoints'/name;(run/'data/moderate').mkdir(parents=True);(run/'logs').mkdir()
            write_yaml(run/'manifest.yaml',{'status':'completed'})
            write_yaml(run/'config.resolved.yaml',self.cfg)
            df=self.frame('ppo',.1)
            if index==0:df=df.iloc[:1]
            df.to_csv(run/'data/moderate/ppo_square.csv',index=False)
            write_yaml(run/'logs/episodes.yaml',[dict(trajectory='square',terminated=index==0,truncated=False)])
        plan=dict(spec={'checkpoint_comparison':{'scenario':'moderate','trajectories':['square'],'seed':42}},policies={})
        checkpoint_plot(self.root,plan)
        df=pd.read_csv(self.root/'checkpoint_comparison/metrics.csv')
        self.assertTrue(pd.isna(df.iloc[0].rmse_m))
        self.assertFalse(df.iloc[0].completed)
        self.assertTrue((self.root/'checkpoint_comparison/checkpoint_comparison.pdf').exists())


if __name__=='__main__':unittest.main()
