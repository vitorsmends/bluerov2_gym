import unittest
from copy import deepcopy
from bluerov2_gym.workflows.training.curriculum_progression import early_skip_reason

class EarlySkippingTests(unittest.TestCase):
    settings=dict(enabled=True,first_stage=5,min_evaluations=15,patience_evaluations=10)
    def state(self):
        return dict(index=9,evaluations=70,successful_batches=0,last_evaluation={'passed':False},
                    progress_monitor=dict(active=True,stale=14,current={'rmse':.1}))
    def test_protects_all_four_preparation_and_transfer_phases(self):
        for stage in range(1,5):
            s=self.state();s['index']=stage
            self.assertIsNone(early_skip_reason(s,self.settings))
    def test_requires_exposure_plateau_and_finite_errors(self):
        s=self.state();self.assertIsNotNone(early_skip_reason(s,self.settings))
        for key,val in [('evaluations',14),('successful_batches',1)]:
            c=deepcopy(s);c[key]=val;self.assertIsNone(early_skip_reason(c,self.settings))
        for key,val in [('active',False),('stale',9),('current',{'rmse':float('nan')})]:
            c=deepcopy(s);c['progress_monitor'][key]=val;self.assertIsNone(early_skip_reason(c,self.settings))
        s['last_evaluation']['passed']=True;self.assertIsNone(early_skip_reason(s,self.settings))
    def test_disabled_does_not_change_article_curriculum(self):
        self.assertIsNone(early_skip_reason(self.state(),{}))

if __name__=='__main__':unittest.main()
