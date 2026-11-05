"""Workflow contracts: selection, provenance and data-only plotting."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from bluerov2_gym.workflows.configuration import resolve, evaluation_plan
from bluerov2_gym.workflows.artifacts import create_run
from bluerov2_gym.workflows.plotting import PlotManager

ROOT=Path(__file__).resolve().parents[1]


class WorkflowTests(unittest.TestCase):
    def test_config_relative_paths_independent_of_cwd(self):
        _,config=resolve(ROOT/'configs/evaluation/stationkeeping_single.yaml')
        self.assertEqual(Path(config['model']),ROOT/'artifacts/models/current/bluerov_ppo.zip')
        self.assertEqual(Path(config['dynamics']),ROOT/'configs/dynamics/dynamics_config.yaml')
        _,config=resolve(ROOT/'configs/evaluation/article.yaml')
        self.assertNotIn('config_path',config['disturbance'])
        self.assertIn('moderate',config['disturbance']['scenarios'])
        self.assertEqual(evaluation_plan(config)['runs'],200)

    def test_evaluation_inherits_training_snapshot_and_rejects_dynamics_changes(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            training=root/'train'
            training.mkdir()
            snapshot={'environment': {'config': {'dt': .2}, 'dynamics_config': {
                'dt': .2, 'mechanics_model': 'corrected',
                'disturbance_model': {'preset': 'std'}}}}
            (training/'config.resolved.yaml').write_text(yaml.safe_dump(snapshot))
            path=root/'evaluate.yaml'
            path.write_text(yaml.safe_dump({'environment': {'training_run': 'train'}}))
            _,config=resolve(path)
            self.assertEqual(config['environment']['dynamics_config'],snapshot['environment']['dynamics_config'])
            self.assertEqual(config['environment']['config'],snapshot['environment']['config'])
            path.write_text(yaml.safe_dump({'environment': {'training_run': 'train',
                'dynamics_config': {'mechanics_model': 'std'}}}))
            with self.assertRaisesRegex(ValueError,'differ from training_run'):
                resolve(path)

    def test_runner_applies_inherited_environment_limits(self):
        from bluerov2_gym.workflows.evaluation.runner import ExperimentRunner
        with tempfile.TemporaryDirectory() as folder:
            runner=ExperimentRunner({'experiment':{},'environment':{
                'config': {'dt': .2,'action_space': {'low': -12.,'high': 12.}},
                'dynamics_config': {'mechanics_model':'corrected'}}})
            env=runner._make_env(Path(folder))
            try:
                self.assertEqual(env.unwrapped.dynamics.dt,.2)
                self.assertEqual(float(env.action_space.high[0]),12.)
            finally:
                env.close()

    def test_default_run_root_is_centralized(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'artifacts'
            with patch('bluerov2_gym.workflows.paths.ARTIFACTS_ROOT',root):
                run=create_run('evaluation',{}, {})
            self.assertEqual(run.parent,(root/'evaluation').resolve())

    def test_plot_defaults_are_inside_artifacts(self):
        from bluerov2_gym.workflows.paths import ARTIFACTS_ROOT
        from bluerov2_gym.workflows.plotting.article.config import OUTPUT_DIR, TABLE_DIR
        self.assertTrue(OUTPUT_DIR.is_relative_to(ARTIFACTS_ROOT))
        self.assertTrue(TABLE_DIR.is_relative_to(ARTIFACTS_ROOT))
        self.assertTrue(PlotManager().output.is_relative_to(ARTIFACTS_ROOT))

    def test_agent_assets_exist_at_relocated_paths(self):
        _,config=resolve(ROOT/'configs/evaluation/stationkeeping_single.yaml')
        self.assertTrue(Path(config['model']).is_file())
        self.assertTrue(Path(config['normalizer']).is_file())
        for old in ['models','results','training_runs','bluerov_tensorboard','plots_training']:
            self.assertFalse((ROOT/old).exists())

    def test_plan_counts_enabled_controllers_and_cases(self):
        config={'experiment':{'mode':'trajectory_suite','repetitions':3},
                'controllers':[{'name':'pid'},{'name':'ppo','enabled':False}],
                'trajectories':[{'name':'circle'},{'name':'square','enabled':False}],
                'disturbance':{'scenario_names':['calm','moderate']}}
        self.assertEqual(evaluation_plan(config)['runs'],6)
        self.assertEqual(evaluation_plan(config)['controllers'],[{'name':'pid'}])

    def test_stationkeeping_plan_matches_existing_all_scenarios_semantics(self):
        config={'experiment':{'mode':'stationkeeping','repetitions':2},
                'controllers':[{'name':'pid'}],
                'stationkeeping':{'targets':[{'id':1},{'id':2}]},
                'disturbance':{'default_scenario':'moderate','scenarios':{'moderate':{},'severe':{}}}}
        self.assertEqual(evaluation_plan(config)['runs'],8)

    def test_run_creation_never_overwrites_and_keeps_requested_input(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'run'
            requested={'model':'relative.zip'}
            run=create_run('evaluation',requested,{'model':'absolute.zip'},path)
            self.assertEqual(yaml.safe_load((run/'config.requested.yaml').read_text()),requested)
            self.assertTrue((run/'logs').is_dir())
            for name in ['data','figures','tables']:self.assertFalse((run/name).exists())
            self.assertIn('git_commit',yaml.safe_load((run/'manifest.yaml').read_text()))
            with self.assertRaises(FileExistsError):create_run('evaluation',{}, {},path)

    def test_old_entrypoint_directories_are_removed(self):
        for directory in ['training', 'experiments', 'plots', 'config']:
            self.assertFalse((ROOT/directory).exists())

    def test_plot_import_does_not_load_controllers_or_create_files(self):
        with tempfile.TemporaryDirectory() as folder:
            code='''import sys
from bluerov2_gym.workflows.plotting import PlotManager
import bluerov2_gym.workflows.plotting.article.plot_energy
import bluerov2_gym.workflows.plotting.stationkeeping.stationkeeping
assert 'torch' not in sys.modules
assert 'casadi' not in sys.modules
'''
            subprocess.run([sys.executable,'-c',code],cwd=folder,check=True)
            self.assertEqual(list(Path(folder).iterdir()),[])

    def test_pid_factory_does_not_import_optional_nmpc(self):
        code='''import sys
from bluerov2_gym.workflows.evaluation.controllers.factory import ControllerFactory
from bluerov2_gym.envs.core.dynamics import Dynamics
ControllerFactory.create('pid',dynamics=Dynamics(),reference_provider=None,dt=.1)
assert 'casadi' not in sys.modules
'''
        subprocess.run([sys.executable,'-c',code],check=True)

    def test_plot_preflight_rejects_missing_data_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            manager=PlotManager(config={'input_dir':str(path/'missing'),'output_dir':str(path/'out')})
            with self.assertRaises(ValueError):manager.generate_suite('stationkeeping')
            self.assertFalse((path/'out').exists())

    def test_article_metrics_keep_repetition_and_legacy_grouping(self):
        from bluerov2_gym.workflows.plotting.article import plot_rmse
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'pid_circle.csv').write_text('time,tracking_error_m,repetition_id\n0,3,0\n1,4,0\n0,0,1\n1,0,1\n')
            (path/'pid_square.csv').write_text('time,tracking_error_m\n0,2\n1,2\n')
            plot_rmse.configure({'scenarios':{'Moderate':str(path)},'output_dir':str(path/'figs'),
                                 'table_dir':str(path/'tables'),'controllers':['pid']})
            table=plot_rmse.compute_rmse_table()
            self.assertEqual(len(table),3)
            values=sorted(table['rmse_error_m'].tolist())
            self.assertEqual(values[:2],[0.,2.])
            self.assertAlmostEqual(values[2],(12.5)**.5)

    def test_training_run_selects_one_log_without_claiming_multi_seed_statistics(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/'config.resolved.yaml').write_text('training: {seed: 42}\n')
            log=run/'logs/tensorboard/PPO_seed_42_1'
            log.mkdir(parents=True)
            (log/'events.out.tfevents.fixture').touch()
            settings=PlotManager(run).settings('training')
            self.assertEqual(settings['run'],'PPO_seed_42_1')

    def test_spatial_scenario_selection_keeps_inputs_and_outputs_separate(self):
        with tempfile.TemporaryDirectory() as folder:
            run=Path(folder)
            (run/'config.resolved.yaml').write_text('disturbance: {scenario_names: [moderate, severe]}\n')
            for name in ['moderate','severe']:
                (run/'data'/name).mkdir(parents=True)
            settings=PlotManager(run,{'scenario':'severe'}).settings('trajectory_spatial')
            self.assertEqual(Path(settings['input_dir']),(run/'data/severe').resolve())
            self.assertEqual(Path(settings['output_dir']),(run/'figures/trajectory_spatial/severe').resolve())
            self.assertEqual(Path(settings['representative_output_dir']),(run/'figures/trajectory_spatial/severe/representative').resolve())

    def test_cli_help_from_another_directory(self):
        result=subprocess.run([sys.executable,'-m','bluerov2_gym.workflows','train','--help'],
                               cwd='/tmp',capture_output=True,text=True,check=True)
        self.assertIn('--seeds',result.stdout)


if __name__=='__main__':unittest.main()
