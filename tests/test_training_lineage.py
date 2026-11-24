"""Keep only the ancestry that produced the final checkpoint."""
from pathlib import Path
import json,pickle,tempfile,unittest
import yaml
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from bluerov2_gym.workflows.plotting.training.lineage import build


class TrainingLineageTests(unittest.TestCase):
    def test_discards_parent_future_and_child_boundary_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);parent=root/'parent';child=root/'child'
            for run,end,points in [(parent,15,[(5,1.),(10,2.),(15,99.)]),
                                   (child,20,[(10,100.),(15,3.),(20,4.)])]:
                (run/'models').mkdir(parents=True)
                cfg=dict(environment={},ppo={},vecnormalize={},training={'seed':42})
                if run==child:cfg['training'].update(resume_run=str(parent),resume_checkpoint='agent_stage_1.zip')
                (run/'config.resolved.yaml').write_text(yaml.safe_dump(cfg))
                (run/'manifest.yaml').write_text(yaml.safe_dump({'status':'completed'}))
                (run/'models/curriculum_status.yaml').write_text(yaml.safe_dump({'step':end}))
                path=run/'logs/tensorboard/run';writer=EventFileWriter(str(path))
                for step,value in points:
                    writer.add_event(Event(step=step,wall_time=100.+step,summary=Summary(value=[
                        Summary.Value(tag='rollout/position_error_mean_m',simple_value=value)])))
                writer.close()
            with (parent/'models/agent_stage_1_curriculum_state.pkl').open('wb') as f:pickle.dump({'status':{'step':10}},f)
            result=build(child,root/'view')
            import csv
            with (root/'view/retained_scalars.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual([(int(r['step']),float(r['value'])) for r in rows],[(5,1.),(10,2.),(15,3.),(20,4.)])
            self.assertEqual(result['last_position_step'],20)
            self.assertEqual(result['segments'][0]['end_inclusive'],10)
            self.assertTrue((parent/'logs/tensorboard/run').is_dir())
            with self.assertRaises(FileExistsError):build(child,root/'view')


if __name__=='__main__':unittest.main()
