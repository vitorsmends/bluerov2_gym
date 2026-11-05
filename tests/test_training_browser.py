"""Live monitoring must read growing logs and display the recorded sea phase."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from urllib.request import urlopen

import yaml
from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

from bluerov2_gym.workflows.plotting.training.browser import make_server
from bluerov2_gym.workflows.plotting.training.status import training_phase


class TrainingBrowserTests(unittest.TestCase):
    def test_http_dashboard_reads_new_training_events(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run/'config.resolved.yaml').write_text(yaml.safe_dump({
                'training': {'task': 'stationkeeping'}, 'ppo': {'n_steps': 2048},
                'environment': {'dynamics_config': {'jonswap': {'Hs': 2., 'Tp': 12.}}}}))
            server = make_server(run, port=0, interval=0)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            writer = None
            try:
                base = f'http://127.0.0.1:{server.server_port}'
                with urlopen(base) as response:
                    self.assertIn('Treinamento PPO ao vivo', response.read().decode())
                def fetch():
                    with urlopen(base+'/api/training') as response:
                        return json.load(response)
                self.assertEqual(fetch()['step'], 0)
                writer = EventFileWriter(str(run/'logs/tensorboard/PPO_1'))
                for step, value in [(2048, 2.), (4096, .5)]:
                    writer.add_event(Event(step=step, summary=Summary(value=[
                        Summary.Value(tag='rollout/position_error_mean_m', simple_value=value)])))
                    writer.flush()
                    data = fetch()
                    self.assertEqual(data['step'], step)
                    self.assertEqual(data['series']['error'][-1], [step, value])
                    self.assertIn('sem currículo', data['phase'])
                    self.assertIn('Hs=2 m', data['sea'])
            finally:
                if writer:
                    writer.close()
                server.shutdown()
                server.server_close()
                thread.join()

    def test_curriculum_transition_accounts_for_rounded_rollouts(self):
        config = {'ppo': {'n_steps': 2048}, 'curriculum': [
            {'name': 'Calm', 'timesteps': 3000, 'jonswap': {'Hs': 1.}},
            {'name': 'Storm', 'timesteps': 3000, 'jonswap': {'Hs': 4.5}}]}
        phase, sea, boundaries = training_phase(config, 4096)
        self.assertIn('Calm', phase)
        self.assertEqual(boundaries, [4096])
        phase, sea, _ = training_phase(config, 6144)
        self.assertIn('Storm', phase)
        self.assertIn('Hs=4.5 m', sea)


if __name__ == '__main__':
    unittest.main()
