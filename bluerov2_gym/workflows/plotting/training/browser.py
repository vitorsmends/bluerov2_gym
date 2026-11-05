"""Serve a live, local training dashboard using existing TensorBoard logs."""
import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
from pathlib import Path
import time
import webbrowser

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from ...configuration import read_yaml
from .status import training_phase


TAGS = {
    'reward': 'rollout/ep_rew_mean',
    'reward_step': 'rollout/reward_mean_per_step',
    'error': 'rollout/position_error_mean_m',
    'rmse': 'rollout/position_error_rmse_m',
}


class TrainingData:
    def __init__(self, run, interval=5.):
        self.run = Path(run).expanduser().resolve()
        self.interval = interval
        self.accumulator = None
        self.cached = None
        self.last_read = float('-inf')

    def read(self):
        if self.cached is not None and time.monotonic() - self.last_read < self.interval:
            return self.cached
        if self.accumulator is None:
            events = list((self.run / 'logs/tensorboard').rglob('events.out.tfevents.*'))
            if events:
                folder = max(events, key=lambda p: p.stat().st_mtime).parent
                self.accumulator = EventAccumulator(str(folder), size_guidance={'scalars': 0})
        series = {}
        if self.accumulator is not None:
            self.accumulator.Reload()
            available = self.accumulator.Tags().get('scalars', [])
            for name, tag in TAGS.items():
                # A resumed log can contain duplicate steps; retain the latest.
                values = {e.step: float(e.value) for e in self.accumulator.Scalars(tag)
                          if math.isfinite(e.value)} if tag in available else {}
                series[name] = [[s, values[s]] for s in sorted(values)]
        latest = max((points[-1][0] for points in series.values() if points), default=0)
        path = self.run / 'config.resolved.yaml'
        config = read_yaml(path) if path.is_file() else {}
        phase_path = self.run / 'models/curriculum_status.yaml'
        runtime = read_yaml(phase_path) if phase_path.is_file() else None
        phase, sea, boundaries = training_phase(config, latest, runtime)
        manifest = self.run / 'manifest.yaml'
        status = read_yaml(manifest).get('status', 'unknown') if manifest.is_file() else 'unknown'
        self.cached = dict(run=self.run.name, step=latest, phase=phase, sea=sea,
                           boundaries=boundaries, series=series, status=status,
                           refresh_seconds=self.interval,
                           rollout_steps=config.get('ppo', {}).get('n_steps'),
                           read_at=time.time())
        self.last_read = time.monotonic()
        return self.cached


def make_server(run, port=8765, interval=5.):
    data = TrainingData(run, interval)
    page = Path(__file__).with_name('browser.html').read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = self.path.split('?', 1)[0]
            if route == '/':
                body, content_type, code = page, 'text/html; charset=utf-8', 200
            elif route == '/api/training':
                try:
                    payload, code = data.read(), 200
                except (OSError, ValueError) as error:
                    payload, code = {'error': str(error)}, 503
                body = json.dumps(payload, allow_nan=False).encode()
                content_type = 'application/json; charset=utf-8'
            else:
                self.send_error(404)
                return
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    return HTTPServer(('127.0.0.1', port), Handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--interval', type=float, default=5.)
    parser.add_argument('--no-open', action='store_true', help='Print the URL without opening a tab')
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535 or not math.isfinite(args.interval) or args.interval <= 0:
        parser.error('Use a valid port and a positive finite refresh interval.')
    try:
        server = make_server(args.run, args.port, args.interval)
    except OSError as error:
        parser.error(f'{error}. Try a different --port.')
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Training dashboard: {url}\nReading: {args.run.resolve()}\nCtrl+C closes the viewer; training continues.', flush=True)
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
