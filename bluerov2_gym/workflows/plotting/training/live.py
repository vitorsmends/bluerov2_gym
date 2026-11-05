"""Read a running training's TensorBoard events and update a small plot."""
import argparse
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from .curves import load_scalars, moving_average
from ...configuration import read_yaml
from .status import training_phase




class LiveTrainingPlot:
    def __init__(self, run, window=5):
        self.run = Path(run).expanduser().resolve()
        self.log_dir = self.run / 'logs/tensorboard'
        self.window = window
        self.event_dir = None
        self.config = {}
        self.fig, self.axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                           layout='constrained')

    def refresh(self):
        if (self.run / 'config.resolved.yaml').is_file():
            self.config = read_yaml(self.run / 'config.resolved.yaml')
        if self.event_dir is None:
            events = list(self.log_dir.rglob('events.out.tfevents.*'))
            if events:
                self.event_dir = max(events, key=lambda p: p.stat().st_mtime).parent
        metrics = load_scalars(self.event_dir) if self.event_dir else {}
        latest = max((int(s[-1]) for s, _ in metrics.values() if len(s)), default=0)
        phase_path = self.run / 'models/curriculum_status.yaml'
        runtime = read_yaml(phase_path) if phase_path.is_file() else None
        phase, sea, boundaries = training_phase(self.config, latest, runtime)
        specifications = [
            [('rollout/ep_rew_mean', 'Reward médio por episódio', '#59A14F')],
            [('rollout/position_error_mean_m', 'Erro médio', '#4E79A7'),
             ('rollout/position_error_rmse_m', 'RMSE', '#E15759')],
        ]
        for ax, series in zip(self.axes, specifications):
            ax.clear()
            ax.grid(alpha=.2)
            for boundary in boundaries:
                if boundary <= latest:
                    ax.axvline(boundary, color='gray', ls=':', alpha=.6)
            found = False
            for tag, label, color in series:
                if tag not in metrics or not len(metrics[tag][0]):
                    continue
                steps, values = metrics[tag]
                smoothed = moving_average(values, self.window)
                ax.plot(steps, values, color=color, alpha=.25)
                ax.plot(steps[-len(smoothed):], smoothed, color=color,
                        marker='.' if len(smoothed) == 1 else None,
                        label=f'{label}: {values[-1]:.3f}')
                found = True
            if found:
                ax.legend(loc='best')
            else:
                ax.text(.5, .5, 'Aguardando registros desta métrica no TensorBoard…',
                        ha='center', va='center', transform=ax.transAxes)
        self.axes[0].set_ylabel('Reward por episódio')
        self.axes[1].set_ylabel('Erro euclidiano (m)')
        self.axes[1].set_xlabel('Passos de interação com o ambiente')
        self.fig.suptitle(f'{self.run.name} — {latest:,} passos registrados\n'
                          f'{phase}\n{sea}\n'
                          f'Média móvel: {self.window} registros; atualização: {time.strftime("%H:%M:%S")}')
        self.fig.canvas.draw_idle()
        return metrics


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Training run directory')
    parser.add_argument('--interval', type=float, default=5., help='Refresh interval in seconds')
    parser.add_argument('--window', type=int, default=5, help='Moving average in logged records; 1 disables smoothing')
    parser.add_argument('--once', action='store_true', help='Save one PNG and exit (works without a GUI)')
    args = parser.parse_args(argv)
    if args.interval <= 0 or not np.isfinite(args.interval) or args.window < 1:
        parser.error('interval must be positive and finite; window must be at least 1')
    if not args.once and plt.get_backend().lower() in {'agg', 'pdf', 'svg', 'ps', 'cairo', 'template'}:
        parser.error('Matplotlib has no interactive window backend. Run in a local graphical terminal, or use --once.')
    viewer = LiveTrainingPlot(args.run, args.window)
    try:
        if args.once:
            if not viewer.refresh():
                parser.error('No recorded training metrics yet.')
            output = viewer.run / 'figures/live/training_live.png'
            output.parent.mkdir(parents=True, exist_ok=True)
            viewer.fig.savefig(output, dpi=150)
            print(output)
            return
        plt.ion()
        plt.show(block=False)
        print(f'Reading {viewer.log_dir}; close the window or press Ctrl+C to exit.', flush=True)
        while plt.fignum_exists(viewer.fig.number):
            viewer.refresh()
            plt.pause(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        plt.close(viewer.fig)


if __name__ == '__main__':
    main()
