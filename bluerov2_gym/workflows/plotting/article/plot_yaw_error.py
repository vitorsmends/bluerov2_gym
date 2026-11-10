"""Absolute wrapped yaw error: mean and population standard deviation in degrees."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as defaults
from .plot_tracking_error import aggregate_runs
from .run_data import iter_csv_runs


def main(settings=None):
    settings = settings or {}
    output = Path(settings.get('output_dir', defaults.OUTPUT_DIR))
    tables = Path(settings.get('table_dir', defaults.TABLE_DIR))
    output.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    scenarios = {k: Path(v) for k, v in settings.get('scenarios', defaults.SCENARIOS).items()}
    if settings.get('pool_scenarios', False):
        panels = [('All sea states', list(scenarios.values()))]
    else:
        panels = [(scenario, [folder]) for scenario, folder in scenarios.items()]
    selected = {c.lower() for c in settings.get('controllers', defaults.CONTROLLERS)}
    controllers = {k: v for k, v in defaults.CONTROLLERS.items() if k.lower() in selected}
    exported = []
    with plt.rc_context({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(len(panels), 1,
                                 figsize=(defaults.FIGSIZE[0], defaults.FIGSIZE[1]*len(panels)),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, (scenario, folders) in zip(axes[:, 0], panels):
            for controller, info in controllers.items():
                runs = []
                for folder in folders:
                    for path, repetition, df in iter_csv_runs(
                            folder, info['prefix'], {'time', 'yaw_error'},
                            trajectories=settings.get('trajectory_names')):
                        run = df[['time', 'yaw_error']].copy()
                        angle = run['yaw_error'].to_numpy(dtype=float)
                        # Magnitudes cannot cancel across positive/negative errors;
                        # wrap first so +/-pi crossings never appear as 360 degrees.
                        run['yaw_error'] = np.degrees(np.abs(np.arctan2(np.sin(angle), np.cos(angle))))
                        run.attrs['run_id'] = f'{path.stem}_rep{repetition}'
                        runs.append(run)
                result = aggregate_runs(runs, column='yaw_error')
                if result is None:
                    continue
                time, mean, std = result
                ax.plot(time, mean, color=info['color'], linewidth=defaults.LINE_WIDTH, label=controller)
                ax.fill_between(time, np.maximum(0., mean-std), np.minimum(180., mean+std),
                                color=info['color'], alpha=defaults.BAND_ALPHA, linewidth=0)
                exported.extend(dict(scenario=scenario, controller=controller, time_s=float(t),
                                     mean_absolute_yaw_error_deg=float(m), std_absolute_yaw_error_deg=float(s))
                                for t, m, s in zip(time, mean, std))
            ax.set_ylabel('Absolute yaw error [deg]')
            ax.set_ylim(bottom=0.)
            ax.grid(True, linestyle='--', alpha=.5)
        if not exported:
            plt.close(fig)
            raise ValueError('No yaw_error samples found for the selected controllers/trajectories.')
        axes[-1, 0].set_xlabel('Time [s]')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=max(1, len(labels)),
                   frameon=True, bbox_to_anchor=(.5, .995))
        fig.tight_layout(rect=(0, 0, 1, .90 if len(panels) == 1 else .95))
        for extension in ['png', 'pdf']:
            fig.savefig(output/f'yaw_error_mean_std.{extension}', dpi=defaults.DPI, bbox_inches='tight')
        plt.close(fig)
    pd.DataFrame(exported).to_csv(tables/'yaw_error_mean_std.csv', index=False)
