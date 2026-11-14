"""Temporal position/yaw error distributions for paired Walker sea tests.

One realization per sea: these boxes describe correlated time samples, not
variation across independent seeds or distributions of run-level RMSE.
"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate(run, settling_seconds=10.):
    run = Path(run).resolve()
    if not np.isfinite(settling_seconds) or not 0 <= settling_seconds < 100:
        raise ValueError('settling_seconds must be between 0 and 100, exclusive of 100.')
    out = run / 'figures'
    out.mkdir(exist_ok=True)
    seas = ['walker_w1', 'walker_w2', 'walker_w3']
    colors = {'PPO': '#0072B2', 'NMPC': '#E15759'}
    data, full_data, hashes, stats = {}, {}, {}, []
    for sea in seas:
        for name in colors:
            path = (run / f'evaluation/data/{sea}/ppo_figure_eight_depth.csv'
                    if name == 'PPO' else run / f'nmpc_cached/{sea}.csv')
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            df = pd.read_csv(path).sort_values('time')
            if len(df) != 1000 or df.repetition_id.nunique() != 1 or not np.allclose(df.time, np.arange(1000)*.1):
                raise ValueError(f'Expected one complete paired 100 s realization: {path}')
            full_data[sea, name] = df.copy()
            df = df.loc[df.time >= settling_seconds].copy()
            df['yaw_abs_deg'] = np.degrees(np.abs(np.arctan2(np.sin(df.yaw_error), np.cos(df.yaw_error))))
            data[sea, name] = df
            for column in ['tracking_error_m', 'yaw_abs_deg']:
                values = df[column].to_numpy()
                if not np.all(np.isfinite(values)):
                    raise ValueError(f'Non-finite errors in {path}')
                stats.append(dict(sea=sea, controller=name, metric=column, samples=len(values),
                                  mean=values.mean(), median=np.median(values),
                                  q25=np.quantile(values, .25), q75=np.quantile(values, .75)))
    with plt.rc_context({'font.family': 'serif', 'font.size': 12, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
        for ax, column, label in [(axes[0], 'tracking_error_m', 'Position error [m]'),
                                  (axes[1], 'yaw_abs_deg', 'Absolute yaw error [deg]')]:
            for offset, (name, color) in zip([-.17, .17], colors.items()):
                boxes = [data[sea, name][column].to_numpy() for sea in seas]
                ax.boxplot(boxes, positions=np.arange(3)+offset, widths=.27,
                           patch_artist=True, manage_ticks=False,
                           boxprops={'facecolor': color, 'alpha': .5},
                           medianprops={'color': 'black'},
                           flierprops={'marker': '.', 'markersize': 2, 'alpha': .25,
                                       'markeredgecolor': color})
                ax.plot([], [], color=color, lw=6, label=name)
            ax.set_xticks(range(3), ['W1', 'W2', 'W3'])
            ax.set_ylabel(label)
            ax.grid(axis='y', alpha=.2)
            ax.legend(frameon=False)
        stem = 'walker_boxplots' if settling_seconds == 10 else f'walker_boxplots_after_{settling_seconds:g}s'
        for ext in ['png', 'pdf']:
            fig.savefig(out / f'{stem}.{ext}', dpi=250)
        plt.close(fig)
        for sea in seas:
            fig, axes = plt.subplots(1, 3, figsize=(10, 3.6),
                                     gridspec_kw={'width_ratios': [1.7, 1., 1.]})
            spatial = axes[0]
            reference = full_data[sea, 'PPO']
            handles = spatial.plot(reference.x_ref, reference.y_ref, 'k--',
                                   lw=1.2, label='Reference')
            for name, color in colors.items():
                df = full_data[sea, name]
                handles += spatial.plot(df.x, df.y, color=color, lw=1.2, label=name)
            spatial.set_aspect('equal', adjustable='box')
            spatial.set_xlabel('X [m]')
            spatial.set_ylabel('Y [m]')
            spatial.grid(alpha=.2)
            for ax, column, label in [(axes[1], 'tracking_error_m', 'Position error [m]'),
                                      (axes[2], 'yaw_abs_deg', 'Absolute yaw error [deg]')]:
                for position, (name, color) in enumerate(colors.items(), 1):
                    ax.boxplot([data[sea, name][column].to_numpy()], positions=[position],
                               widths=.5, patch_artist=True, manage_ticks=False,
                               boxprops={'facecolor': color, 'alpha': .5},
                               medianprops={'color': 'black'},
                               flierprops={'marker': '.', 'markersize': 2, 'alpha': .25,
                                           'markeredgecolor': color})
                ax.set_xticks([1, 2], list(colors))
                ax.set_ylabel(label)
                ax.grid(axis='y', alpha=.2)
            fig.legend(handles=handles, loc='upper center', ncol=3, frameon=False)
            fig.tight_layout(rect=[0, 0, 1, .87], w_pad=1.4)
            name = f'{stem}_{sea}_trajectory'
            for ext in ['png', 'pdf']:
                fig.savefig(out / f'{name}.{ext}', dpi=250)
            plt.close(fig)
    pd.DataFrame(stats).to_csv(out / f'{stem}_statistics.csv', index=False)
    (out / f'{stem}_provenance.json').write_text(json.dumps(dict(
        time_interval_seconds=[settling_seconds, 100], sha256=hashes,
        trajectory_interval_seconds=[0, 100], trajectory_projection='XY; reference depth -5 m',
        interpretation='Temporal error distribution, one paired wave realization per sea; not independent-seed uncertainty or run-level RMSE.',
        whiskers='1.5 IQR; outliers shown; median and 25th/75th percentiles.'), indent=2))
    print(out / f'{stem}.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--settling-seconds', type=float, default=10.)
    args = parser.parse_args()
    generate(args.run, args.settling_seconds)
