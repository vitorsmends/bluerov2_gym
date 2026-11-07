"""Publication figures from a recorded curriculum, without rerunning simulation."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import EngFormatter
import numpy as np
import yaml

from .curves import find_run, moving_average


class CurriculumFigures:
    def __init__(self, settings):
        self.settings = settings
        self.root = Path(settings['log_dir']).parents[1]
        self.output = Path(settings['output_dir'])
        self.tables = Path(settings['table_dir'])
        self.output.mkdir(parents=True, exist_ok=True)
        self.tables.mkdir(parents=True, exist_ok=True)
        _, self.metrics = find_run(Path(settings['log_dir']), settings.get('run'))
        self.status = settings.get('curriculum_status', {})
        self.phases = list(self.status.get('history', []))
        if not self.phases or self.phases[-1]['index'] != self.status.get('index'):
            self.phases.append(self.status)
        self.phases = [s for s in self.phases if 'start_step' in s]
        self.colors = plt.get_cmap('tab10').colors
        self.window = int(settings.get('smooth_window', 10))
        if self.window < 1:
            raise ValueError('smooth_window must be positive.')
        self.batches = []
        for path in (self.root/'logs/curriculum_evaluations').glob('*.yaml'):
            with path.open() as stream:
                self.batches.append(yaml.safe_load(stream))
        self.batches.sort(key=lambda b: b['step'])

    def save(self, fig, name):
        for suffix in ['pdf', 'png']:
            fig.savefig(self.output/f'{name}.{suffix}', dpi=300, bbox_inches='tight')
        plt.close(fig)

    def decorate(self, axes):
        for ax in np.asarray(axes).reshape(-1):
            for i, phase in enumerate(self.phases):
                end = phase.get('end_step', phase.get('step', phase['start_step']))
                ax.axvspan(phase['start_step'], end, color=self.colors[i], alpha=.06)
                if i:
                    ax.axvline(phase['start_step'], color='.45', lw=.65, ls=':')
            ax.grid(alpha=.2)
            ax.xaxis.set_major_formatter(EngFormatter())
        handles = [Patch(color=self.colors[i], alpha=.5, label=p['name'].split(' - ')[0])
                   for i, p in enumerate(self.phases)]
        np.asarray(axes).reshape(-1)[0].legend(handles=handles, loc='upper right', ncol=6, fontsize=8)

    def learning(self):
        specifications = [('rollout/reward_mean_per_step', 'Reward bruto médio por passo', '#29996a'),
                          ('rollout/position_error_mean_m', 'Erro euclidiano médio (m)', '#357ac5'),
                          ('rollout/position_error_rmse_m', 'RMSE por rollout (m)', '#dc625b')]
        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True, layout='constrained')
        for ax, (tag, label, color) in zip(axes, specifications):
            ax.set_ylabel(label)
            if tag not in self.metrics:
                ax.text(.5, .5, 'Métrica não registrada', transform=ax.transAxes, ha='center')
                continue
            steps, values = self.metrics[tag]
            ax.plot(steps, values, color=color, alpha=.23, lw=.55)
            # Never average data from different mechanics/sea stages together.
            for phase in self.phases:
                end = phase.get('end_step', phase.get('step', np.inf))
                mask = (steps > phase['start_step']) & (steps <= end)
                if mask.any():
                    y = moving_average(values[mask], self.window)
                    ax.plot(steps[mask][-len(y):], y, color=color, lw=1.25)
        self.decorate(axes)
        axes[-1].set_xlabel('Passos de treinamento no ambiente')
        self.save(fig, 'curriculum_learning')

    def validation(self):
        if not self.batches:
            return
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, layout='constrained')
        for i, phase in enumerate(self.phases):
            rows = [b for b in self.batches if b['phase'] == phase['index']]
            steps = [b['step'] for b in rows]
            axes[0].plot(steps, [100*b['success_rate'] for b in rows], '.-', color=self.colors[i], lw=.9, ms=3)
            valid = [b for b in rows if b.get('mean_rmse_m') is not None]
            axes[1].plot([b['step'] for b in valid], [b['mean_rmse_m'] for b in valid], '.-', color=self.colors[i], lw=.9, ms=3)
            passed = [b for b in rows if b['passed']]
            axes[0].scatter([b['step'] for b in passed], [100*b['success_rate'] for b in passed], marker='*', s=28, color=self.colors[i])
        axes[0].set(ylabel='Episódios aprovados (%)', ylim=(-3, 105))
        axes[1].set_ylabel('Média do RMSE dos episódios (m)')
        threshold = self.settings['training_config']['curriculum_transition']['max_rmse_m']
        axes[1].axhline(threshold, color='.3', lw=.8, ls='--')
        axes[-1].set_xlabel('Passos de treinamento no ambiente')
        self.decorate(axes)
        self.save(fig, 'curriculum_validation')

    def costs_and_tables(self):
        phases = []
        for phase in self.phases:
            end = phase.get('end_step', phase.get('step', phase['start_step']))
            phases.append(dict(phase=phase['index'], name=phase['name'],
                               start_step=phase['start_step'], end_step=end,
                               training_steps=end-phase['start_step'],
                               validation_steps=phase.get('validation_timesteps', 0),
                               status=phase.get('status'),
                               final_success_rate=phase.get('last_evaluation', {}).get('success_rate')))
        fig, ax = plt.subplots(figsize=(7.2, 3.4), layout='constrained')
        x = np.arange(len(phases))
        ax.bar(x-.18, [p['training_steps'] for p in phases], .36, label='Treino', color='#357ac5')
        ax.bar(x+.18, [p['validation_steps'] for p in phases], .36, label='Validação', color='#e5a246')
        ax.set_xticks(x, [p['name'].split(' - ')[0] for p in phases])
        ax.set(xlabel='Fase', ylabel='Interações com o simulador')
        ax.yaxis.set_major_formatter(EngFormatter()); ax.grid(axis='y', alpha=.2); ax.legend()
        self.save(fig, 'curriculum_sample_cost')
        self.csv('phases.csv', phases)
        self.csv('validation_batches.csv', [{k: b[k] for k in ['step', 'phase', 'success_rate', 'mean_rmse_m', 'passed']} for b in self.batches])
        episodes = []
        for batch in self.batches:
            for ep in batch['episodes']:
                row = {k: ep.get(k) for k in ['group', 'steps', 'completed', 'success', 'rmse_m', 'rmse_full_m', 'tracking_fraction', 'reset_seed']}
                row.update(step=batch['step'], phase=batch['phase'], **{k: ep['context']['jonswap'].get(k) for k in ['Hs', 'Tp', 'seed']})
                episodes.append(row)
        self.csv('validation_episodes.csv', episodes)
        self.csv('training_metrics.csv', [dict(step=int(s), tag=tag, value=float(v)) for tag, (steps, values) in self.metrics.items() for s, v in zip(steps, values)])
        sources = [self.root/'config.resolved.yaml', self.root/'manifest.yaml', self.root/'models/curriculum_status.yaml']
        sources += list((self.root/'logs/curriculum_evaluations').glob('*.yaml'))
        sources += list((self.root/'logs/tensorboard').rglob('events.out.tfevents.*'))
        (self.tables/'provenance.json').write_text(json.dumps(dict(run=str(self.root), smoothing_window=self.window,
            input_sha256={str(p.relative_to(self.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources if p.is_file()}), indent=2))
        (self.output/'captions.md').write_text('''Figuras de uma execução de treinamento, seed 42; sem média entre seeds de PPO.

curriculum_learning: reward bruto e erros de posição por rollout. Traços claros:
valores registrados; traços fortes: média móvel de até 10 registros, separada
por fase. Fundos e pontilhados indicam mudanças de mecânica/estado do mar.

curriculum_validation: política determinística e normalização congelada em
episódios de 100 s. RMSE usa o intervalo após 10 s de acomodação. Estrelas:
baterias aprovadas pelo critério completo, incluindo grupos. Os CSVs também
contêm RMSE incluindo o transitório. Sucesso não é uma estimativa entre agentes.

curriculum_sample_cost: passos de atualização e de validação por fase, separados.
A contagem global do PPO não inclui as interações adicionais de validação.
''')

    def csv(self, name, rows):
        if not rows:
            return
        with (self.tables/name).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    def generate(self):
        with plt.rc_context({'font.family':'serif', 'font.size':10, 'pdf.fonttype':42, 'ps.fonttype':42}):
            self.learning(); self.validation(); self.costs_and_tables()


def main(settings=None):
    CurriculumFigures(settings).generate()
