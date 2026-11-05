"""Organize unchanged plotting routines, using saved data only."""
from ..paths import artifact_path
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from ..configuration import read_yaml
from ..artifacts import write_yaml
from .registry import PLOTS, SUITES


class PlotManager:
    def __init__(self, run_directory=None, config=None):
        self.run = Path(run_directory).expanduser().resolve() if run_directory else None
        self.config = dict(config or {})
        if self.run:
            self.resolved=read_yaml(self.run/'config.resolved.yaml')
            self.output=self.run/'figures'
            self.tables=self.run/'tables'
            self.logs=self.run/'logs'
        else:
            self.resolved={}
            self.output=Path(self.config.get('output_dir',artifact_path('figures','manual'))).resolve()
            self.tables=Path(self.config.get('table_dir',self.output/'tables')).resolve()
            self.logs=self.output/'logs'

    def generate_suite(self, name):
        if name not in SUITES:raise ValueError(f'Unknown suite {name!r}; available: {list(SUITES)}')
        return self.generate(SUITES[name])

    def settings(self, name):
        spec=PLOTS[name]
        settings=dict(self.config)
        settings.update(self.config.get('plots',{}).get(name,{}))
        settings.setdefault('output_dir',str(self.output/('article' if spec.family=='article' else name)))
        settings.setdefault('table_dir',str(self.tables/('article' if spec.family=='article' else name)))
        if self.run:
            data=self.run/'data'
            experiment=self.resolved.get('experiment',{})
            mode=experiment.get('mode')
            settings.setdefault('controllers',[c['name'] for c in self.resolved.get('controllers',[]) if c.get('enabled',True)])
            if spec.family=='article':
                folders={p.name.capitalize():str(p) for p in sorted(data.iterdir()) if p.is_dir() and list(p.glob('*.csv'))}
                settings.setdefault('scenarios',folders)
            elif spec.family=='trajectories':
                for scenario in ([settings["scenario"]] if settings.get("scenario") else self._scenarios()):
                    if (data/scenario).is_dir():
                        settings.setdefault('scenario',scenario)
                        settings.setdefault('input_dir',str(data/scenario));break
            elif spec.family in {'stationkeeping','single_policy','legacy_path_tracking','checkpoints'}:
                settings.setdefault('input_dir',str(data))
            elif spec.family=='training':
                settings.setdefault('training_config',self.resolved)
                phase_status=self.run/'models/curriculum_status.yaml'
                if phase_status.is_file():
                    settings.setdefault('curriculum_status',read_yaml(phase_status))
                settings.setdefault('log_dir',str(self.run/'logs/tensorboard'))
                logs=[p for p in Path(settings['log_dir']).glob('*')
                      if p.is_dir() and list(p.glob('events.out.tfevents.*'))]
                if len(logs)==1 and not settings.get('groups'):
                    settings.setdefault('run',logs[0].name)

        if spec.family=='trajectories':
            explicit=self.config.get('plots',{}).get(name,{})
            if self.run and settings.get('scenario'):
                if 'output_dir' not in self.config and 'output_dir' not in explicit:
                    settings['output_dir']=str(self.output/name/settings['scenario'])
                if 'table_dir' not in self.config and 'table_dir' not in explicit:
                    settings['table_dir']=str(self.tables/name/settings['scenario'])
            settings.setdefault('representative_output_dir',str(Path(settings['output_dir'])/'representative'))
        return settings

    def _scenarios(self):
        disturbance=self.resolved.get('disturbance',{})
        return disturbance.get('scenario_names') or [disturbance.get('default_scenario','moderate')]

    def _validate(self, name, settings):
        spec=PLOTS[name]
        if spec.family in {'training','optimization'}:return
        if spec.family=='article':
            folders=[Path(p) for p in settings.get('scenarios',{}).values()]
        elif spec.family=='ftc':
            paths=[Path(settings[k]) for k in ['nominal','ftc'] if settings.get(k)]
            if len(paths)!=2:raise ValueError('FTC requires explicit nominal and ftc CSV paths.')
            folders=[]
        else:
            if not settings.get('input_dir'):raise ValueError(f'{name}: input_dir is required.')
            folders=[Path(settings['input_dir'])]
        if spec.family!='ftc': paths=[p for folder in folders for p in folder.glob('*.csv')]
        if not paths:raise ValueError(f'{name}: no CSV inputs found; check run/suite/input_dir.')
        valid=False
        for path in paths:
            with path.open(newline='') as stream:columns=next(csv.reader(stream),[])
            if set(spec.required_columns)<=set(columns):valid=True
        if not valid:raise ValueError(f'{name}: required columns {spec.required_columns} not found.')

    def generate(self, names):
        names=list(names)
        for name in names:
            if name not in PLOTS:raise ValueError(f'Unknown plot: {name}')
            self._validate(name,self.settings(name))
        tasks=[]
        for name in names:
            settings=self.settings(name)
            if self.run and PLOTS[name].family=='trajectories' and not self.config.get('scenario'):
                for scenario in self._scenarios():
                    variant=dict(settings,scenario=scenario,input_dir=str(self.run/'data'/scenario),
                                 output_dir=str(self.output/name/scenario),table_dir=str(self.tables/name/scenario),
                                 representative_output_dir=str(self.output/name/scenario/'representative'))
                    self._validate(name,variant)
                    tasks.append((name,variant))
            else:tasks.append((name,settings))
        self.logs.mkdir(parents=True,exist_ok=True)
        package_root=Path(__file__).resolve().parents[3]
        env=dict(os.environ,MPLBACKEND='Agg')
        env['PYTHONPATH']=str(package_root)+os.pathsep+env.get('PYTHONPATH','')
        for name,settings in tasks:
            print(f'[PLOT] {name} → {settings["output_dir"]}',flush=True)
            result=subprocess.run([sys.executable,'-m','bluerov2_gym.workflows.plotting.worker',name,json.dumps(settings)],
                                  env=env,capture_output=True,text=True)
            log_path=self.logs/f'plot_{name}_{settings.get("scenario", "all")}.log'
            log_path.write_text(result.stdout+result.stderr)
            if result.returncode:raise RuntimeError(f'Plot {name} failed. See {log_path}\n{result.stderr[-1500:]}')
        outputs=sorted(str(p) for p in self.output.rglob('*') if p.is_file() and p.suffix in {'.png','.pdf'})
        write_yaml(self.logs/'plots.yaml',dict(plots=names,outputs=outputs))
        return outputs
