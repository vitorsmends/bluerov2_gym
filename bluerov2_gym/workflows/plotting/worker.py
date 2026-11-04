"""Run each plotting routine in an isolated process (Matplotlib state)."""
import importlib
import json
import sys
from pathlib import Path


def main():
    import matplotlib
    matplotlib.use('Agg')
    from .registry import PLOTS
    name, settings = sys.argv[1], json.loads(sys.argv[2])
    module=importlib.import_module(f'bluerov2_gym.workflows.plotting.{PLOTS[name].module}')
    if name=='single_policy':
        import pandas as pd
        from ..configuration import read_yaml
        data=pd.read_csv(Path(settings['input_dir'])/'stationkeeping.csv').to_numpy()
        manifest=read_yaml(Path(settings['input_dir'])/'evaluation.yaml')
        summary=manifest['summary']
        Path(settings['output_dir']).mkdir(parents=True,exist_ok=True)
        module.plot_errors(data,manifest['target_m'],manifest['resolved_disturbance_model']['preset'],
                           summary['terminated'],summary['truncated'],Path(settings['output_dir']))
    elif name=='training':
        args=['--log-dir',settings['log_dir'],'--output-dir',settings['output_dir']]
        for key in ['run','smooth_window','min_seeds']:
            if key in settings:args += ['--'+key.replace('_','-'),str(settings[key])]
        for group in settings.get('groups',[]):args += ['--group',group]
        module.main(args)
    elif name=='trajectory_spatial':
        module.main(['--repetition-id',str(settings.get('repetition_id',0)),
                     '--representative-trajectory',settings.get('representative_trajectory','figure_eight_depth')],settings=settings)
    else:
        module.main(settings=settings)

if __name__=='__main__':main()
