"""Plot names and suites; no Matplotlib/controller imports here."""
from dataclasses import dataclass

@dataclass(frozen=True)
class PlotSpec:
    module: str
    family: str
    required_columns: tuple[str, ...] = ()

PLOTS = {
    'tracking_error': PlotSpec('article.plot_tracking_error','article',('time','tracking_error_m')),
    'rmse': PlotSpec('article.plot_rmse','article',('time','tracking_error_m')),
    'cpu_time': PlotSpec('article.plot_cpu_time','article',('time','controller_cpu_time_s')),
    'energy': PlotSpec('article.plot_energy','article',('time',*[f'T{i}' for i in range(1,7)])),
    'pareto': PlotSpec('article.plot_pareto','article',('time','tracking_error_m','controller_cpu_time_s')),
    'stationkeeping': PlotSpec('stationkeeping.stationkeeping','stationkeeping',('target_id','repetition_id','position_error_m')),
    'station_cost': PlotSpec('stationkeeping.station_cost','stationkeeping',('target_id','repetition_id','position_error_m')),
    'trajectory_spatial': PlotSpec('trajectories.plot_trajectory','trajectories',('trajectory','x_ref','y_ref')),
    'path_tracking': PlotSpec('trajectories.plot','legacy_path_tracking',('tracking_error_m','time')),
    'cost': PlotSpec('trajectories.cost','legacy_path_tracking',('controller_cpu_time_s','time')),
    'cost_tradeoff': PlotSpec('trajectories.cost_tradeoff','legacy_path_tracking',('tracking_error_m','controller_cpu_time_s')),
    'violin': PlotSpec('trajectories.violin','legacy_path_tracking',('tracking_error_m',)),
    'single_policy': PlotSpec('stationkeeping.single_policy','single_policy',('error_x','error_y','error_z')),
    'training': PlotSpec('training.curves','training'),
    'optimization': PlotSpec('training.optimization','optimization'),
    'ftc': PlotSpec('ftc.comparison','ftc',('x','y','x_ref','y_ref','tracking_error_m')),
}
SUITES = {
    'article': ('tracking_error','rmse','cpu_time','energy','pareto'),
    'stationkeeping': ('stationkeeping','station_cost'),
    'trajectories': ('trajectory_spatial',),
    'single_policy': ('single_policy',),
    'legacy_path_tracking': ('path_tracking','cost','cost_tradeoff','violin'),
    'training': ('training',), 'optimization': ('optimization',), 'ftc': ('ftc',),
}
