"""Use the same physical model for controller design and experiment execution."""
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bluerov2_gym.envs.core.config_utils import load_yaml
from bluerov2_gym.envs.core.dynamics import Dynamics as PlantDynamics


class Dynamics(PlantDynamics):
    def __init__(self, **kwargs):
        kwargs.setdefault('dynamics_config', load_yaml(PROJECT_ROOT / 'config/dynamics_config.yaml'))
        super().__init__(**kwargs)
