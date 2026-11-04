"""Single default location for generated and imported experiment artifacts."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_ROOT = PROJECT_ROOT / 'artifacts'
CURRENT_MODELS = ARTIFACTS_ROOT / 'models' / 'current'
CURRICULUM_MODELS = ARTIFACTS_ROOT / 'models' / 'curriculum'
TENSORBOARD_LOGS = ARTIFACTS_ROOT / 'logs' / 'tensorboard'
IMPORTED_RESULTS = ARTIFACTS_ROOT / 'evaluation' / 'imported'


def artifact_path(*parts):
    """Return a destination; directories are created only when writing."""
    return ARTIFACTS_ROOT.joinpath(*parts)
