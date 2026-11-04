"""Compatibility entry point; implementation: bluerov2_gym.workflows.plotting.ftc.comparison."""
import importlib
import runpy
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parents[0])
if _root not in sys.path:
    sys.path.insert(0, _root)
if __name__ == "__main__":
    runpy.run_module("bluerov2_gym.workflows.plotting.ftc.comparison", run_name="__main__")
else:
    sys.modules[__name__] = importlib.import_module("bluerov2_gym.workflows.plotting.ftc.comparison")
