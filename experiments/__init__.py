"""Compatibility exports, loaded on demand."""
from importlib import import_module
__all__ = ['ExperimentRunner']
def __getattr__(name):
    if name in __all__:
        return getattr(import_module('bluerov2_gym.workflows.evaluation.runner'), name)
    raise AttributeError(name)
