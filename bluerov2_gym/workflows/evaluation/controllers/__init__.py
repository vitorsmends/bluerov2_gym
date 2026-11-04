"""Controller exports without importing optional dependencies eagerly."""
from importlib import import_module
_MODULES = {'BaseController':'base', 'PIDController':'pid', 'PPOController':'ppo',
            'NMPCController':'nmpc', 'SMCController':'smc'}
__all__ = list(_MODULES)
def __getattr__(name):
    if name in _MODULES:
        return getattr(import_module(f'{__name__}.{_MODULES[name]}'), name)
    raise AttributeError(name)
