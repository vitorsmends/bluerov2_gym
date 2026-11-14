"""Controller factory for the implemented experiment controllers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from importlib import import_module


class ControllerFactory:
    """Instantiate controllers from the common experiment configuration."""

    _registry = {
        "pid": ("pid", "PIDController"),
        "ppo": ("ppo", "PPOController"),
        "smc": ("smc", "SMCController"),
        "nmpc": ("nmpc", "NMPCController"),
    }

    @classmethod
    def register(
        cls,
        name: str,
        controller_cls: Callable[..., Any],
    ) -> None:
        """Register another controller."""
        cls._registry[name.strip().lower()] = controller_cls

    @classmethod
    def create(
        cls,
        name: str,
        *,
        dynamics: Any,
        reference_provider: Any,
        dt: float,
        config: dict[str, Any] | None = None,
    ) -> Any:

        key = name.strip().lower()

        controller_cls = cls._registry.get(key)

        if controller_cls is None:
            implemented = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"Controller '{name}' is not implemented. "
                f"Available controllers: {implemented}."
            )

        if isinstance(controller_cls, tuple):
            module, class_name = controller_cls
            controller_cls = getattr(import_module(f"{__package__}.{module}"), class_name)

        params = dict((config or {}).get("params", {}))

        # PPO does not require the vehicle dynamics or the reference provider.
        if key == "ppo":
            if params.get('action_control', {}).get('mode') == 'body_wrench':
                params['allocation_dynamics'] = dynamics
            return controller_cls(**params)

        return controller_cls(
            dynamics=dynamics,
            dt=dt,
            reference_provider=reference_provider,
            **params,
        )
