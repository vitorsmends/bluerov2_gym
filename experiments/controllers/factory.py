"""Controller factory for the implemented experiment controllers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .nmpc import NMPCController
from .pid import PIDController
from .ppo import PPOController
from .smc import SMCController


class ControllerFactory:
    """Instantiate controllers from the common experiment configuration."""

    _registry = {
        "pid": PIDController,
        "ppo": PPOController,
        "smc": SMCController,
        "nmpc": NMPCController,
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

        params = dict((config or {}).get("params", {}))

        # PPO does not require the vehicle dynamics or the reference provider.
        if key == "ppo":
            return controller_cls(**params)

        return controller_cls(
            dynamics=dynamics,
            dt=dt,
            reference_provider=reference_provider,
            **params,
        )