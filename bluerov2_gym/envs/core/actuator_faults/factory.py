from __future__ import annotations

from typing import Any

from .deadtime import DeadTime
from .loss_of_effectiveness import LossOfEffectiveness
from .manager import FaultManager
from .outage import CompleteOutage
from .saturation import ReducedSaturation
from .stuck import StuckThruster


class FaultFactory:
    """
    Factory responsible for creating actuator faults and FaultManager
    instances from configuration dictionaries.

    Expected configuration example
    ------------------------------

    faults = [
        {
            "type": "loss_of_effectiveness",
            "thruster": 2,
            "params": {
                "effectiveness": 0.75
            }
        },
        {
            "type": "deadtime",
            "thruster": 5,
            "params": {
                "delay_steps": 3
            }
        }
    ]
    """

    @staticmethod
    def build_fault(config: dict):
        """
        Build a single fault object.

        Parameters
        ----------
        config : dict

        Returns
        -------
        BaseActuatorFault
        """

        fault_type = str(config["type"]).lower().strip()
        params = dict(config.get("params", {}))

        if fault_type in {
            "loss_of_effectiveness",
            "loe",
        }:
            return LossOfEffectiveness(
                effectiveness=float(params["effectiveness"])
            )

        if fault_type in {
            "complete_outage",
            "outage",
        }:
            return CompleteOutage()

        if fault_type in {
            "stuck",
            "stuck_thruster",
        }:
            return StuckThruster(
                stuck_value=float(params["stuck_value"])
            )

        if fault_type in {
            "reduced_saturation",
            "saturation",
        }:
            return ReducedSaturation(
                limit=float(params["limit"])
            )

        if fault_type in {
            "deadtime",
            "dead_time",
        }:
            return DeadTime(
                delay_steps=int(params["delay_steps"])
            )

        raise ValueError(
            f"Unknown actuator fault type: {fault_type}"
        )

    @classmethod
    def build_manager(
        cls,
        faults: list[dict[str, Any]] | None,
    ) -> FaultManager | None:
        """
        Build a FaultManager from a list of fault configurations.

        Parameters
        ----------
        faults
            List of dictionaries.

        Returns
        -------
        FaultManager | None
        """

        if faults is None or len(faults) == 0:
            return None

        manager = FaultManager()

        for cfg in faults:

            fault = cls.build_fault(cfg)

            manager.add_fault(
                thruster=int(cfg["thruster"]),
                fault=fault,
            )

        return manager