from __future__ import annotations

from typing import Any

import numpy as np


def _sample_numeric_value(
    value_cfg: Any,
    rng: np.random.Generator,
    *,
    integer: bool = False,
):
    """
    Resolve a scalar value or sample a value from a ``min``/``max`` range.

    Parameters
    ----------
    value_cfg:
        Numeric scalar or dictionary containing ``min`` and ``max``.

    rng:
        NumPy random-number generator used for reproducible sampling.

    integer:
        When True, sample an integer from an inclusive range.

    Returns
    -------
    int | float
        The resolved numeric value.
    """
    if isinstance(value_cfg, dict):
        if "min" not in value_cfg or "max" not in value_cfg:
            raise ValueError(
                "Numeric ranges must contain 'min' and 'max'."
            )

        minimum = value_cfg["min"]
        maximum = value_cfg["max"]

        if integer:
            minimum = int(minimum)
            maximum = int(maximum)

            if maximum < minimum:
                raise ValueError(
                    f"Invalid integer range: [{minimum}, {maximum}]"
                )

            return int(
                rng.integers(
                    low=minimum,
                    high=maximum + 1,
                )
            )

        minimum = float(minimum)
        maximum = float(maximum)

        if maximum < minimum:
            raise ValueError(
                f"Invalid numeric range: [{minimum}, {maximum}]"
            )

        return float(rng.uniform(minimum, maximum))

    return int(value_cfg) if integer else float(value_cfg)


def sample_fault_parameters(
    fault_type: str,
    params_cfg: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Sample the parameters associated with one actuator-fault type.

    Notes
    -----
    ``effectiveness`` represents the remaining actuator effectiveness:

    - ``1.0``: nominal actuator;
    - ``0.7``: actuator retains 70% effectiveness;
    - ``0.0``: complete loss of effectiveness.
    """
    normalized_type = str(fault_type).strip().lower()
    params_cfg = dict(params_cfg or {})

    if normalized_type in {
        "loe",
        "loss_of_effectiveness",
    }:
        return {
            "effectiveness": _sample_numeric_value(
                params_cfg["effectiveness"],
                rng,
            )
        }

    if normalized_type in {
        "outage",
        "complete_outage",
    }:
        return {}

    if normalized_type in {
        "stuck",
        "stuck_thruster",
    }:
        return {
            "stuck_value": _sample_numeric_value(
                params_cfg["stuck_value"],
                rng,
            )
        }

    if normalized_type in {
        "saturation",
        "reduced_saturation",
    }:
        return {
            "limit": _sample_numeric_value(
                params_cfg["limit"],
                rng,
            )
        }

    if normalized_type in {
        "deadtime",
        "dead_time",
    }:
        return {
            "delay_steps": _sample_numeric_value(
                params_cfg["delay_steps"],
                rng,
                integer=True,
            )
        }

    raise ValueError(
        f"Unsupported actuator fault type: {fault_type}"
    )


def _weighted_candidate_index(
    candidates: list[dict],
    rng: np.random.Generator,
) -> int:
    """
    Select one fault candidate according to its configured weight.
    """
    weights = np.asarray(
        [
            float(candidate.get("weight", 1.0))
            for candidate in candidates
        ],
        dtype=np.float64,
    )

    if np.any(weights < 0.0):
        raise ValueError(
            "Fault candidate weights cannot be negative."
        )

    weight_sum = float(weights.sum())

    if weight_sum <= 0.0:
        raise ValueError(
            "At least one fault candidate must have a positive weight."
        )

    probabilities = weights / weight_sum

    return int(
        rng.choice(
            len(candidates),
            p=probabilities,
        )
    )


def _sample_number_of_faults(
    number_cfg: int | dict,
    rng: np.random.Generator,
) -> int:
    """
    Resolve the number of simultaneous faults for one episode.
    """
    number_of_faults = _sample_numeric_value(
        number_cfg,
        rng,
        integer=True,
    )

    if number_of_faults < 0:
        raise ValueError(
            "The number of simultaneous faults cannot be negative."
        )

    return number_of_faults


def sample_faults_for_episode(
    stage_cfg: dict,
    n_thrusters: int,
    rng: np.random.Generator,
) -> list[dict]:
    """
    Sample the complete actuator-fault realization for one episode.

    Expected stage structure
    ------------------------
    .. code-block:: yaml

        fault_stage:
          activation_probability: 1.0
          number_of_faults: 1
          allow_same_thruster: false
          candidates:
            - type: loss_of_effectiveness
              weight: 1.0
              params:
                effectiveness:
                  min: 0.70
                  max: 0.80

    Parameters
    ----------
    stage_cfg:
        Fault-stage configuration from the FTC curriculum.

    n_thrusters:
        Number of available thruster commands.

    rng:
        NumPy random-number generator.

    Returns
    -------
    list[dict]
        Sampled fault descriptions ready for ``FaultFactory.build_manager``.
    """
    stage_cfg = dict(stage_cfg or {})
    n_thrusters = int(n_thrusters)

    if n_thrusters <= 0:
        raise ValueError(
            "n_thrusters must be greater than zero."
        )

    activation_probability = float(
        stage_cfg.get("activation_probability", 1.0)
    )

    if not 0.0 <= activation_probability <= 1.0:
        raise ValueError(
            "activation_probability must be between 0 and 1."
        )

    if rng.random() > activation_probability:
        return []

    candidates = list(stage_cfg.get("candidates", []))

    if not candidates:
        return []

    number_of_faults = _sample_number_of_faults(
        stage_cfg.get("number_of_faults", 1),
        rng,
    )

    if number_of_faults == 0:
        return []

    allow_same_thruster = bool(
        stage_cfg.get("allow_same_thruster", False)
    )

    if not allow_same_thruster and number_of_faults > n_thrusters:
        raise ValueError(
            "number_of_faults exceeds the number of available thrusters "
            "while allow_same_thruster is false."
        )

    if allow_same_thruster:
        selected_thrusters = [
            int(rng.integers(0, n_thrusters))
            for _ in range(number_of_faults)
        ]
    else:
        selected_thrusters = [
            int(index)
            for index in rng.choice(
                n_thrusters,
                size=number_of_faults,
                replace=False,
            )
        ]

    sampled_faults: list[dict] = []

    for thruster in selected_thrusters:
        candidate = candidates[
            _weighted_candidate_index(
                candidates,
                rng,
            )
        ]

        if "type" not in candidate:
            raise ValueError(
                "Each fault candidate must contain a 'type' field."
            )

        fault_type = str(candidate["type"])

        params = sample_fault_parameters(
            fault_type=fault_type,
            params_cfg=candidate.get("params", {}),
            rng=rng,
        )

        sampled_faults.append(
            {
                "type": fault_type,
                "thruster": thruster,
                "params": params,
            }
        )

    return sampled_faults


__all__ = [
    "sample_fault_parameters",
    "sample_faults_for_episode",
]
