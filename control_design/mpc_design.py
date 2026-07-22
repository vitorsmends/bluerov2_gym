"""
Efficient NMPC design report for BlueROV2 path tracking with JONSWAP disturbances.

Goal:
    Reduce computational cost while preserving nonlinear prediction and disturbance rejection.

Main strategy:
    - Shorter prediction horizon
    - Move blocking
    - JONSWAP current and wave force forecast by blocks
    - Terminal cost compensation
    - Input-rate penalty
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import numpy as np


def weight_from_tolerance(max_value: float, priority: float) -> float:
    if max_value <= 0.0:
        raise ValueError("max_value must be positive.")
    return priority / (max_value ** 2)


@dataclass
class EfficientNMPCDesignConfig:
    # Sampling
    dt: float = 0.1

    # JONSWAP peak period from the plant model
    Tp: float = 12.0

    # Efficient horizon
    horizon_steps: int = 10

    # Move blocking
    control_blocks: int = 5

    # Physical error tolerances
    max_x_error: float = 0.30
    max_y_error: float = 0.30
    max_z_error: float = 0.25

    max_roll_error_deg: float = 12.0
    max_pitch_error_deg: float = 12.0
    max_yaw_error_deg: float = 15.0

    max_u_error: float = 0.35
    max_v_error: float = 0.35
    max_w_error: float = 0.25

    max_p_error_deg_s: float = 25.0
    max_q_error_deg_s: float = 25.0
    max_r_error_deg_s: float = 30.0

    # Priorities
    position_priority: float = 5.0
    attitude_priority: float = 1.5
    velocity_priority: float = 0.8
    angular_velocity_priority: float = 0.3

    # Stronger terminal cost compensates shorter horizon
    terminal_multiplier: float = 8.0

    # Actuator limits
    thrust_min: float = -40.0
    thrust_max: float = 40.0

    # Rate limits
    max_delta_thrust: float = 12.0

    # Input penalties
    input_priority: float = 0.08
    input_rate_priority: float = 0.35

    # Soft attitude constraints
    roll_soft_limit_deg: float = 25.0
    pitch_soft_limit_deg: float = 25.0
    attitude_soft_weight: float = 600.0

    # Disturbance parameters to pass to the NMPC internal predictor
    max_expected_current: float = 0.7
    max_expected_wave_force: float = 25.0
    max_expected_wave_moment: float = 8.0

    # Solver
    solver_name: str = "SLSQP"
    solver_ftol: float = 3e-3
    solver_maxiter: int = 12


class EfficientNMPCDesign:
    state_names = [
        "x", "y", "z",
        "roll", "pitch", "yaw",
        "u", "v", "w",
        "p", "q", "r",
    ]

    input_names = ["T1", "T2", "T3", "T4", "T5", "T6"]
    disturbance_names = ["u_c", "v_c", "w_c", "X_w", "Y_w", "Z_w", "K_w", "M_w", "N_w"]

    def __init__(self, cfg: EfficientNMPCDesignConfig):
        self.cfg = cfg

        if cfg.horizon_steps % cfg.control_blocks != 0:
            raise ValueError(
                "horizon_steps must be divisible by control_blocks "
                "for simple move blocking."
            )

        self.block_size = cfg.horizon_steps // cfg.control_blocks
        self.prediction_horizon_s = cfg.horizon_steps * cfg.dt

        self.full_decision_variables = cfg.horizon_steps * 6
        self.blocked_decision_variables = cfg.control_blocks * 6
        self.reduction_percent = (
            100.0
            * (1.0 - self.blocked_decision_variables / self.full_decision_variables)
        )

        self.state_max = self._state_max_vector()
        self.state_priorities = self._state_priority_vector()

        self.Q = np.diag(self.state_priorities / (self.state_max ** 2))
        self.Qf = cfg.terminal_multiplier * self.Q

        self.R = np.eye(6) * (
            cfg.input_priority / (abs(cfg.thrust_max) ** 2)
        )

        self.R_delta = np.eye(6) * (
            cfg.input_rate_priority / (cfg.max_delta_thrust ** 2)
        )

    def _state_max_vector(self) -> np.ndarray:
        cfg = self.cfg

        return np.array(
            [
                cfg.max_x_error,
                cfg.max_y_error,
                cfg.max_z_error,
                np.deg2rad(cfg.max_roll_error_deg),
                np.deg2rad(cfg.max_pitch_error_deg),
                np.deg2rad(cfg.max_yaw_error_deg),
                cfg.max_u_error,
                cfg.max_v_error,
                cfg.max_w_error,
                np.deg2rad(cfg.max_p_error_deg_s),
                np.deg2rad(cfg.max_q_error_deg_s),
                np.deg2rad(cfg.max_r_error_deg_s),
            ],
            dtype=float,
        )

    def _state_priority_vector(self) -> np.ndarray:
        cfg = self.cfg

        return np.array(
            [
                cfg.position_priority,
                cfg.position_priority,
                cfg.position_priority,
                cfg.attitude_priority,
                cfg.attitude_priority,
                cfg.attitude_priority,
                cfg.velocity_priority,
                cfg.velocity_priority,
                cfg.velocity_priority,
                cfg.angular_velocity_priority,
                cfg.angular_velocity_priority,
                cfg.angular_velocity_priority,
            ],
            dtype=float,
        )

    def to_dict(self) -> dict:
        return {
            "controller_type": "Efficient NMPC with Disturbance Aware Predictor",
            "main_strategy": "move blocking with nonlinear 6-DoF prediction and frozen disturbance blocks",
            "dt": self.cfg.dt,
            "jonswap_peak_period_Tp": self.cfg.Tp,
            "horizon_steps": self.cfg.horizon_steps,
            "prediction_horizon_s": self.prediction_horizon_s,
            "control_blocks": self.cfg.control_blocks,
            "block_size": self.block_size,
            "full_decision_variables": self.full_decision_variables,
            "blocked_decision_variables": self.blocked_decision_variables,
            "decision_variable_reduction_percent": self.reduction_percent,
            "state_names": self.state_names,
            "input_names": self.input_names,
            "disturbance_names": self.disturbance_names,
            "state_max": self.state_max.tolist(),
            "state_priorities": self.state_priorities.tolist(),
            "Q_diag": np.diag(self.Q).tolist(),
            "Qf_diag": np.diag(self.Qf).tolist(),
            "R_diag": np.diag(self.R).tolist(),
            "R_delta_diag": np.diag(self.R_delta).tolist(),
            "disturbance_bounds": {
                "max_expected_current_m_s": self.cfg.max_expected_current,
                "max_expected_wave_force_N": self.cfg.max_expected_wave_force,
                "max_expected_wave_moment_Nm": self.cfg.max_expected_wave_moment,
            },
            "input_bounds_N": {
                "min": self.cfg.thrust_min,
                "max": self.cfg.thrust_max,
            },
            "delta_input_bound_N_per_step": self.cfg.max_delta_thrust,
            "soft_constraints": {
                "roll_soft_limit_deg": self.cfg.roll_soft_limit_deg,
                "pitch_soft_limit_deg": self.cfg.pitch_soft_limit_deg,
                "attitude_soft_weight": self.cfg.attitude_soft_weight,
            },
            "solver": {
                "name": self.cfg.solver_name,
                "ftol": self.cfg.solver_ftol,
                "maxiter": self.cfg.solver_maxiter,
            },
            "implementation_note": (
                "The optimizer decides one 6-thruster command per block. "
                "Disturbances (nu_c and tau_wave) must be evaluated at time t and "
                "passed as parameter vectors to the NMPC solver, remaining constant "
                "or updated block by block over the horizon."
            ),
        }

    def print_report(self):
        np.set_printoptions(precision=6, suppress=True)

        print("\n" + "=" * 80)
        print("EFFICIENT NMPC DESIGN REPORT (DISTURBANCE AWARE)")
        print("=" * 80)

        print("\nHorizon & Environment")
        print(f"  dt: {self.cfg.dt:.3f} s")
        print(f"  N: {self.cfg.horizon_steps}")
        print(f"  prediction horizon: {self.prediction_horizon_s:.3f} s")
        print(f"  JONSWAP Tp (Updated): {self.cfg.Tp:.3f} s")
        print(f"  horizon/Tp: {self.prediction_horizon_s / self.cfg.Tp:.3f}")

        print("\nMove blocking")
        print(f"  control blocks: {self.cfg.control_blocks}")
        print(f"  block size: {self.block_size} steps")
        print(f"  full decision variables: {self.full_decision_variables}")
        print(f"  blocked decision variables: {self.blocked_decision_variables}")
        print(f"  reduction: {self.reduction_percent:.1f}%")

        print("\nDisturbance Handling Limits")
        print(f"  Max expected current (nu_c): {self.cfg.max_expected_current:.2f} m/s")
        print(f"  Max expected wave force (tau_w): {self.cfg.max_expected_wave_force:.2f} N")
        print(f"  Max expected wave moment (tau_w): {self.cfg.max_expected_wave_moment:.2f} Nm")

        print("\nQ diagonal")
        print(np.diag(self.Q))

        print("\nQf diagonal")
        print(np.diag(self.Qf))

        print("\nR diagonal")
        print(np.diag(self.R))

        print("\nR_delta diagonal")
        print(np.diag(self.R_delta))

        print("\nSolver")
        print(f"  method: {self.cfg.solver_name}")
        print(f"  ftol: {self.cfg.solver_ftol}")
        print(f"  maxiter: {self.cfg.solver_maxiter}")

        print("\nExpected computational effect")
        print(
            f"  The optimization vector is reduced from "
            f"{self.full_decision_variables} to "
            f"{self.blocked_decision_variables} variables."
        )

        print("=" * 80 + "\n")

    def save_json(self, filename: str = "efficient_nmpc_design_output.json"):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=4)

        print(f"Saved design data to: {filename}")


def main():
    cfg = EfficientNMPCDesignConfig(
        dt=0.1,
        Tp=12.0,  # Sincronizado com os 12 segundos do novo modelo JONSWAP

        # Efficient setting
        horizon_steps=10,
        control_blocks=5,

        # Admissible errors
        max_x_error=0.30,
        max_y_error=0.30,
        max_z_error=0.25,

        max_roll_error_deg=12.0,
        max_pitch_error_deg=12.0,
        max_yaw_error_deg=15.0,

        max_u_error=0.35,
        max_v_error=0.35,
        max_w_error=0.25,

        max_p_error_deg_s=25.0,
        max_q_error_deg_s=25.0,
        max_r_error_deg_s=30.0,

        position_priority=5.0,
        attitude_priority=1.5,
        velocity_priority=0.8,
        angular_velocity_priority=0.3,

        terminal_multiplier=8.0,

        thrust_min=-40.0,
        thrust_max=40.0,
        max_delta_thrust=12.0,

        input_priority=0.08,
        input_rate_priority=0.35,

        # Casamento com os limites superiores do modelo de ondas
        max_expected_current=0.7,
        max_expected_wave_force=25.0,
        max_expected_wave_moment=8.0,

        solver_name="SLSQP",
        solver_ftol=3e-3,
        solver_maxiter=12,
    )

    design = EfficientNMPCDesign(cfg)
    design.print_report()
    design.save_json()


if __name__ == "__main__":
    main()