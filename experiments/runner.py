from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from .controllers.factory import ControllerFactory
from .config_loader import load_yaml
from .env_utils import make_env, obs_to_state, tracking_errors, wrap_angle
from .references import StationKeepingReference
from .trajectories import create_trajectory

METRIC_COLUMNS = [
    "controller_wall_time_s", "controller_cpu_time_s", "controller_frequency_hz",
    "controller_prepare_time_s", "controller_solver_time_s", "controller_post_time_s",
    "controller_success",
]

STATIONKEEPING_HEADER = [
    "controller", "sea_scenario", "target_id", "repetition_id", "seed", "time",
    "Hs", "Tp", "gamma", "wave_dir_x", "wave_dir_y", "scale", "max_current", "noise_std",
    "x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r",
    "x_ref", "y_ref", "z_ref", "roll_ref", "pitch_ref", "yaw_ref",
    "position_error_m", "tracking_error_m", "velocity_error", "yaw_error", "reward",
    "T1", "T2", "T3", "T4", "T5", "T6",
    "control_effort", "control_effort_normalized",
    "thruster_1_power_W", "thruster_2_power_W", "thruster_3_power_W",
    "thruster_4_power_W", "thruster_5_power_W", "thruster_6_power_W",
    "total_power_W", "step_energy_J", "cumulative_energy_J",
    *METRIC_COLUMNS,
]

TRAJECTORY_HEADER = [
    "trajectory", "controller", "time",
    "x", "y", "z", "roll", "pitch", "yaw", "u", "v", "w", "p", "q", "r",
    "x_ref", "y_ref", "z_ref", "roll_ref", "pitch_ref", "yaw_ref",
    "u_ref", "v_ref", "w_ref", "p_ref", "q_ref", "r_ref",
    "tracking_error_m", "velocity_error", "yaw_error", "reward",
    "T1", "T2", "T3", "T4", "T5", "T6",
    "control_effort", "control_effort_normalized",
]


def _metrics(controller: Any) -> list[float]:
    values = getattr(controller, "last_metrics", {})
    return [values.get(name, np.nan) for name in METRIC_COLUMNS]


def _reset_with_jonswap(env, params: dict | None, seed: int | None = None):
    if params is None:
        obs, _ = env.reset(seed=seed)
        return obs
    current = dict(params)
    if seed is not None:
        current["seed"] = int(seed)
    try:
        obs, _ = env.reset(seed=seed, options={"jonswap_params": current})
    except TypeError:
        obs, _ = env.reset()
        env.unwrapped.dynamics.reset(jonswap_params=current)
    return obs


def _set_stationkeeping_state(env, position, yaw_rad, offset=True):
    position = np.asarray(position, dtype=float).reshape(3)
    if offset:
        position = position + np.array([0.20, -0.15, 0.08], dtype=float)
        yaw_rad = wrap_angle(yaw_rad + np.deg2rad(10.0))
    env.unwrapped.state = {
        "x": float(position[0]), "y": float(position[1]), "z": float(position[2]),
        "roll": 0.0, "pitch": 0.0, "yaw": float(yaw_rad),
        "u": 0.0, "v": 0.0, "w": 0.0, "p": 0.0, "q": 0.0, "r": 0.0,
    }
    return env.unwrapped._get_obs()


def _estimate_thruster_power(thrust, max_thrust=51.5, max_power=390.0):
    ratio = np.clip(np.abs(np.asarray(thrust, dtype=float).reshape(6)) / max_thrust, 0.0, 1.0)
    return max_power * ratio**1.5


class ExperimentRunner:
    def __init__(self, config: dict[str, Any]):
        self.config = config
        disturbance = self.config.setdefault("disturbance", {})
        config_path = disturbance.get("config_path")
        if config_path:
            external = load_yaml(config_path)
            disturbance.setdefault("default_scenario", external.get("default_scenario"))
            disturbance.setdefault("scenarios", external.get("scenarios", {}))

    def run(self) -> None:
        mode = self.config["experiment"]["mode"].lower()
        if mode == "stationkeeping":
            self.run_stationkeeping()
        elif mode in {"trajectory", "trajectory_suite", "path_tracking"}:
            self.run_trajectory_suite()
        else:
            raise ValueError(f"Unknown experiment mode: {mode}")

    def run_stationkeeping(self) -> None:
        cfg = self.config
        exp = cfg["experiment"]
        dt = float(exp.get("dt", 0.1))
        steps = int(exp.get("steps", 500))
        repetitions = int(exp.get("repetitions", 5))
        thrust_limit = float(exp.get("thrust_limit", 40.0))
        output_dir = Path(cfg["output"]["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)

        for controller_cfg in cfg["controllers"]:
            if not controller_cfg.get("enabled", True):
                continue
            name = controller_cfg["name"].lower()
            rows = []
            started = time.time()
            for sea_name, sea_params in cfg["disturbance"]["scenarios"].items():
                sea_params = dict(sea_params)
                if "wave_dir" in sea_params:
                    sea_params["wave_dir"] = tuple(sea_params["wave_dir"])
                for target in cfg["stationkeeping"]["targets"]:
                    for rep in range(repetitions):
                        seed = int(sea_params.get("seed", 42)) + rep
                        position = np.asarray(target["pos"], dtype=float)
                        yaw_rad = np.deg2rad(float(target["yaw_deg"]))
                        reference = StationKeepingReference(position, yaw_rad)
                        env = make_env(render_mode=exp.get("render_mode"))
                        obs = _reset_with_jonswap(env, sea_params, seed)
                        obs = _set_stationkeeping_state(env, position, yaw_rad, offset=True)
                        controller = ControllerFactory.create(
                            name, dynamics=env.unwrapped.dynamics,
                            reference_provider=reference, dt=dt, config=controller_cfg,
                        )
                        controller.reset()
                        cumulative_energy = 0.0
                        for k in range(steps):
                            t = k * dt
                            state = obs_to_state(obs)
                            ref = reference.get_reference(t)
                            errors = tracking_errors(state, ref)
                            action = np.asarray(controller.get_action(obs=obs, state=state, reference=ref, t=t), dtype=np.float32).reshape(6)
                            action = np.clip(action, -thrust_limit, thrust_limit)
                            power = _estimate_thruster_power(action)
                            total_power = float(np.sum(power))
                            step_energy = total_power * dt
                            cumulative_energy += step_energy
                            obs, reward, terminated, truncated, _ = env.step(action)
                            pos_error = float(np.linalg.norm(state[0:3] - ref[0:3]))
                            rows.append([
                                name, sea_name, target["id"], rep, seed, t,
                                sea_params.get("Hs", np.nan), sea_params.get("Tp", np.nan), sea_params.get("gamma", np.nan),
                                sea_params.get("wave_dir", (np.nan, np.nan))[0], sea_params.get("wave_dir", (np.nan, np.nan))[1],
                                sea_params.get("scale", np.nan), sea_params.get("max_current", np.nan), sea_params.get("noise_std", np.nan),
                                *state.tolist(), *ref[0:6].tolist(), pos_error,
                                errors["tracking_error_m"], errors["velocity_error"], errors["yaw_error"], reward,
                                *action.tolist(), float(np.sum(action**2)), float(np.mean((action / thrust_limit)**2)),
                                *power.tolist(), total_power, step_energy, cumulative_energy, *_metrics(controller),
                            ])
                            if terminated or truncated:
                                break
                        env.close()
            output = output_dir / f"{name}_stationkeeping.csv"
            with output.open("w", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(STATIONKEEPING_HEADER); writer.writerows(rows)
            print(f"[OK] Saved: {output} | rows={len(rows)} | elapsed={time.time()-started:.2f}s")

    def run_trajectory_suite(self) -> None:
        cfg = self.config
        exp = cfg["experiment"]
        dt = float(exp.get("dt", 0.1))
        steps = int(exp.get("steps", 1000))
        thrust_limit = float(exp.get("thrust_limit", 40.0))
        output_dir = Path(cfg["output"]["directory"])
        output_dir.mkdir(parents=True, exist_ok=True)
        scenario_name = cfg["disturbance"].get("default_scenario")
        jonswap = cfg["disturbance"]["scenarios"].get(scenario_name) if scenario_name else None

        trajectories = [create_trajectory(item["name"], item.get("params")) for item in cfg["trajectories"] if item.get("enabled", True)]
        for controller_cfg in cfg["controllers"]:
            if not controller_cfg.get("enabled", True):
                continue
            name = controller_cfg["name"].lower()
            for trajectory in trajectories:
                env = make_env(render_mode=exp.get("render_mode"))
                if jonswap is not None:
                    env.unwrapped.dynamics.reset(jonswap_params=jonswap)
                controller = ControllerFactory.create(
                    name, dynamics=env.unwrapped.dynamics,
                    reference_provider=trajectory, dt=dt, config=controller_cfg,
                )
                obs = _reset_with_jonswap(env, jonswap)
                controller.reset()
                rows = []
                for k in range(steps):
                    t = k * dt
                    state = obs_to_state(obs)
                    reference = trajectory.get_reference(t)
                    errors = tracking_errors(state, reference)
                    action = np.asarray(controller.get_action(obs=obs, state=state, reference=reference, t=t), dtype=np.float32).reshape(-1)
                    action = np.clip(action, -thrust_limit, thrust_limit)
                    obs, reward, terminated, truncated, _ = env.step(action)
                    rows.append([
                        trajectory.name, name, t, *state.tolist(), *reference.tolist(),
                        errors["tracking_error_m"], errors["velocity_error"], errors["yaw_error"], reward,
                        *action.tolist(), float(np.sum(action**2)), float(np.mean((action / thrust_limit)**2)),
                    ])
                    if terminated or truncated:
                        break
                output = output_dir / f"{name}_{trajectory.name}.csv"
                with output.open("w", newline="") as stream:
                    writer = csv.writer(stream); writer.writerow(TRAJECTORY_HEADER); writer.writerows(rows)
                env.close()
                print(f"[OK] Saved: {output}")
