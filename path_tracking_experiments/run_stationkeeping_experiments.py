from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import yaml

from env_utils import make_env, obs_to_state, tracking_errors, wrap_angle

from pid_controller import PIDController
from smc_controller import SMCController
from nmpc_controller import NMPCController
from ppo_controller import PPOController


OUTPUT_DIR = Path("results/stationkeeping")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DT = 0.1
STEPS = 1000
REPETITIONS = 10

CONTROLLERS = ["pid", "smc", "nmpc", "ppo"]

CONFIG_PATH = Path(__file__).resolve().parent / "jonswap_config.yaml"


# ============================================================
# Station-keeping targets
# ============================================================

STATIONKEEPING_TARGETS = [

    # ==========================================
    # Região central
    # ==========================================
    {
        "id": 1,
        "name": "center_forward",
        "pos": [0.0, 0.0, -0.8],
        "yaw_deg": 0.0,
    },

    # ==========================================
    # Frente
    # ==========================================
    {
        "id": 2,
        "name": "front",
        "pos": [2.0, 0.0, -0.8],
        "yaw_deg": 0.0,
    },

    {
        "id": 3,
        "name": "front_right",
        "pos": [2.0, 2.0, -0.8],
        "yaw_deg": 45.0,
    },

    {
        "id": 4,
        "name": "front_left",
        "pos": [2.0, -2.0, -0.8],
        "yaw_deg": -45.0,
    },

    # ==========================================
    # Trás
    # ==========================================
    {
        "id": 5,
        "name": "rear",
        "pos": [-2.0, 0.0, -0.8],
        "yaw_deg": 180.0,
    },

    {
        "id": 6,
        "name": "rear_right",
        "pos": [-2.0, 2.0, -0.8],
        "yaw_deg": 135.0,
    },

    {
        "id": 7,
        "name": "rear_left",
        "pos": [-2.0, -2.0, -0.8],
        "yaw_deg": -135.0,
    },

    # ==========================================
    # Profundidade rasa
    # ==========================================
    {
        "id": 8,
        "name": "shallow",
        "pos": [1.5, 0.0, -0.4],
        "yaw_deg": 90.0,
    },

    # ==========================================
    # Profundidade média
    # ==========================================
    {
        "id": 9,
        "name": "mid_depth",
        "pos": [0.0, 2.5, -1.5],
        "yaw_deg": 90.0,
    },

    # ==========================================
    # Profundidade alta
    # ==========================================
    {
        "id": 10,
        "name": "deep",
        "pos": [0.0, -2.5, -3.0],
        "yaw_deg": -90.0,
    },

    # ==========================================
    # Inspeção lateral
    # ==========================================
    {
        "id": 11,
        "name": "side_inspection",
        "pos": [3.0, 0.0, -1.2],
        "yaw_deg": 90.0,
    },

    # ==========================================
    # Inspeção de estrutura
    # ==========================================
    {
        "id": 12,
        "name": "structure_view",
        "pos": [3.0, 3.0, -2.0],
        "yaw_deg": 225.0,
    },

    # ==========================================
    # Distância máxima
    # ==========================================
    {
        "id": 13,
        "name": "far_corner",
        "pos": [4.0, -4.0, -2.0],
        "yaw_deg": 315.0,
    },

    # ==========================================
    # Distância máxima oposta
    # ==========================================
    {
        "id": 14,
        "name": "far_corner_opposite",
        "pos": [-4.0, 4.0, -2.0],
        "yaw_deg": 135.0,
    },

    # ==========================================
    # Caso extremo
    # ==========================================
    {
        "id": 15,
        "name": "deep_far",
        "pos": [4.0, 4.0, -4.0],
        "yaw_deg": 45.0,
    },
]


# ============================================================
# Approximate T200 power model
# ============================================================

T200_MAX_THRUST_N = 51.5
T200_MAX_POWER_W = 390.0
THRUST_LIMIT = 40.0


def estimate_thruster_power_watts(thrust):
    thrust = np.asarray(thrust, dtype=float).reshape(6)
    ratio = np.clip(np.abs(thrust) / T200_MAX_THRUST_N, 0.0, 1.0)
    return T200_MAX_POWER_W * ratio**1.5


# ============================================================
# Reference object
# ============================================================

class StationKeepingReference:
    def __init__(self, position, yaw_rad):
        self.position = np.asarray(position, dtype=float).reshape(3)
        self.yaw_rad = float(yaw_rad)

    def get_reference(self, t):
        ref = np.zeros(12, dtype=float)
        ref[0:3] = self.position
        ref[3] = 0.0
        ref[4] = 0.0
        ref[5] = self.yaw_rad
        ref[6:12] = 0.0
        return ref


# ============================================================
# Config
# ============================================================

def load_jonswap_scenarios():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    scenarios = config["scenarios"]

    clean = {}
    for name, params in scenarios.items():
        p = params.copy()
        if "wave_dir" in p:
            p["wave_dir"] = tuple(p["wave_dir"])
        clean[name] = p

    return clean


def reset_env_with_jonswap(env, jonswap_params, seed=None):
    params = jonswap_params.copy()

    if seed is not None:
        params["seed"] = int(seed)

    try:
        obs, _ = env.reset(
            seed=seed,
            options={"jonswap_params": params},
        )
    except TypeError:
        obs, _ = env.reset()
        env.unwrapped.dynamics.reset(jonswap_params=params)

    return obs


def set_env_state(env, position, yaw_rad, offset=True):
    base_env = env.unwrapped

    position = np.asarray(position, dtype=float).reshape(3)

    if offset:
        init_pos = position + np.array([0.20, -0.15, 0.08], dtype=float)
        init_yaw = wrap_angle(yaw_rad + np.deg2rad(10.0))
    else:
        init_pos = position.copy()
        init_yaw = yaw_rad

    base_env.state = {
        "x": float(init_pos[0]),
        "y": float(init_pos[1]),
        "z": float(init_pos[2]),
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": float(init_yaw),
        "u": 0.0,
        "v": 0.0,
        "w": 0.0,
        "p": 0.0,
        "q": 0.0,
        "r": 0.0,
    }

    return base_env._get_obs()


def make_controller(controller_name, dynamics, reference):
    name = controller_name.lower()

    if name == "pid":
        return PIDController(dynamics=dynamics, dt=DT)

    if name == "smc":
        return SMCController(dynamics=dynamics, dt=DT)

    if name == "nmpc":
        return NMPCController(
            trajectory=reference,
            dynamics=dynamics,
            dt=DT,
            horizon=10,
            control_blocks=5,
        )

    if name == "ppo":
        return PPOController()

    raise ValueError(f"Unknown controller: {controller_name}")


def get_controller_metrics(controller):
    metrics = getattr(controller, "last_metrics", {})

    return [
        metrics.get("controller_wall_time_s", np.nan),
        metrics.get("controller_cpu_time_s", np.nan),
        metrics.get("controller_frequency_hz", np.nan),
        metrics.get("controller_prepare_time_s", np.nan),
        metrics.get("controller_solver_time_s", np.nan),
        metrics.get("controller_post_time_s", np.nan),
        metrics.get("controller_success", np.nan),
    ]


# ============================================================
# CSV
# ============================================================

HEADER = [
    "controller",
    "sea_scenario",
    "target_id",
    "repetition_id",
    "seed",
    "time",

    "Hs",
    "Tp",
    "gamma",
    "wave_dir_x",
    "wave_dir_y",
    "scale",
    "max_current",
    "noise_std",

    "x", "y", "z",
    "roll", "pitch", "yaw",
    "u", "v", "w",
    "p", "q", "r",

    "x_ref", "y_ref", "z_ref",
    "roll_ref", "pitch_ref", "yaw_ref",

    "position_error_m",
    "tracking_error_m",
    "velocity_error",
    "yaw_error",
    "reward",

    "T1", "T2", "T3", "T4", "T5", "T6",

    "control_effort",
    "control_effort_normalized",

    "thruster_1_power_W",
    "thruster_2_power_W",
    "thruster_3_power_W",
    "thruster_4_power_W",
    "thruster_5_power_W",
    "thruster_6_power_W",
    "total_power_W",
    "step_energy_J",
    "cumulative_energy_J",

    "controller_wall_time_s",
    "controller_cpu_time_s",
    "controller_frequency_hz",
    "controller_prepare_time_s",
    "controller_solver_time_s",
    "controller_post_time_s",
    "controller_success",
]


def run_case(controller_name, sea_name, sea_params, target, repetition_id):
    seed = int(sea_params.get("seed", 42)) + repetition_id

    position = np.asarray(target["pos"], dtype=float)
    yaw_rad = np.deg2rad(float(target["yaw_deg"]))

    reference = StationKeepingReference(
        position=position,
        yaw_rad=yaw_rad,
    )

    env = make_env(render_mode=None)
    obs = reset_env_with_jonswap(env, sea_params, seed=seed)
    obs = set_env_state(env, position, yaw_rad, offset=True)

    dynamics = env.unwrapped.dynamics
    controller = make_controller(controller_name, dynamics, reference)

    if hasattr(controller, "reset"):
        controller.reset()

    rows = []
    cumulative_energy = 0.0

    print(
        f"\n[INFO] controller={controller_name.upper()} | "
        f"sea={sea_name} | target={target['id']} | "
        f"rep={repetition_id + 1}/{REPETITIONS}"
    )

    for k in range(STEPS):
        t = k * DT

        state = obs_to_state(obs)
        ref = reference.get_reference(t)
        errors = tracking_errors(state, ref)

        action = controller.get_action(
            obs=obs,
            state=state,
            reference=ref,
            t=t,
        )

        metrics = get_controller_metrics(controller)

        action = np.asarray(action, dtype=np.float32).reshape(6)
        action = np.clip(action, -THRUST_LIMIT, THRUST_LIMIT)

        thruster_power = estimate_thruster_power_watts(action)
        total_power = float(np.sum(thruster_power))
        step_energy = total_power * DT
        cumulative_energy += step_energy

        obs, reward, terminated, truncated, info = env.step(action)

        pos_error = float(np.linalg.norm(state[0:3] - ref[0:3]))

        rows.append([
            controller_name,
            sea_name,
            target["id"],
            repetition_id,
            seed,
            t,

            sea_params.get("Hs", np.nan),
            sea_params.get("Tp", np.nan),
            sea_params.get("gamma", np.nan),
            sea_params.get("wave_dir", (np.nan, np.nan))[0],
            sea_params.get("wave_dir", (np.nan, np.nan))[1],
            sea_params.get("scale", np.nan),
            sea_params.get("max_current", np.nan),
            sea_params.get("noise_std", np.nan),

            state[0], state[1], state[2],
            state[3], state[4], state[5],
            state[6], state[7], state[8],
            state[9], state[10], state[11],

            ref[0], ref[1], ref[2],
            ref[3], ref[4], ref[5],

            pos_error,
            errors["tracking_error_m"],
            errors["velocity_error"],
            errors["yaw_error"],
            reward,

            action[0], action[1], action[2],
            action[3], action[4], action[5],

            float(np.sum(action**2)),
            float(np.mean((action / THRUST_LIMIT) ** 2)),

            thruster_power[0],
            thruster_power[1],
            thruster_power[2],
            thruster_power[3],
            thruster_power[4],
            thruster_power[5],
            total_power,
            step_energy,
            cumulative_energy,

            *metrics,
        ])

        if k % 100 == 0:
            print(
                f"[{controller_name.upper()} | {sea_name} | target={target['id']}] "
                f"step={k:04d}/{STEPS} | "
                f"t={t:5.1f}s | "
                f"pos_error={pos_error:.3f} m | "
                f"yaw_error={errors['yaw_error']:.3f} rad"
            )

        if terminated or truncated:
            print(
                f"[INFO] Episode finished early | "
                f"t={t:.1f}s | controller={controller_name} | "
                f"sea={sea_name} | target={target['id']}"
            )
            break

    env.close()
    return rows


def run_controller(controller_name):
    sea_scenarios = load_jonswap_scenarios()

    output_csv = OUTPUT_DIR / f"{controller_name}_stationkeeping.csv"

    all_rows = []
    start = time.time()

    for sea_name, sea_params in sea_scenarios.items():
        for target in STATIONKEEPING_TARGETS:
            for rep in range(REPETITIONS):
                rows = run_case(
                    controller_name=controller_name,
                    sea_name=sea_name,
                    sea_params=sea_params,
                    target=target,
                    repetition_id=rep,
                )

                all_rows.extend(rows)

    with output_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(all_rows)

    elapsed = time.time() - start

    print(f"\n[OK] Saved: {output_csv}")
    print(f"[INFO] Controller: {controller_name}")
    print(f"[INFO] Rows: {len(all_rows)}")
    print(f"[INFO] Elapsed time: {elapsed:.2f} s")


def main():
    for controller_name in CONTROLLERS:
        run_controller(controller_name)

    print("\n[OK] All station-keeping experiments finished.")


if __name__ == "__main__":
    main()