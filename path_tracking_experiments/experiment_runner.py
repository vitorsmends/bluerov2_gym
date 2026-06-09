"""Shared path-tracking experiment runner."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from env_utils import make_env, obs_to_state, tracking_errors


METRIC_COLUMNS = [
    "controller_wall_time_s",
    "controller_cpu_time_s",
    "controller_frequency_hz",
    "controller_prepare_time_s",
    "controller_solver_time_s",
    "controller_post_time_s",
    "controller_success",
]


def _get_controller_metrics(controller):
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


def _build_repetition_jonswap_params(jonswap_params, rep_seed):
    if jonswap_params is None:
        return None

    params = jonswap_params.copy()

    if rep_seed is not None:
        params["seed"] = rep_seed

    return params


def run_path_tracking_experiment(
    controller,
    trajectory,
    output_csv: str,
    steps: int = 1000,
    dt: float = 0.1,
    repetitions: int = 1,
    seed: int | None = 42,
    jonswap_params: dict | None = None,
    render_mode=None,
):
    """Run repeated path-tracking experiments and save a CSV log.

    The controller must implement:

        action = controller.get_action(obs, state, reference, t)

    where action is a six-element vector of direct thruster commands.
    """
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = make_env(render_mode=render_mode)

    rows = []
    start = time.time()

    for rep in range(repetitions):
        rep_seed = None if seed is None else int(seed + rep)
        rep_jonswap_params = _build_repetition_jonswap_params(
            jonswap_params,
            rep_seed,
        )

        reset_options = None
        if rep_jonswap_params is not None:
            reset_options = {
                "jonswap_params": rep_jonswap_params,
            }

        try:
            obs, _ = env.reset(
                seed=rep_seed,
                options=reset_options,
            )
        except TypeError:
            obs, _ = env.reset()

            if rep_jonswap_params is not None:
                env.unwrapped.dynamics.reset(
                    jonswap_params=rep_jonswap_params,
                )

        if hasattr(controller, "reset"):
            controller.reset()

        print(
            f"\n[INFO] Starting repetition {rep + 1}/{repetitions} "
            f"for {controller.__class__.__name__} | seed={rep_seed}"
        )

        for k in range(steps):
            t = k * dt

            state = obs_to_state(obs)
            reference = trajectory.get_reference(t)
            errors = tracking_errors(state, reference)

            action = controller.get_action(
                obs=obs,
                state=state,
                reference=reference,
                t=t,
            )

            controller_metrics = _get_controller_metrics(controller)

            action = np.asarray(action, dtype=np.float32).reshape(-1)
            action = np.clip(action, -40.0, 40.0)

            obs, reward, terminated, truncated, info = env.step(action)

            rows.append([
                rep,
                rep_seed,
                t,
                state[0], state[1], state[2],
                state[3], state[4], state[5],
                state[6], state[7], state[8],
                state[9], state[10], state[11],
                reference[0], reference[1], reference[2],
                reference[5],
                errors["tracking_error_m"],
                errors["velocity_error"],
                errors["yaw_error"],
                reward,
                action[0], action[1], action[2],
                action[3], action[4], action[5],
                *controller_metrics,
            ])

            if k % 50 == 0:
                print(
                    f"[{controller.__class__.__name__}] "
                    f"rep={rep + 1:02d}/{repetitions} | "
                    f"step={k:04d}/{steps} | t={t:5.1f}s | "
                    f"tracking_error={errors['tracking_error_m']:.3f} m | "
                    f"reward={reward:.3f}"
                )

            if terminated or truncated:
                print(
                    f"[INFO] Episode finished at t={t:.1f}s | "
                    f"rep={rep + 1}/{repetitions}"
                )
                break

    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            "repetition_id",
            "seed",
            "time",
            "x", "y", "z",
            "roll", "pitch", "yaw",
            "u", "v", "w",
            "p", "q", "r",
            "x_ref", "y_ref", "z_ref", "yaw_ref",
            "tracking_error_m", "velocity_error", "yaw_error",
            "reward",
            "T1", "T2", "T3", "T4", "T5", "T6",
            *METRIC_COLUMNS,
        ])

        writer.writerows(rows)

    env.close()

    elapsed = time.time() - start

    print(f"[INFO] Saved results to: {output_path}")
    print(f"[INFO] Repetitions: {repetitions}")
    print(f"[INFO] Elapsed time: {elapsed:.2f} s")