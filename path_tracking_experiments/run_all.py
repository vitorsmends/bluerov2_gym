"""Run all path-tracking experiments."""

from __future__ import annotations

import sys
import time
import subprocess
from pathlib import Path


CONTROLLERS = [
    ("PID", "run_pid.py"),
    ("PPO", "run_ppo.py"),
    ("NMPC", "run_nmpc.py"),
    ("SMC", "run_smc.py"),
]


def run(script_name: str):
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / script_name

    print("\n" + "=" * 80)
    print(f"[INFO] Running {script_name}")
    print("=" * 80)

    start = time.perf_counter()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=False,
    )

    elapsed = time.perf_counter() - start

    return result.returncode, elapsed


def main():
    summary = []

    total_start = time.perf_counter()

    for controller_name, script_name in CONTROLLERS:

        try:
            return_code, elapsed = run(script_name)

            status = "OK" if return_code == 0 else "FAILED"

            summary.append(
                {
                    "controller": controller_name,
                    "status": status,
                    "elapsed_s": elapsed,
                }
            )

        except Exception as exc:
            summary.append(
                {
                    "controller": controller_name,
                    "status": f"EXCEPTION ({exc})",
                    "elapsed_s": np.nan,
                }
            )

    total_elapsed = time.perf_counter() - total_start

    print("\n")
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    for row in summary:
        print(
            f"{row['controller']:>6s} | "
            f"{row['status']:<10s} | "
            f"{row['elapsed_s']:8.2f} s"
        )

    print("-" * 80)
    print(f"TOTAL WALL TIME: {total_elapsed:.2f} s")
    print("=" * 80)


if __name__ == "__main__":
    main()