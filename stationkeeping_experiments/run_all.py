"""Run all station-keeping experiments."""

import subprocess
import sys


SCRIPTS = [
    "run_pid.py",
    "run_mpc.py",
    "run_ppo.py",
]


def main():
    for script in SCRIPTS:
        print(f"\n[INFO] Running {script}")
        subprocess.run([sys.executable, script], check=True)


if __name__ == "__main__":
    main()
