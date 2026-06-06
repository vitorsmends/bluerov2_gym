"""Run all path-tracking experiments."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run(script_name: str):
    script_dir = Path(__file__).resolve().parent
    script_path = script_dir / script_name

    print(f"\n[INFO] Running {script_path.name}")
    subprocess.run([sys.executable, str(script_path)], check=True)


def main():
    run("run_pid.py")
    run("run_ppo.py")
    run("run_nmpc.py")


if __name__ == "__main__":
    main()