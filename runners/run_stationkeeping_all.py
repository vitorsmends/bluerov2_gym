import subprocess
import sys


def run(script):
    print(f"\n[INFO] Running {script}")
    subprocess.run([sys.executable, script], check=True)


def main():
    run("runners/run_stationkeeping_pid.py")
    run("runners/run_stationkeeping_mpc.py")
    run("runners/run_stationkeeping_ppo.py")


if __name__ == "__main__":
    main()
