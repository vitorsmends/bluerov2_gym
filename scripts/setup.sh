#!/usr/bin/env bash
# Run with bash, then source the activation command printed at the end.
set -euo pipefail

bluerov_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
bluerov_venv="${BLUEROV_VENV:-.venv-ppo-reference}"
bluerov_python="${BLUEROV_PYTHON:-3.10}"
bluerov_extras='[control,optimize]'
bluerov_dry_run=false

usage() {
    cat <<'EOF'
Usage: bash scripts/setup.sh [--minimal] [--dry-run]

Creates/reuses .venv-ppo-reference and installs the project with uv.
Includes NMPC and optimization dependencies by default.
  --minimal   Install only training, PID/SMC, evaluation and plotting dependencies.
  --dry-run   Resolve dependencies without installing them (may create the venv).

BLUEROV_VENV selects another environment directory.
BLUEROV_PYTHON selects Python for a new environment (default: 3.10).
EOF
}

for bluerov_arg in "$@"; do
    case "$bluerov_arg" in
        --minimal) bluerov_extras='' ;;
        --dry-run) bluerov_dry_run=true ;;
        -h|--help) usage; exit 0 ;;
        *) usage >&2; exit 2 ;;
    esac
done

case "$bluerov_venv" in
    /*) ;;
    *) bluerov_venv="$bluerov_root/$bluerov_venv" ;;
esac

if command -v uv >/dev/null 2>&1; then
    bluerov_uv="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
    bluerov_uv="$HOME/.local/bin/uv"
else
    command -v curl >/dev/null 2>&1 || { echo 'Install curl before running this setup.' >&2; exit 1; }
    echo 'Installing uv with the official Astral installer...'
    bluerov_installer="$(mktemp)"
    trap 'rm -f -- "$bluerov_installer"' EXIT
    curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$bluerov_installer"
    UV_UNMANAGED_INSTALL="$HOME/.local/bin" sh "$bluerov_installer"
    bluerov_uv="$HOME/.local/bin/uv"
fi

if [[ ! -x "$bluerov_venv/bin/python" ]]; then
    if [[ -e "$bluerov_venv" ]]; then
        echo "Target exists but has no working Python: $bluerov_venv. Select a new BLUEROV_VENV." >&2
        exit 1
    fi
    "$bluerov_uv" venv --python "$bluerov_python" "$bluerov_venv"
fi

bluerov_install_args=(pip install --python "$bluerov_venv/bin/python"
    --constraint "$bluerov_root/configs/setup/ppo_reference_constraints.txt"
    --editable "${bluerov_root}${bluerov_extras}")
if [[ "$bluerov_dry_run" == true ]]; then
    bluerov_install_args+=(--dry-run)
fi
"$bluerov_uv" "${bluerov_install_args[@]}"

if [[ "$bluerov_dry_run" == true ]]; then
    echo 'Dependency resolution finished; no packages were installed.'
    exit 0
fi

"$bluerov_uv" pip check --python "$bluerov_venv/bin/python"
"$bluerov_venv/bin/python" -m bluerov2_gym.workflows --help >/dev/null
echo
echo 'Setup completed. Activate the environment in this terminal:'
printf 'source %q\n' "$bluerov_venv/bin/activate"
echo
echo 'Standard training (curriculum, corrected dynamics):'
echo 'bluerov train --config configs/training/curriculum.yaml --output artifacts/training/com_curriculo'
