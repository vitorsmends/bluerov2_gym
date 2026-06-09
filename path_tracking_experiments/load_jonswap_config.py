from pathlib import Path

import yaml


def load_jonswap_config(
    yaml_file: str = "jonswap_config.yaml",
    scenario: str | None = None,
):
    base_dir = Path(__file__).resolve().parent
    yaml_path = Path(yaml_file)

    if not yaml_path.is_absolute():
        yaml_path = base_dir / yaml_path

    if not yaml_path.exists():
        raise FileNotFoundError(f"JONSWAP config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if scenario is None:
        scenario = config["default_scenario"]

    if scenario not in config["scenarios"]:
        available = ", ".join(config["scenarios"].keys())
        raise ValueError(
            f"Unknown JONSWAP scenario '{scenario}'. "
            f"Available scenarios: {available}"
        )

    params = config["scenarios"][scenario].copy()

    if "wave_dir" in params:
        params["wave_dir"] = tuple(params["wave_dir"])

    return params