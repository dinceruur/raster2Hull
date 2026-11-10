from types import SimpleNamespace
import yaml
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def dict_to_ns(d):
    """Recursively converts a dictionary to SimpleNamespace."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_ns(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_ns(i) for i in d]
    return d


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at: {CONFIG_PATH}")

    try:
        with CONFIG_PATH.open("r") as f:
            cfg_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {CONFIG_PATH}: {e}")

    if cfg_dict is None:
        raise ValueError(f"Config file {CONFIG_PATH} is empty")

    return dict_to_ns(cfg_dict)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load it once so it can be imported globally
cfg = load_config()