from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config" / "settings.yaml"


def load_config() -> dict:
    """Load config/settings.yaml and return as dict."""
    with open(_CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)
