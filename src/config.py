"""Load the central project configuration."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    """Read the YAML config and return it as a dict."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """Resolve a path from the config against the project root."""
    return PROJECT_ROOT / relative
