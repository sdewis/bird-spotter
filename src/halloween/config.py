"""Configuration loader for the Halloween animatronic skull project."""

from pathlib import Path
import tomllib
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.toml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Load TOML configuration from disk."""
    cfg_path = path or DEFAULT_CONFIG_PATH
    with cfg_path.open("rb") as f:
        return tomllib.load(f)
