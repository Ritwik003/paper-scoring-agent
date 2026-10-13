"""Load settings from config/settings.yaml into an easy-to-use object."""
from pathlib import Path
import yaml

# The project root is two folders up from this file.
ROOT = Path(__file__).resolve().parents[2]
SETTINGS_FILE = ROOT / "config" / "settings.yaml"


def load_settings() -> dict:
    """Read settings.yaml and return it as a dictionary."""
    with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_paths(settings: dict) -> dict:
    """Turn the relative paths in settings into full folder locations."""
    p = settings["paths"]
    paths = {
        "pdf_dir": ROOT / p["pdf_dir"],
        "metadata_dir": ROOT / p["metadata_dir"],
    }
    # Newer, optional paths (present once the scoring layer is configured).
    if "scores_dir" in p:
        paths["scores_dir"] = ROOT / p["scores_dir"]
    if "picks_dir" in p:
        paths["picks_dir"] = ROOT / p["picks_dir"]
    return paths