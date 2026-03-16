import importlib
import sys
from pathlib import Path


def load_remoteclip_classes(remoteclip_repo_path: str | None = None):
    """Load RemoteVisionTower and TowerConfig from a configurable repo path."""
    if remoteclip_repo_path:
        repo_path = str(Path(remoteclip_repo_path).expanduser().resolve())
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)

    try:
        module = importlib.import_module("RemoteVisionTower")
    except ImportError as exc:
        raise ImportError(
            "Could not import RemoteVisionTower. Set model.remoteclip_repo_path "
            "to the RemoteClip source directory or install the package."
        ) from exc

    return module.RemoteVisionTower, module.TowerConfig
