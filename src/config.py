"""Configuration loading and shared helpers (paths, logging, seeds)."""

from __future__ import annotations

import logging
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load and cache the project configuration from ``config.yaml``.

    Args:
        path: Location of the YAML config file.

    Returns:
        The parsed configuration as a nested dictionary.
    """
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str, create: bool = True) -> Path:
    """Resolve a path from the ``paths`` section of the config, relative to the project root.

    Args:
        key: Key under ``paths`` in config.yaml (e.g. ``"raw"``).
        create: If True and the path looks like a directory, create it.

    Returns:
        Absolute path.
    """
    path = PROJECT_ROOT / load_config()["paths"][key]
    if create and not path.suffix:
        path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int | None = None) -> int:
    """Seed Python and NumPy RNGs for reproducibility.

    Args:
        seed: Seed to use; defaults to ``project.random_seed`` in config.

    Returns:
        The seed that was applied.
    """
    seed = load_config()["project"]["random_seed"] if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to the console and to ``logs/<name>.log``.

    Args:
        name: Logger name, also used for the log file name.

    Returns:
        Configured logger.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    file_handler = logging.FileHandler(get_path("logs") / f"{name.split('.')[-1]}.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger
