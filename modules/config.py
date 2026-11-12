"""Settings, with defaults here, overridable from a .env file in the repo root
or from real environment variables (e.g. docker-compose `env_file:`).

Precedence: command-line flag > environment variable > .env > default.
An empty value (e.g. `GCS_MAGIC=`) means "use the default".
See .env.example for every setting.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

## Doesn't override variables already set in the environment.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _get(name: str, default, cast=str):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as e:
        raise SystemExit(f"bad value {name}={raw!r} (from environment or .env): {e}")


def _int(raw: str) -> int:
    return int(raw, 0)  # accepts 0x5053 as well as 20563


def _bool(raw: str) -> bool:
    v = raw.lower()
    if v in ("1", "true", "yes", "on"):
        return True
    if v in ("0", "false", "no", "off"):
        return False
    raise ValueError("expected true/false")


## ---- firmware / wire protocol ----
PORT: int = _get("GCS_PORT", 5005, _int)
MAGIC: int | None = _get("GCS_MAGIC", None, _int)  # None skips the check
VERSION: int | None = _get("GCS_VERSION", None, _int)  # None skips the check
CAM_LEFT: int = _get("GCS_CAM_LEFT", 0, _int)
CAM_RIGHT: int = _get("GCS_CAM_RIGHT", 1, _int)

## ---- network ----
HOST: str = _get("GCS_HOST", "0.0.0.0")
SOURCE: str | None = _get("GCS_SOURCE", None)  # Pico IP; None accepts any sender
RCVBUF: int = _get("GCS_RCVBUF", 4 << 20, _int)

## ---- frame assembly ----
FRAME_TIMEOUT_S: float = _get("GCS_FRAME_TIMEOUT_S", 0.5, float)
MAX_SKEW_US: int = _get("GCS_MAX_SKEW_US", 1000, _int)

## ---- calibration / rectification ----
DATA_DIR: Path = _get("GCS_DATA_DIR", Path("data"), Path)
PATTERN: str = _get("GCS_PATTERN", "8x5")
SQUARE_MM: float | None = _get("GCS_SQUARE_MM", None, float)
ALPHA: float = _get("GCS_ALPHA", 0.0, float)
FIX_K3: bool = _get("GCS_FIX_K3", False, _bool)  # hold distortion k3 at 0 in `calibrate solve`

## ---- depth (stereo matching) ----
DEPTH_ALGO: str = _get("GCS_DEPTH_ALGO", "sgbm")  # sgbm | census-sgm | elas
NUM_DISPARITIES: int = _get("GCS_NUM_DISPARITIES", 64, _int)  # search range 0..N-1 px
SGBM_BLOCK_SIZE: int = _get("GCS_SGBM_BLOCK_SIZE", 5, _int)
CENSUS_WINDOW: str = _get("GCS_CENSUS_WINDOW", "9x7")
SGM_P1: int = _get("GCS_SGM_P1", 10, _int)
SGM_P2: int = _get("GCS_SGM_P2", 120, _int)
SGM_PATHS: int = _get("GCS_SGM_PATHS", 8, _int)

if CAM_LEFT == CAM_RIGHT:
    raise SystemExit(f"GCS_CAM_LEFT and GCS_CAM_RIGHT are both {CAM_LEFT}")

## ---- obstacles (polytope3d on live depth, `modules.depth --obstacles`) ----
OBSTACLE_ALGO: str = _get("GCS_OBSTACLE_ALGO", "aabb")  # aabb|obb|kdop14|kdop26|hull|hull_simple|split_obb
OBSTACLE_MAX_RANGE: float = _get("GCS_OBSTACLE_MAX_RANGE", 4.0, float)  # metres; stereo error grows with Z²
