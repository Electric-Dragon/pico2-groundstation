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

if CAM_LEFT == CAM_RIGHT:
    raise SystemExit(f"GCS_CAM_LEFT and GCS_CAM_RIGHT are both {CAM_LEFT}")
