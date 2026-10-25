"""Stereo rectification: remaps a left/right pair so matching points lie on the
same row, which is what stereo matchers expect.

Built from the calib.npz that `calibrate solve` writes. Maps are built lazily
for each frame size seen, and K is rescaled if the Pico sends a different
resolution than the one calibration used.

Run directly to view the rectified live stream:
  uv run python -m modules.rectify [--pattern 8x5] [--headless]
With --pattern, a board held in view gives a live row-mismatch reading, which
checks that the calibration still holds (e.g. after the rig was bumped).
"""
import argparse
import logging
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path

if __name__ == "__main__" and not __package__:
    ## Allow `python modules/rectify.py` as well as `python -m modules.rectify`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "modules"

import cv2 as cv
import numpy as np

from . import config
from .receiver import Receiver, no_display

log = logging.getLogger(__name__)

DEFAULT_CALIB = config.DATA_DIR / "calib.npz"
DEFAULT_SNAPSHOT = config.DATA_DIR / "rectified.png"
WINDOW = "rectified"


def load_calibration(path: Path = DEFAULT_CALIB) -> dict[str, np.ndarray]:
    with np.load(path) as f:
        return dict(f)


def scale_K(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] = (K[0, 2] + 0.5) * sx - 0.5  # OpenCV measures from pixel centres
    K[1, 1] *= sy
    K[1, 2] = (K[1, 2] + 0.5) * sy - 0.5
    return K


def draw_rows(gray: np.ndarray, step: int = 16) -> np.ndarray:
    """Colour copy with horizontal lines, for eyeballing row alignment."""
    view = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    for y in range(0, view.shape[0], step):
        cv.line(view, (0, y), (view.shape[1], y), (0, 255, 0), 1)
    return view


@dataclass(frozen=True)
class Rectification:
    """Everything needed to rectify frames of one size. P1/P2 are the rectified
    projection matrices, and Q reprojects disparity to 3D (cv.reprojectImageTo3D)."""
    size: tuple[int, int]
    K_l: np.ndarray
    D_l: np.ndarray
    K_r: np.ndarray
    D_r: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    map_l: tuple[np.ndarray, np.ndarray]
    map_r: tuple[np.ndarray, np.ndarray]

    @property
    def focal_px(self) -> float:
        return float(self.P1[0, 0])

    @property
    def baseline_mm(self) -> float:
        return float(-self.P2[0, 3] / self.P2[0, 0])

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (cv.remap(left, *self.map_l, cv.INTER_LINEAR),
                cv.remap(right, *self.map_r, cv.INTER_LINEAR))

    def rectify_points(self, pts: np.ndarray, right: bool = False) -> np.ndarray:
        """Where raw image points (N, 1, 2) land in the rectified image."""
        if right:
            return cv.undistortPoints(pts, self.K_r, self.D_r, R=self.R2, P=self.P2)
        return cv.undistortPoints(pts, self.K_l, self.D_l, R=self.R1, P=self.P1)


class Rectifier:
    """Builds and caches a Rectification per frame size.

    alpha=0 crops so every output pixel is valid; alpha=1 keeps the whole field
    of view at the cost of black borders.
    """

    def __init__(self, calib: dict[str, np.ndarray], alpha: float = config.ALPHA):
        self.calib = calib
        self.alpha = alpha
        self._cache: dict[tuple[int, int], Rectification] = {}

    @classmethod
    def from_file(cls, path: Path = DEFAULT_CALIB, alpha: float = config.ALPHA) -> "Rectifier":
        return cls(load_calibration(path), alpha)

    @property
    def calib_size(self) -> tuple[int, int]:
        w, h = (int(v) for v in self.calib["image_size"])
        return w, h

    def at(self, size: tuple[int, int] | None = None) -> Rectification:
        size = size or self.calib_size
        if size not in self._cache:
            self._cache[size] = self._build(size)
        return self._cache[size]

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if left.shape != right.shape:
            raise ValueError(f"left {left.shape} and right {right.shape} differ in size")
        return self.at(left.shape[::-1]).rectify(left, right)

    def _build(self, size: tuple[int, int]) -> Rectification:
        c = self.calib
        w0, h0 = self.calib_size
        w, h = size
        if size != (w0, h0):
            log.info("frames are %dx%d, calibrated at %dx%d: rescaling K", w, h, w0, h0)
        K_l = scale_K(c["K_l"], w / w0, h / h0)
        K_r = scale_K(c["K_r"], w / w0, h / h0)
        R1, R2, P1, P2, Q, _, _ = cv.stereoRectify(
            K_l, c["D_l"], K_r, c["D_r"], size, c["R"], c["T"],
            flags=cv.CALIB_ZERO_DISPARITY, alpha=self.alpha)
        return Rectification(
            size, K_l, c["D_l"], K_r, c["D_r"], R1, R2, P1, P2, Q,
            cv.initUndistortRectifyMap(K_l, c["D_l"], R1, P1, size, cv.CV_16SC2),
            cv.initUndistortRectifyMap(K_r, c["D_r"], R2, P2, size, cv.CV_16SC2))


## ---- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="View the rectified live stream from the Pico.")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--alpha", type=float, default=config.ALPHA, help="0 = crop to valid pixels, 1 = keep all")
    parser.add_argument("--pattern", help="inner corners as COLSxROWS: measure row mismatch on a visible board")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--source", default=config.SOURCE, help="only accept chunks from this IP")
    parser.add_argument("--headless", action="store_true", help="no window: log, and write --snapshot")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT,
                        help=f"headless: latest rectified pair, rewritten every 5 s (default {DEFAULT_SNAPSHOT})")
    parser.add_argument("--scale", type=int, default=2, help="window: preview magnification")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ## `docker stop` sends SIGTERM, which PID 1 otherwise ignores.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if not args.calib.exists():
        sys.exit(f"{args.calib} not found: run `calibrate capture` and `calibrate solve` first")
    rect = Rectifier.from_file(args.calib, args.alpha)
    r0 = rect.at()
    log.info("calibrated at %dx%d, rectified focal %.1f px, baseline %.1f mm",
             *rect.calib_size, r0.focal_px, r0.baseline_mm)

    pattern = None
    if args.pattern:
        from .calibrate import find_corners, match_orientation, parse_pattern
        pattern = parse_pattern(args.pattern)
    headless = args.headless or no_display()
    if headless and args.snapshot:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)

    count, dys, view = 0, [], None
    last_log = time.monotonic()
    try:
        with Receiver(args.port, args.host, args.source) as rx:
            while True:
                pair = rx.get(timeout=0.1)
                if pair is not None:
                    count += 1
                    r = rect.at(pair.left.image.shape[::-1])
                    left, right = r.rectify(pair.left.image, pair.right.image)
                    view = draw_rows(np.hstack([left, right]))
                    if pattern is not None:
                        cl = find_corners(pair.left.image, pattern, fast=True)
                        cr = find_corners(pair.right.image, pattern, fast=True)
                        if cl is not None and cr is not None:
                            cr = match_orientation(cl, cr)
                            dy = np.abs(r.rectify_points(cl)[..., 1] - r.rectify_points(cr, right=True)[..., 1])
                            dys.append(float(dy.mean()))
                            cv.putText(view, f"row mismatch {dys[-1]:.2f} px", (8, 20),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv.LINE_AA)
                    if not headless:
                        cv.imshow(WINDOW, cv.resize(view, None, fx=args.scale, fy=args.scale,
                                                    interpolation=cv.INTER_NEAREST))

                if not headless:
                    key = cv.waitKey(1) & 0xFF
                    if key == ord("q") or (view is not None and cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1):
                        break

                now = time.monotonic()
                if now - last_log >= 5.0:
                    mismatch = f"row mismatch mean {np.mean(dys):.2f} px over {len(dys)} board views" if dys else ""
                    log.info("%.1f pairs/s  %s  %s", count / (now - last_log), rx.asm.stats, mismatch)
                    if headless and view is not None and args.snapshot:
                        cv.imwrite(str(args.snapshot), view)
                    count, dys, last_log = 0, [], now
    except KeyboardInterrupt:
        pass
    if not headless:
        cv.destroyAllWindows()


if __name__ == "__main__":
    main()
