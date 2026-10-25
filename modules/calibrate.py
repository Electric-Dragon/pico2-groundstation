"""Stereo calibration for the two HM01B0 cameras, following the OpenCV
camera-calibration tutorial, extended to a stereo pair.

  capture  Live from the Pico. Saves left/right PNG pairs in which the board is visible.
           The default is a preview window (space = save, a = toggle auto-capture,
           q = quit). --headless auto-captures with no display, for Docker.
  solve    Offline, from the saved pairs. Calibrates each camera on its own, then
           solves stereo R, T with those intrinsics fixed, and writes calib.npz
           plus a rectification check image.

  uv run python -m modules.calibrate capture --pattern 8x5
  uv run python -m modules.calibrate solve --pattern 8x5 --square-mm 30
"""
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

if __name__ == "__main__" and not __package__:
    ## Allow `python modules/calibrate.py` as well as `python -m modules.calibrate`.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    __package__ = "modules"

import cv2 as cv
import numpy as np

from .assembler import StereoPair
from . import config
from .receiver import Receiver, no_display
from .rectify import DEFAULT_CALIB, Rectifier, draw_rows

log = logging.getLogger(__name__)

## termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

SUBPIX_WIN = (5, 5)  # the tutorial's (11, 11) is too wide for ~320 px images
MIN_VIEWS = 10
DEFAULT_IMAGES = config.DATA_DIR / "calib"
DEFAULT_OUTPUT = DEFAULT_CALIB
WINDOW = "calibration capture"
CELL_NAMES = ["top-left", "top", "top-right", "left", "centre", "right",
              "bottom-left", "bottom", "bottom-right"]


## ---- board detection -------------------------------------------------------

def parse_pattern(s: str) -> tuple[int, int]:
    """'8x5' -> (8, 5): inner corners per row, per column."""
    cols, rows = (int(v) for v in s.lower().split("x"))
    return cols, rows


def board_points(pattern: tuple[int, int], square_mm: float) -> np.ndarray:
    """Object points (0,0,0), (1,0,0), (2,0,0) ... scaled to millimetres."""
    cols, rows = pattern
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return objp


def find_corners(gray: np.ndarray, pattern: tuple[int, int], fast: bool = False) -> np.ndarray | None:
    flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv.CALIB_CB_FAST_CHECK  # bails out quickly on frames with no board
    found, corners = cv.findChessboardCorners(gray, pattern, flags=flags)
    if not found:
        return None
    return cv.cornerSubPix(gray, corners, SUBPIX_WIN, (-1, -1), criteria)


def match_orientation(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """The detector may number a board's corners from either end (as if rotated 180°).
    Stereo needs both views numbered the same way, so flip right if it disagrees."""
    vl = left[-1, 0] - left[0, 0]
    vr = right[-1, 0] - right[0, 0]
    return right[::-1].copy() if np.dot(vl, vr) < 0 else right


def missing_cells(corner_sets: list[np.ndarray], size: tuple[int, int]) -> list[str]:
    """Which cells of a 3x3 grid over the image no saved corner has landed in."""
    w, h = size
    hit = np.zeros(9, bool)
    for c in corner_sets:
        xy = c.reshape(-1, 2)
        col = np.clip((xy[:, 0] * 3 // w).astype(int), 0, 2)
        row = np.clip((xy[:, 1] * 3 // h).astype(int), 0, 2)
        hit[row * 3 + col] = True
    return [name for name, h in zip(CELL_NAMES, hit) if not h]


def _mean_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=-1).mean())


## ---- capture ---------------------------------------------------------------

class PairStore:
    """Writes pairs as NNNN_L.png / NNNN_R.png, continuing after any already in the folder.
    Numbered by save order, not frame_id, which restarts when the Pico reboots."""

    def __init__(self, folder: Path):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        taken = [int(p.name[:-6]) for p in folder.glob("*_L.png") if p.name[:-6].isdigit()]
        self.next = max(taken, default=-1) + 1

    def save(self, pair: StereoPair) -> str:
        stem = f"{self.next:04d}"
        cv.imwrite(str(self.folder / f"{stem}_L.png"), pair.left.image)
        cv.imwrite(str(self.folder / f"{stem}_R.png"), pair.right.image)
        self.next += 1
        return stem


class AutoCapture:
    """Decides when a live view is worth saving: the board is held still (no motion
    blur), the last save was a moment ago, and the pose isn't one already captured."""

    def __init__(self, still_px: float = 1.5, novel_px: float = 20.0, cooldown_s: float = 1.0):
        self.still_px = still_px
        self.novel_px = novel_px
        self.cooldown_s = cooldown_s
        self.saved: tuple[list[np.ndarray], list[np.ndarray]] = ([], [])  # corners per side
        self.count = 0
        self.stereo = 0  # saves where both cameras saw the board
        self._prev: tuple[np.ndarray | None, np.ndarray | None] = (None, None)
        self._last_save = float("-inf")

    def update(self, corners: tuple[np.ndarray | None, np.ndarray | None], now: float) -> bool:
        prev, self._prev = self._prev, corners
        if corners[0] is None and corners[1] is None:
            return False
        for c, p in zip(corners, prev):
            if c is not None and (p is None or np.abs(c - p).max() > self.still_px):
                return False
        if now - self._last_save < self.cooldown_s:
            return False
        return any(c is not None and self._novel(c, seen) for c, seen in zip(corners, self.saved))

    def record(self, corners: tuple[np.ndarray | None, np.ndarray | None], now: float) -> None:
        self._last_save = now
        self.count += 1
        self.stereo += corners[0] is not None and corners[1] is not None
        for c, seen in zip(corners, self.saved):
            if c is not None:
                seen.append(c)

    def _novel(self, c: np.ndarray, seen: list[np.ndarray]) -> bool:
        return all(min(_mean_dist(c, s), _mean_dist(c[::-1], s)) > self.novel_px for s in seen)


def _draw_view(gray: np.ndarray, corners: np.ndarray | None, pattern: tuple[int, int],
               seen: list[np.ndarray], scale: int) -> np.ndarray:
    img = cv.resize(gray, None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)
    img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    for s in seen:  # corners of earlier saves, to show which areas are covered
        for x, y in s.reshape(-1, 2) * scale:
            cv.circle(img, (int(x), int(y)), 1, (0, 180, 255), -1)
    if corners is not None:
        cv.drawChessboardCorners(img, pattern, corners * scale, True)
    return img


def capture(args: argparse.Namespace) -> None:
    headless = args.headless or no_display()
    if headless and not args.headless:
        log.info("no display found, running headless")
    store = PairStore(args.images)
    auto = AutoCapture()
    auto_on = headless
    latest = None
    size = None
    shown = False
    last_log = time.monotonic()
    log.info("saving to %s, pattern %dx%d, %s", args.images, *args.pattern,
             f"auto-capturing {args.target} pairs" if headless else "space = save, a = auto, q = quit")

    with Receiver(args.port, args.host, args.source) as rx:
        while not (headless and auto.count >= args.target):
            pair = rx.get(timeout=0.1)
            now = time.monotonic()
            save = False
            if pair is not None:
                size = pair.left.image.shape[::-1]
                corners = (find_corners(pair.left.image, args.pattern, fast=True),
                           find_corners(pair.right.image, args.pattern, fast=True))
                latest = (pair, corners)
                save = auto_on and auto.update(corners, now)

            if not headless:
                if pair is not None:
                    view = np.hstack([
                        _draw_view(pair.left.image, corners[0], args.pattern, auto.saved[0], args.scale),
                        _draw_view(pair.right.image, corners[1], args.pattern, auto.saved[1], args.scale)])
                    cv.putText(view, f"saved {auto.count} (both {auto.stereo})  auto {'on' if auto_on else 'off'}"
                               "   [space] save  [a] auto  [q] quit",
                               (8, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                    cv.imshow(WINDOW, view)
                    shown = True
                key = cv.waitKey(1) & 0xFF
                if key == ord("q") or (shown and cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1):
                    break
                if key == ord("a"):
                    auto_on = not auto_on
                if key == ord(" ") and latest is not None and any(c is not None for c in latest[1]):
                    save = True

            if save:
                pair, corners = latest
                stem = store.save(pair)
                auto.record(corners, now)
                sides = "+".join(s for s, c in zip(("L", "R"), corners) if c is not None)
                log.info("saved %s (board in %s)", stem, sides)

            if now - last_log >= 5.0:
                last_log = now
                if size is None:
                    log.info("waiting for frames on port %d ... %s", args.port, rx.asm.stats)
                else:
                    log.info("saved %d (both %d)  not yet covered: left %s, right %s",
                             auto.count, auto.stereo,
                             missing_cells(auto.saved[0], size) or "-",
                             missing_cells(auto.saved[1], size) or "-")

    if not headless:
        cv.destroyAllWindows()
    log.info("captured %d pairs (%d with the board in both) in %s", auto.count, auto.stereo, args.images)


## ---- solve -----------------------------------------------------------------

def _calibrate_mono(name: str, views: list[tuple[str, np.ndarray]], objp: np.ndarray,
                    size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, float]:
    if len(views) < MIN_VIEWS:
        sys.exit(f"{name}: board found in only {len(views)} images, need at least {MIN_VIEWS}")
    imgpoints = [c for _, c in views]
    rms, K, D, rvecs, tvecs = cv.calibrateCamera([objp] * len(views), imgpoints, size, None, None)
    errs = [float(np.sqrt(np.mean(np.sum((cv.projectPoints(objp, rv, tv, K, D)[0] - c) ** 2, axis=2))))
            for rv, tv, c in zip(rvecs, tvecs, imgpoints)]
    log.info("%s: %d views, RMS reprojection error %.3f px", name, len(views), rms)
    limit = 2 * float(np.median(errs))
    bad = [f"{stem} ({e:.2f} px)" for (stem, _), e in zip(views, errs) if e > limit]
    if bad:
        log.info("%s: views well above the median error, consider deleting: %s", name, ", ".join(bad))
    return K, D, rms


def solve(args: argparse.Namespace) -> None:
    objp = board_points(args.pattern, args.square_mm)
    lefts = sorted(args.images.glob("*_L.png"))
    if not lefts:
        sys.exit(f"no *_L.png images in {args.images}")

    size = None
    views = []  # (stem, left corners | None, right corners | None)
    for lp in lefts:
        stem = lp.name[:-6]
        gl = cv.imread(str(lp), cv.IMREAD_GRAYSCALE)
        gr = cv.imread(str(lp.with_name(f"{stem}_R.png")), cv.IMREAD_GRAYSCALE)
        if gl is None or gr is None:
            log.warning("%s: missing or unreadable image, skipped", stem)
            continue
        size = size or gl.shape[::-1]
        if gl.shape[::-1] != size or gr.shape[::-1] != size:
            log.warning("%s: size differs from %dx%d, skipped", stem, *size)
            continue
        views.append((stem, find_corners(gl, args.pattern), find_corners(gr, args.pattern)))
    log.info("%d pairs at %dx%d: board found in %d left, %d right, %d both",
             len(views), *size, sum(l is not None for _, l, _ in views),
             sum(r is not None for _, _, r in views),
             sum(l is not None and r is not None for _, l, r in views))

    K_l, D_l, rms_l = _calibrate_mono("left", [(s, l) for s, l, _ in views if l is not None], objp, size)
    K_r, D_r, rms_r = _calibrate_mono("right", [(s, r) for s, _, r in views if r is not None], objp, size)

    stereo = [(l, match_orientation(l, r)) for _, l, r in views if l is not None and r is not None]
    if len(stereo) < MIN_VIEWS:
        sys.exit(f"board found in both cameras in only {len(stereo)} pairs, need at least {MIN_VIEWS}")
    rms_s, _, _, _, _, R, T, E, F = cv.stereoCalibrate(
        [objp] * len(stereo), [l for l, _ in stereo], [r for _, r in stereo],
        K_l, D_l, K_r, D_r, size, criteria=criteria, flags=cv.CALIB_FIX_INTRINSIC)
    log.info("stereo: %d pairs, RMS reprojection error %.3f px, baseline %.1f mm",
             len(stereo), rms_s, float(np.linalg.norm(T)))

    calib = dict(image_size=np.array(size), K_l=K_l, D_l=D_l, K_r=K_r, D_r=D_r, R=R, T=T, E=E, F=F,
                 rms=np.array([rms_l, rms_r, rms_s]), pattern=np.array(args.pattern),
                 square_mm=np.array(args.square_mm))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **calib)
    log.info("wrote %s", args.output)

    check_rectification(calib, stereo, views, args)


def check_rectification(calib, stereo, views, args) -> None:
    """After rectification a board corner should sit on the same row in both images."""
    rect = Rectifier(calib).at()
    dy = np.concatenate([
        np.abs(rect.rectify_points(l)[..., 1] - rect.rectify_points(r, right=True)[..., 1]).ravel()
        for l, r in stereo])
    log.info("rectified row mismatch: mean %.2f px, max %.2f px (well under 1 px is good)",
             dy.mean(), dy.max())

    stem = next(s for s, l, r in views if l is not None and r is not None)
    left, right = rect.rectify(*(cv.imread(str(args.images / f"{stem}_{side}.png"), cv.IMREAD_GRAYSCALE)
                                 for side in ("L", "R")))
    view = draw_rows(np.hstack([left, right]))
    out = args.output.with_name("rectify_check.png")
    cv.imwrite(str(out), view)
    log.info("wrote %s (pair %s rectified, features should line up along the green rows)", out, stem)
    if args.show:
        cv.imshow("rectification check", view)
        cv.waitKey(0)
        cv.destroyAllWindows()


## ---- CLI -------------------------------------------------------------------

def main() -> None:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--pattern", type=parse_pattern, default=config.PATTERN,
                        help=f"inner corners as COLSxROWS (default {config.PATTERN})")
    common.add_argument("--images", type=Path, default=DEFAULT_IMAGES,
                        help=f"folder of NNNN_L.png / NNNN_R.png pairs (default {DEFAULT_IMAGES})")

    parser = argparse.ArgumentParser(description="Stereo calibration for the Pico ground station.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    cap = sub.add_parser("capture", parents=[common], help="save board views from the live stream")
    cap.add_argument("--port", type=int, default=config.PORT)
    cap.add_argument("--host", default=config.HOST)
    cap.add_argument("--source", default=config.SOURCE, help="only accept chunks from this IP")
    cap.add_argument("--headless", action="store_true", help="no window, auto-capture (Docker)")
    cap.add_argument("--target", type=int, default=40, help="headless: stop after this many saves")
    cap.add_argument("--scale", type=int, default=2, help="window: preview magnification")

    sol = sub.add_parser("solve", parents=[common], help="calibrate from saved pairs")
    sol.add_argument("--square-mm", type=float, default=config.SQUARE_MM, required=config.SQUARE_MM is None,
                     help="measured square size in mm (or GCS_SQUARE_MM in .env)")
    sol.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    sol.add_argument("--show", action="store_true", help="show the rectification check in a window")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ## `docker stop` sends SIGTERM, which PID 1 otherwise ignores.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        {"capture": capture, "solve": solve}[args.cmd](args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
