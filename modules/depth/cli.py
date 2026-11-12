"""Live disparity/depth viewer: receive -> rectify -> match.

  uv run python -m modules.depth [--algo sgbm|census-sgm|elas] [--headless]
  uv run python -m modules.depth --images left.png right.png [--rectified]

The window shows the rectified left image next to the colour-coded disparity
(near = warm, far = cool, black = no match), with the matching time and the
distance at the centre cross. Headless, it logs and rewrites --snapshot every 5 s.
--images runs once on a pair from disk (e.g. data/calib/0000_L.png and _R.png)
and writes --snapshot; the pair is rectified with --calib unless --rectified.
"""
import argparse
import logging
import signal
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from .. import config
from ..rectify import DEFAULT_CALIB, Rectification, Rectifier
from . import ALGORITHMS, StereoMatcher, create, depth_mm

log = logging.getLogger(__name__)

DEFAULT_SNAPSHOT = config.DATA_DIR / "disparity.png"
WINDOW = "depth"


def colorize(disp: np.ndarray, num_disparities: int) -> np.ndarray:
    """BGR image of a disparity map: TURBO colour map over 0..num_disparities, black where NaN."""
    valid = np.isfinite(disp)
    scaled = np.zeros(disp.shape, np.uint8)
    scaled[valid] = np.clip(disp[valid] * (255 / num_disparities), 0, 255).astype(np.uint8)
    view = cv.applyColorMap(scaled, cv.COLORMAP_TURBO)
    view[~valid] = 0
    return view


def render(left: np.ndarray, disp: np.ndarray, matcher: StereoMatcher, ms: float,
           r: Rectification | None) -> np.ndarray:
    """Rectified left | disparity, with timing, coverage and the distance at the centre."""
    view = np.hstack([cv.cvtColor(left, cv.COLOR_GRAY2BGR), colorize(disp, matcher.num_disparities)])
    h, w = disp.shape
    cx, cy = w + w // 2, h // 2
    cv.drawMarker(view, (cx, cy), (255, 255, 255), cv.MARKER_CROSS, 10, 1)
    lines = [f"{matcher.name} {ms:.0f} ms  {np.isfinite(disp).mean():.0%} valid"]
    d = disp[cy - 2:cy + 3, w // 2 - 2:w // 2 + 3]
    if np.isfinite(d).any():
        d_c = float(np.nanmedian(d))
        lines.append(f"centre d={d_c:.1f} px" + (
            f"  {depth_mm(np.array([d_c]), r.focal_px, r.baseline_mm)[0] / 1000:.2f} m" if r and d_c > 0 else ""))
    for i, text in enumerate(lines):
        org = (6, 16 + 16 * i)
        cv.putText(view, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv.LINE_AA)
        cv.putText(view, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv.LINE_AA)
    return view


def timed(matcher: StereoMatcher, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, float]:
    t0 = time.perf_counter()
    disp = matcher.compute(left, right)
    return disp, (time.perf_counter() - t0) * 1000


def run_images(args, matcher: StereoMatcher) -> None:
    left, right = (cv.imread(str(p), cv.IMREAD_GRAYSCALE) for p in args.images)
    for p, img in zip(args.images, (left, right)):
        if img is None:
            sys.exit(f"can't read {p}")
    r = None
    if not args.rectified:
        if not args.calib.exists():
            sys.exit(f"{args.calib} not found: pass --rectified if the images already are, or calibrate first")
        r = Rectifier.from_file(args.calib, args.alpha).at(left.shape[::-1])
        left, right = r.rectify(left, right)
    disp, ms = timed(matcher, left, right)
    log.info("%s: %.0f ms, %.0f%% valid, disparity %.1f..%.1f px", matcher.name, ms,
             100 * np.isfinite(disp).mean(), np.nanmin(disp), np.nanmax(disp))
    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(args.snapshot), render(left, disp, matcher, ms, r))
    log.info("wrote %s", args.snapshot)


def run_live(args, matcher: StereoMatcher) -> None:
    from ..receiver import Receiver, no_display

    if not args.calib.exists():
        sys.exit(f"{args.calib} not found: run `calibrate capture` and `calibrate solve` first")
    rect = Rectifier.from_file(args.calib, args.alpha)
    headless = args.headless or no_display()
    if headless:
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)

    count, times, view = 0, [], None
    last_log = time.monotonic()
    try:
        with Receiver(args.port, args.host, args.source) as rx:
            while True:
                pair = rx.get(timeout=0.1)
                if pair is not None:
                    count += 1
                    r = rect.at(pair.left.image.shape[::-1])
                    left, right = r.rectify(pair.left.image, pair.right.image)
                    disp, ms = timed(matcher, left, right)
                    times.append(ms)
                    view = render(left, disp, matcher, ms, r)
                    if not headless:
                        cv.imshow(WINDOW, cv.resize(view, None, fx=args.scale, fy=args.scale,
                                                    interpolation=cv.INTER_NEAREST))

                if not headless:
                    key = cv.waitKey(1) & 0xFF
                    if key == ord("q") or (view is not None and cv.getWindowProperty(WINDOW, cv.WND_PROP_VISIBLE) < 1):
                        break

                now = time.monotonic()
                if now - last_log >= 5.0:
                    match = f"match {np.mean(times):.0f} ms" if times else ""
                    log.info("%.1f pairs/s  %s  consumer-dropped %d  %s",
                             count / (now - last_log), match, rx.dropped, rx.asm.stats)
                    if headless and view is not None:
                        cv.imwrite(str(args.snapshot), view)
                    count, times, last_log = 0, [], now
    except KeyboardInterrupt:
        pass
    if not headless:
        cv.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Stereo depth from the Pico's live stream (or a saved pair).")
    parser.add_argument("--algo", choices=list(ALGORITHMS), default=config.DEPTH_ALGO,
                        help=f"stereo matcher (default {config.DEPTH_ALGO}, from GCS_DEPTH_ALGO)")
    parser.add_argument("--num-disparities", type=int, default=config.NUM_DISPARITIES,
                        help=f"disparity search range in px (default {config.NUM_DISPARITIES})")
    parser.add_argument("--calib", type=Path, default=DEFAULT_CALIB)
    parser.add_argument("--alpha", type=float, default=config.ALPHA, help="0 = crop to valid pixels, 1 = keep all")
    parser.add_argument("--images", nargs=2, type=Path, metavar=("LEFT", "RIGHT"),
                        help="run once on this pair instead of the live stream")
    parser.add_argument("--rectified", action="store_true", help="--images are already rectified")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--source", default=config.SOURCE, help="only accept chunks from this IP")
    parser.add_argument("--headless", action="store_true", help="no window: log, and write --snapshot")
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT,
                        help=f"output image: headless (every 5 s) or --images (default {DEFAULT_SNAPSHOT})")
    parser.add_argument("--scale", type=int, default=2, help="window: preview magnification")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ## `docker stop` sends SIGTERM, which PID 1 otherwise ignores.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    try:
        matcher = create(args.algo, num_disparities=args.num_disparities)
    except ValueError as e:  # bad settings, e.g. from .env
        sys.exit(str(e))
    log.info("depth algorithm: %r", matcher)
    if args.images:
        run_images(args, matcher)
    else:
        run_live(args, matcher)
