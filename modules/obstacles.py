"""Bridge from stereo depth to obstacle polytopes (polytope3d).

  frame = to_depth_frame(disp, rect, pair.frame_id, pair.left.t_capture_us)
  result = run_pipeline(frame, pipeline_config())
  draw(view, result, frame.intrinsics, "aabb")

The depth image is in the rectified left camera's frame, so its intrinsics come
from P1 (not K_l). Used by `python -m modules.depth --obstacles`.
"""

import cv2 as cv
import numpy as np

from polytope3d.algos import ALGORITHMS, MULTI_ALGORITHMS
from polytope3d.contracts import DepthFrame, Intrinsics
from polytope3d.pipeline import PipelineConfig, PipelineResult, run_pipeline
from polytope3d.polytope import DegenerateGeometry
from polytope3d.reconstruction import project_points

from . import config
from .depth import depth_mm
from .rectify import Rectification

__all__ = ["FITTERS", "draw", "pipeline_config", "run_pipeline", "summary", "to_depth_frame"]

## Names accepted by --obstacle-algo / GCS_OBSTACLE_ALGO.
FITTERS: list[str] = [*ALGORITHMS, *MULTI_ALGORITHMS]

## One BGR colour per obstacle, cycled.
COLORS = [(0, 0, 255), (255, 128, 0), (0, 200, 0), (0, 200, 255), (255, 0, 255), (255, 255, 0)]


def intrinsics(r: Rectification, width: int, height: int) -> Intrinsics:
    P = r.P1
    return Intrinsics(
        fx=float(P[0, 0]), fy=float(P[1, 1]), cx=float(P[0, 2]), cy=float(P[1, 2]),
        width=width, height=height,
    )


def to_depth_frame(
    disp: np.ndarray, r: Rectification, frame_id: int = 0, t_capture_us: int = 0
) -> DepthFrame:
    """Disparity (px, NaN = no match) -> DepthFrame in metres (float32, NaN = invalid)."""
    depth = (depth_mm(disp, r.focal_px, r.baseline_mm) / 1000).astype(np.float32)
    h, w = disp.shape
    return DepthFrame(
        depth, intrinsics(r, w, h), frame_id=frame_id, timestamp=t_capture_us / 1e6,
        rig_id="pico-stereo",
    )


def pipeline_config(
    algo: str = config.OBSTACLE_ALGO, max_range: float = config.OBSTACLE_MAX_RANGE
) -> PipelineConfig:
    """Pipeline settings for live stereo: one fitter, and a floor tolerance that grows
    with Z² like stereo depth error does. Placeholders until tuned on real scenes."""
    return PipelineConfig(
        max_range=max_range,
        floor_threshold=0.02,
        floor_threshold_per_z2=0.01,
        eps=0.08,
        algorithms=(algo,),
    )


def nearest_m(obstacle) -> float:
    return float(obstacle.points[:, 2].min())


def total_ms(result: PipelineResult) -> float:
    return sum(result.timings_ms.values())


def summary(result: PipelineResult) -> str:
    """One line for the log: obstacle count, nearest distances, floor, time."""
    near = sorted(nearest_m(o) for o in result.obstacles)
    dists = " ".join(f"{z:.2f}" for z in near[:5]) + (" ..." if len(near) > 5 else "")
    text = f"{len(result.obstacles)} obstacles" + (f" (nearest m: {dists})" if near else "")
    text += f", floor {'found' if result.floor is not None else 'not found'}"
    return text + f", {total_ms(result):.0f} ms"


def draw(view: np.ndarray, result: PipelineResult, k: Intrinsics, algo: str) -> None:
    """Outline each obstacle's polytope(s) on the left half of `view` (in place), with
    the distance to its nearest point, and add an obstacle count line at the top left."""
    for o in result.obstacles:
        color = COLORS[o.id % len(COLORS)]
        outlines = []
        for poly in o.polytopes.get(algo, []):
            try:
                v = poly.vertices()
            except DegenerateGeometry:
                continue
            if (v[:, 2] <= 0.05).any():  # reaches behind the camera: projection is meaningless
                continue
            uv = project_points(v, k)
            outlines.append(cv.convexHull(np.round(uv).astype(np.int32)))
        if not outlines:
            continue
        cv.polylines(view[:, : k.width], outlines, True, color, 1, cv.LINE_AA)
        x, y, _, _ = cv.boundingRect(np.vstack(outlines))
        org = (int(np.clip(x, 0, k.width - 40)), int(np.clip(y - 3, 10, k.height - 3)))
        _text(view, f"{nearest_m(o):.2f}m", org, color)
    floor = "" if result.floor is not None else "  no floor"
    line = f"{len(result.obstacles)} obstacles ({algo}) {total_ms(result):.0f} ms{floor}"
    _text(view, line, (6, 48), (255, 255, 255))


def _text(view, text, org, color):
    cv.putText(view, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv.LINE_AA)
    cv.putText(view, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv.LINE_AA)
