"""Run the full pipeline on a noisy synthetic scene and show the result.

Usage: uv run python scripts/demo.py [--method hull_simple]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

from polytope3d.contracts import Intrinsics
from polytope3d.metrics import containment_rate, gap_stats, total_volume
from polytope3d.pipeline import PipelineConfig, run_pipeline
from polytope3d.synthetic import Box, render_depth

COLORS = ["tab:red", "tab:blue", "tab:green", "tab:orange"]

SCRATCH = Path(__file__).resolve().parent.parent / "scratch"


def draw_polytope(ax, poly, color):
    """Draw polytope edges in the (X, Z, -Y) view used for the scatter plot."""
    v = poly.vertices()
    for tri in ConvexHull(v).simplices:
        loop = np.r_[tri, tri[0]]
        ax.plot(v[loop, 0], v[loop, 2], -v[loop, 1], color=color, linewidth=0.6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="hull_simple")
    args = parser.parse_args()

    k = Intrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)
    boxes = [Box((-0.5, 0.5, 2.0), (0.5, 1.0, 3.0)), Box((1.0, 0.6, 3.5), (1.6, 1.0, 4.0))]
    frame = render_depth(k, cam_height=1.0, boxes=boxes, noise_k=0.002, dropout=0.02, seed=1)

    config = PipelineConfig(floor_threshold=0.02, floor_threshold_per_z2=0.006, eps=0.08)
    result = run_pipeline(frame, config)

    print("counts:", result.counts)
    print("warnings:", result.warnings or "none")
    print("timings (ms):", {name: round(ms, 1) for name, ms in result.timings_ms.items()})
    for obstacle in result.obstacles:
        print(f"\nobstacle {obstacle.id}: {len(obstacle.points)} points")
        for name, parts in obstacle.polytopes.items():
            gap = gap_stats(parts[0], obstacle.points)["mean"] if len(parts) == 1 else float("nan")
            print(
                f"  {name:12} planes {sum(p.n_planes for p in parts):4d}  "
                f"volume {total_volume(parts):.3f} m3  mean gap {gap:.3f} m  "
                f"contained {containment_rate(parts, obstacle.points):.3f}"
            )
        for name, reason in obstacle.failures.items():
            print(f"  {name:12} FAILED: {reason}")

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    shown = np.where(frame.depth <= config.max_range, frame.depth, np.nan)
    im = ax1.imshow(shown, vmin=0, vmax=config.max_range)
    fig.colorbar(im, ax=ax1, label="depth (m)")
    ax1.set_title("Noisy synthetic depth image")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    for obstacle, color in zip(result.obstacles, COLORS):
        pts = obstacle.points
        ax2.scatter(pts[::5, 0], pts[::5, 2], -pts[::5, 1], s=1, c=color)
        for poly in obstacle.polytopes.get(args.method, []):
            draw_polytope(ax2, poly, color)
    ax2.set_xlabel("X right (m)")
    ax2.set_ylabel("Z forward (m)")
    ax2.set_zlabel("height (-Y, m)")
    ax2.set_title(f"Obstacles and '{args.method}' polytopes")
    plt.tight_layout()
    SCRATCH.mkdir(exist_ok=True)
    plt.savefig(SCRATCH / "demo.png", dpi=150)


if __name__ == "__main__":
    main()
