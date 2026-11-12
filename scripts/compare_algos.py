"""Compare every polytope algorithm on identical synthetic clusters.

Usage: uv run python scripts/compare_algos.py [--points 20000] [--repeats 10]
Volumes are relative to the exact convex hull. Timings are measured on your machine.
"""

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polytope3d.algos import ALGORITHMS, MULTI_ALGORITHMS
from polytope3d.metrics import estimated_bytes, gap_stats, total_volume
from polytope3d.synthetic import Box, make_box_surface_cloud

ALL_FITTERS = {**ALGORITHMS, **MULTI_ALGORITHMS}

SCRATCH = Path(__file__).resolve().parent.parent / "scratch"

def rotated_box_cloud(n):
    pts = make_box_surface_cloud(Box((-0.5, -0.2, -0.3), (0.5, 0.2, 0.3)), n).astype(np.float64)
    a = np.radians(30.0)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (pts @ rot.T + np.array([0.0, 0.5, 3.0])).astype(np.float32)


def sphere_cloud(n, radius=0.5):
    v = np.random.default_rng(0).normal(size=(n, 3))
    centre = np.array([0.0, 0.5, 3.0])
    return (radius * v / np.linalg.norm(v, axis=1, keepdims=True) + centre).astype(np.float32)


def l_shape_cloud(n):
    arm_a, arm_b = Box((0, 0, 0), (1.0, 0.2, 0.2)), Box((0, 0, 0), (0.2, 1.0, 0.2))

    def strictly_inside(p, box, eps=1e-4):
        lo, hi = np.array(box.lo), np.array(box.hi)
        return np.all((p > lo + eps) & (p < hi - eps), axis=1)

    pa = make_box_surface_cloud(arm_a, n // 2, seed=1)
    pb = make_box_surface_cloud(arm_b, n // 2, seed=2)
    return np.vstack([pa[~strictly_inside(pa, arm_b)], pb[~strictly_inside(pb, arm_a)]]) + [0.0, 0.5, 3.0]


def median_ms(fit, pts, repeats):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fit(pts)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()

    clouds = {
        "rotated box": rotated_box_cloud(args.points),
        "sphere": sphere_cloud(args.points),
        "L-shape": l_shape_cloud(args.points),
    }
    fig, axes = plt.subplots(1, len(clouds), figsize=(15, 4))
    for ax, (cloud_name, pts) in zip(axes, clouds.items()):
        hull_volume = ALGORITHMS["hull"](pts).volume()
        print(f"\n{cloud_name} ({len(pts)} points)")
        header = f"{'method':12} {'parts':>5} {'planes':>6} {'vol/hull':>9} {'mean gap m':>11}"
        print(header + f" {'bytes':>6} {'median ms':>10}")
        for name, fit in ALL_FITTERS.items():
            result = fit(pts)
            parts = result if isinstance(result, list) else [result]
            planes = sum(p.n_planes for p in parts)
            ratio = total_volume(parts) / hull_volume
            gap = f"{gap_stats(parts[0], pts)['mean']:.4f}" if len(parts) == 1 else "-"
            ms = median_ms(fit, pts, args.repeats)
            print(
                f"{name:12} {len(parts):5d} {planes:6d} {ratio:9.3f} {gap:>11} "
                f"{estimated_bytes(parts):6d} {ms:10.2f}"
            )
            ax.scatter(planes, ratio)
            ax.annotate(name, (planes, ratio), fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("total planes")
        ax.set_ylabel("volume / hull volume")
        ax.set_title(cloud_name)
    SCRATCH.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(SCRATCH / "pareto.png", dpi=150)


if __name__ == "__main__":
    main()