"""Compare all polytope algorithms on identical synthetic clusters (volume, planes, bytes, time)."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from polytope3d.algos import ALGORITHMS
from polytope3d.synthetic import Box, make_box_surface_cloud

FIXED_DIRECTIONS = {"aabb", "kdop14", "kdop26"}


def rotated_box_cloud(n=20000):
    pts = make_box_surface_cloud(Box((-0.5, -0.2, -0.3), (0.5, 0.2, 0.3)), n).astype(np.float64)
    a = np.radians(30.0)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (pts @ rot.T + np.array([0.0, 0.5, 3.0])).astype(np.float32)


def sphere_cloud(n=20000, radius=0.5):
    v = np.random.default_rng(0).normal(size=(n, 3))
    centre = np.array([0.0, 0.5, 3.0])
    return (radius * v / np.linalg.norm(v, axis=1, keepdims=True) + centre).astype(np.float32)


def median_ms(fit, pts, repeats=20):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fit(pts)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


clouds = {"rotated box": rotated_box_cloud(), "sphere": sphere_cloud()}
fig, axes = plt.subplots(1, len(clouds), figsize=(11, 4))
for ax, (cloud_name, pts) in zip(axes, clouds.items()):
    hull_volume = ALGORITHMS["hull"](pts).volume()
    print(f"\n{cloud_name} ({len(pts)} points)")
    print(f"{'method':8} {'planes':>6} {'vol/hull':>9} {'bytes':>6} {'median ms':>10}")
    for name, fit in ALGORITHMS.items():
        poly = fit(pts)
        ratio = poly.volume() / hull_volume
        nbytes = poly.n_planes * (4 if name in FIXED_DIRECTIONS else 16)
        print(f"{name:8} {poly.n_planes:6d} {ratio:9.3f} {nbytes:6d} {median_ms(fit, pts):10.2f}")
        ax.scatter(poly.n_planes, ratio)
        ax.annotate(name, (poly.n_planes, ratio))
    ax.set_xscale("log")
    ax.set_xlabel("number of planes")
    ax.set_ylabel("volume / hull volume")
    ax.set_title(cloud_name)
Path("scratch").mkdir(exist_ok=True)
plt.tight_layout()
plt.savefig("scratch/pareto.png", dpi=150)