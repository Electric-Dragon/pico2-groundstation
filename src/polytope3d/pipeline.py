"""End-to-end pipeline: DepthFrame -> floor -> obstacle clusters -> polytopes, with timings."""

import time
from contextlib import contextmanager
from dataclasses import dataclass

from polytope3d.algos import ALGORITHMS, MULTI_ALGORITHMS
from polytope3d.clustering import (
    cluster_indices,
    dbscan_labels,
    euclidean_labels,
    propagate_labels,
)
from polytope3d.polytope import DegenerateGeometry
from polytope3d.preprocessing import (
    fit_floor_plane,
    remove_statistical_outliers,
    split_by_plane,
    voxel_downsample,
)
from polytope3d.reconstruction import depth_to_points

DEFAULT_ALGORITHMS = ("aabb", "obb", "kdop14", "kdop26", "hull", "hull_simple")


@dataclass(frozen=True)
class PipelineConfig:
    """All tunable settings in one place. The numbers are placeholders to be tuned on real data."""

    min_range: float = 0.3
    max_range: float = 6.0
    voxel_size: float = 0.03
    outlier_k: int = 16
    outlier_std_ratio: float = 2.0
    floor_threshold: float = 0.02
    floor_threshold_per_z2: float = 0.0
    cluster_method: str = "euclidean"
    eps: float = 0.06
    min_points: int = 10
    algorithms: tuple = DEFAULT_ALGORITHMS
    margin: float = 0.0

    def __post_init__(self):
        if not 0 < self.min_range < self.max_range:
            raise ValueError("need 0 < min_range < max_range")
        if self.voxel_size < 0 or self.eps <= 0 or self.min_points < 1 or self.margin < 0:
            raise ValueError("voxel_size >= 0, eps > 0, min_points >= 1 and margin >= 0 required")
        if self.cluster_method not in ("euclidean", "dbscan"):
            raise ValueError("cluster_method must be 'euclidean' or 'dbscan'")
        unknown = set(self.algorithms) - set(ALGORITHMS) - set(MULTI_ALGORITHMS)
        if unknown:
            raise ValueError(f"unknown algorithms: {sorted(unknown)}")


@dataclass(frozen=True, eq=False)
class Obstacle:
    """One obstacle: its points and, per algorithm, a list of polytopes (or the failure reason)."""

    id: int
    points: object
    polytopes: dict
    failures: dict


@dataclass(frozen=True, eq=False)
class PipelineResult:
    floor: object
    obstacles: list
    timings_ms: dict
    counts: dict
    warnings: list


@contextmanager
def _timed(store, name):
    start = time.perf_counter()
    try:
        yield
    finally:
        store[name] = store.get(name, 0.0) + (time.perf_counter() - start) * 1000


def _fit(name, points, obstacle_id):
    if name in MULTI_ALGORITHMS:
        return MULTI_ALGORITHMS[name](points, obstacle_id=obstacle_id)
    return [ALGORITHMS[name](points, obstacle_id=obstacle_id)]


def run_pipeline(frame, config=None):
    """Run every stage on one DepthFrame. A stage that cannot proceed adds a warning; it does
    not stop the run. In particular, if no floor is found, all points are treated as obstacle
    candidates and a warning is recorded."""
    config = config or PipelineConfig()
    timings, counts, warnings = {}, {}, []

    with _timed(timings, "reconstruct"):
        points = depth_to_points(frame, config.min_range, config.max_range)
    counts["points"] = len(points)
    if len(points) < 3:
        warnings.append("too few valid points in range")
        return PipelineResult(None, [], timings, counts, warnings)

    with _timed(timings, "floor"):
        try:
            floor = fit_floor_plane(
                points, config.floor_threshold, config.floor_threshold_per_z2
            )
            above = split_by_plane(
                points, floor, config.floor_threshold, config.floor_threshold_per_z2
            )[0]
        except ValueError as exc:
            floor, above = None, points
            warnings.append(f"no floor plane ({exc}); all points treated as obstacle candidates")
    counts["above_floor"] = len(above)

    with _timed(timings, "outliers"):
        above = remove_statistical_outliers(above, config.outlier_k, config.outlier_std_ratio)
    counts["after_outliers"] = len(above)

    with _timed(timings, "downsample"):
        coarse = voxel_downsample(above, config.voxel_size) if config.voxel_size > 0 else above
    counts["downsampled"] = len(coarse)

    with _timed(timings, "cluster"):
        label_fn = euclidean_labels if config.cluster_method == "euclidean" else dbscan_labels
        coarse_labels = label_fn(coarse, config.eps, config.min_points)
        labels = propagate_labels(coarse, coarse_labels, above, config.eps)
    clusters = cluster_indices(labels)
    counts["clusters"] = len(clusters)

    obstacles = []
    for k, idx in enumerate(clusters):
        cluster = above[idx]
        fits, failures = {}, {}
        for name in config.algorithms:
            with _timed(timings, f"fit_{name}"):
                try:
                    parts = _fit(name, cluster, k)
                    fits[name] = [p.inflated(config.margin) if config.margin > 0 else p for p in parts]
                except DegenerateGeometry as exc:
                    failures[name] = str(exc)
        obstacles.append(Obstacle(id=k, points=cluster, polytopes=fits, failures=failures))
    return PipelineResult(floor, obstacles, timings, counts, warnings)