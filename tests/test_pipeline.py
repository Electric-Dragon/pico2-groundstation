import numpy as np
import pytest

from polytope3d.contracts import DepthFrame
from polytope3d.metrics import containment_rate
from polytope3d.pipeline import DEFAULT_ALGORITHMS, PipelineConfig, run_pipeline


def test_scene_end_to_end(scene_frame):
    result = run_pipeline(scene_frame)
    np.testing.assert_allclose(result.floor.normal, [0, -1, 0], atol=1e-2)
    assert result.warnings == [] and len(result.obstacles) == 2
    assert result.counts["clusters"] == 2
    for stage in ("reconstruct", "floor", "outliers", "downsample", "cluster", "fit_hull"):
        assert result.timings_ms[stage] >= 0
    for obstacle in result.obstacles:
        assert obstacle.failures == {}
        assert set(obstacle.polytopes) == set(DEFAULT_ALGORITHMS)
        for parts in obstacle.polytopes.values():
            assert containment_rate(parts, obstacle.points) == 1.0


def test_margin_is_applied_everywhere(scene_frame):
    result = run_pipeline(scene_frame, PipelineConfig(margin=0.05))
    for obstacle in result.obstacles:
        for parts in obstacle.polytopes.values():
            assert all(p.margin == pytest.approx(0.05) for p in parts)
            assert containment_rate(parts, obstacle.points) == 1.0


def test_split_algorithm_through_pipeline(scene_frame):
    result = run_pipeline(scene_frame, PipelineConfig(algorithms=("split_obb",)))
    for obstacle in result.obstacles:
        parts = obstacle.polytopes["split_obb"]
        assert all(p.obstacle_id == obstacle.id for p in parts)
        assert containment_rate(parts, obstacle.points) == 1.0


def test_wall_without_floor_warns_and_degenerate_fit_is_reported(intrinsics):
    wall = DepthFrame(depth=np.full((240, 320), 2.0, dtype=np.float32), intrinsics=intrinsics)
    result = run_pipeline(wall)
    assert result.floor is None and any("no floor" in w for w in result.warnings)
    assert len(result.obstacles) == 1
    obstacle = result.obstacles[0]
    assert "hull" in obstacle.failures and "aabb" in obstacle.polytopes


def test_empty_frame(intrinsics):
    empty = DepthFrame(depth=np.full((240, 320), np.nan, dtype=np.float32), intrinsics=intrinsics)
    result = run_pipeline(empty)
    assert result.obstacles == [] and result.warnings


def test_config_validation():
    with pytest.raises(ValueError):
        PipelineConfig(min_range=5.0, max_range=1.0)
    with pytest.raises(ValueError):
        PipelineConfig(cluster_method="kmeans")
    with pytest.raises(ValueError):
        PipelineConfig(algorithms=("aabb", "magic"))
