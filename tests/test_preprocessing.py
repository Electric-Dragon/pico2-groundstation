import numpy as np
import pytest

from polytope3d.preprocessing import (
    crop_to_workspace,
    fit_floor_plane,
    remove_statistical_outliers,
    split_by_plane,
    voxel_downsample,
)
from polytope3d.reconstruction import depth_to_points
from polytope3d.synthetic import render_depth


def test_crop_to_workspace():
    pts = np.array([[0, 0, 1], [0, 0, 20], [5, 0, 1]], dtype=np.float32)
    kept = crop_to_workspace(pts, (-1, -1, 0), (1, 1, 10))
    assert kept.shape == (1, 3)


def test_voxel_centroids():
    pts = np.array([[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [5.1, 5.1, 5.1]], dtype=np.float32)
    out = voxel_downsample(pts, 1.0)
    assert out.dtype == np.float32 and out.shape == (2, 3)
    expected = np.array([[0.2, 0.2, 0.2], [5.1, 5.1, 5.1]])
    np.testing.assert_allclose(out[np.argsort(out[:, 0])], expected, atol=1e-6)


def test_voxel_reduces_scene(scene_frame):
    pts = depth_to_points(scene_frame)
    assert len(voxel_downsample(pts, 0.05)) < len(pts) / 2


def test_voxel_bad_size():
    with pytest.raises(ValueError):
        voxel_downsample(np.zeros((3, 3), dtype=np.float32), 0.0)


def test_statistical_outliers_removed():
    rng = np.random.default_rng(0)
    cluster = rng.uniform(0, 1, size=(1000, 3)).astype(np.float32)
    far = np.array([[10, 10, 10], [-10, 10, 10], [10, -10, 10], [10, 10, -10], [-10, -10, -10]])
    out = remove_statistical_outliers(np.vstack([cluster, far.astype(np.float32)]))
    assert len(out) <= 1000 and out.max() <= 1.0 + 1e-6 and out.min() >= -1e-6


def test_floor_plane_on_clean_scene(scene_frame):
    pts = depth_to_points(scene_frame)
    plane = fit_floor_plane(pts)
    np.testing.assert_allclose(plane.normal, [0, -1, 0], atol=1e-3)
    np.testing.assert_allclose(plane.d, 1.0, atol=1e-3)
    above, on, below = split_by_plane(pts, plane)
    assert len(below) == 0
    assert len(on) > len(above) > 0
    assert above[:, 1].max() < 0.98 + 1e-4


def test_wall_not_mistaken_for_floor():
    rng = np.random.default_rng(0)
    wall = np.column_stack([rng.uniform(-2, 2, 2000), rng.uniform(-1, 1, 2000), np.full(2000, 3.0)])
    floor = np.column_stack([rng.uniform(-2, 2, 800), np.full(800, 1.0), rng.uniform(1, 3, 800)])
    pts = np.vstack([wall, floor]).astype(np.float32)
    plane = fit_floor_plane(pts)
    np.testing.assert_allclose(plane.normal, [0, -1, 0], atol=1e-3)
    with pytest.raises(ValueError):
        fit_floor_plane(wall.astype(np.float32))


def test_floor_plane_with_noise(intrinsics, boxes):
    frame = render_depth(intrinsics, 1.0, boxes, noise_k=0.002)
    pts = depth_to_points(frame, max_range=6.0)
    plane = fit_floor_plane(pts, threshold=0.02, threshold_per_z2=0.006)
    tilt = np.degrees(np.arccos(abs(plane.normal @ np.array([0, -1.0, 0]))))
    assert tilt < 2.0
