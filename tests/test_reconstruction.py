import numpy as np
import pytest

from polytope3d.contracts import DepthFrame, Intrinsics
from polytope3d.reconstruction import _ray_grid, depth_to_points, project_points

K3 = Intrinsics(fx=2.0, fy=2.0, cx=1.0, cy=1.0, width=3, height=3)


def frame_of(depth, intrinsics=K3):
    return DepthFrame(depth=np.asarray(depth, dtype=np.float32), intrinsics=intrinsics)


def test_known_pixels():
    pts = depth_to_points(frame_of(np.full((3, 3), 2.0)))
    assert pts.shape == (9, 3)
    assert pts.dtype == np.float32
    np.testing.assert_allclose(pts[4], [0, 0, 2])
    np.testing.assert_allclose(pts[5], [1, 0, 2])


def test_invalid_pixels_dropped():
    depth = np.full((3, 3), np.nan)
    depth[0, 0] = 1.0
    depth[1, 1] = 0.0
    depth[2, 2] = -1.0
    assert depth_to_points(frame_of(depth)).shape == (1, 3)


def test_range_limits():
    depth = np.full((3, 3), np.nan)
    depth[0, 0], depth[0, 1], depth[0, 2] = 0.05, 1.0, 20.0
    pts = depth_to_points(frame_of(depth), min_range=0.1, max_range=10.0)
    assert pts.shape == (1, 3)
    np.testing.assert_allclose(pts[0, 2], 1.0)


def test_bad_range_rejected():
    with pytest.raises(ValueError):
        depth_to_points(frame_of(np.ones((3, 3))), min_range=5.0, max_range=1.0)


def test_round_trip_back_to_pixels(intrinsics):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 5.0, size=(240, 320)).astype(np.float32)
    pts = depth_to_points(frame_of(depth, intrinsics))
    uv = project_points(pts, intrinsics)
    v, u = np.indices(depth.shape)
    np.testing.assert_allclose(uv[:, 0], u.ravel(), atol=1e-2)
    np.testing.assert_allclose(uv[:, 1], v.ravel(), atol=1e-2)


def test_flat_wall_range_exceeds_depth_at_edges(intrinsics):
    pts = depth_to_points(frame_of(np.full((240, 320), 2.0), intrinsics))
    np.testing.assert_allclose(pts[:, 2], 2.0)
    assert np.linalg.norm(pts, axis=1).max() > 2.2


def test_ray_grid_is_cached_and_read_only(intrinsics):
    x_n, _ = _ray_grid(intrinsics)
    assert _ray_grid(intrinsics)[0] is x_n
    with pytest.raises(ValueError):
        x_n[0, 0] = 1.0
