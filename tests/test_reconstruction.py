import numpy as np

from polytope3d.reconstruction import depth_to_points


def test_known_pixels():
    depth = np.full((3, 3), 2.0, dtype=np.float32)
    pts = depth_to_points(depth, fx=2, fy=2, cx=1, cy=1)
    assert pts.shape == (9, 3)
    np.testing.assert_allclose(pts[4], [0, 0, 2])
    np.testing.assert_allclose(pts[5], [1, 0, 2])


def test_invalid_pixels_dropped():
    depth = np.array([[1.0, np.nan], [0.0, -1.0]], dtype=np.float32)
    pts = depth_to_points(depth, fx=1, fy=1, cx=0, cy=0)
    assert pts.shape == (1, 3)


def test_round_trip_back_to_pixels():
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 5.0, size=(4, 5)).astype(np.float32)
    fx, fy, cx, cy = 100.0, 110.0, 2.0, 1.5
    pts = depth_to_points(depth, fx, fy, cx, cy)
    v, u = np.indices(depth.shape)
    np.testing.assert_allclose(fx * pts[:, 0] / pts[:, 2] + cx, u.ravel(), atol=1e-3)
    np.testing.assert_allclose(fy * pts[:, 1] / pts[:, 2] + cy, v.ravel(), atol=1e-3)
