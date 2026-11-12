import numpy as np
import pytest

from polytope3d.algos import ALGORITHMS, fit_aabb, fit_hull, fit_kdop14, fit_kdop26
from polytope3d.algos.kdop import AXES_7, fit_kdop, make_directions
from polytope3d.synthetic import Box, make_box_surface_cloud


def sphere_cloud(n=5000, radius=1.0, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return (radius * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def rotated_box_cloud(angle_deg=30.0, n=20000):
    pts = make_box_surface_cloud(Box((-0.5, -0.2, -0.3), (0.5, 0.2, 0.3)), n).astype(np.float64)
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (pts @ rot.T + np.array([0.0, 0.5, 3.0])).astype(np.float32)


def test_direction_sets():
    d = make_directions(AXES_7)
    assert d.shape == (14, 3)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    np.testing.assert_allclose(d[0::2], -d[1::2])
    expected = [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]
    np.testing.assert_allclose(d[:6], expected)


def test_plane_counts():
    pts = rotated_box_cloud()
    assert fit_kdop14(pts).n_planes == 14
    assert fit_kdop26(pts).n_planes == 26


def test_registry_entries_contain_points_and_carry_their_name():
    pts = rotated_box_cloud()
    for name in ("kdop14", "kdop26"):
        poly = ALGORITHMS[name](pts, obstacle_id=3)
        assert poly.method == name and poly.obstacle_id == 3
        assert poly.contains(pts).all() and poly.volume() > 0


def test_more_directions_are_never_looser():
    for pts in (sphere_cloud(), rotated_box_cloud()):
        volumes = [
            fit_aabb(pts).volume(),
            fit_kdop14(pts).volume(),
            fit_kdop26(pts).volume(),
            fit_hull(pts).volume(),
        ]
        assert all(a >= b - 1e-9 for a, b in zip(volumes, volumes[1:]))


def test_sphere_gets_strictly_tighter_with_more_directions():
    pts = sphere_cloud()
    assert fit_kdop26(pts).volume() < fit_kdop14(pts).volume() < fit_aabb(pts).volume()


def test_kdop_matches_aabb_when_given_only_coordinate_axes():
    pts = rotated_box_cloud()
    only_xyz = fit_kdop(pts, axes=((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert only_xyz.n_planes == 6
    np.testing.assert_allclose(only_xyz.volume(), fit_aabb(pts).volume())


def test_flat_cluster_is_widened_and_contained():
    flat = np.column_stack([np.random.default_rng(0).uniform(0, 1, (80, 2)), np.full(80, 2.0)])
    poly = fit_kdop14(flat, min_extent=0.02)
    assert poly.contains(flat).all() and poly.volume() > 0


def test_bad_axes_rejected():
    with pytest.raises(ValueError):
        make_directions(((0, 1, 0), (1, 0, 0), (0, 0, 1)))
    with pytest.raises(ValueError):
        make_directions(((1, 0, 0), (0, 1, 0)))
