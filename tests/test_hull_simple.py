import numpy as np
import pytest

from polytope3d.algos import fit_aabb, fit_hull, fit_hull_simplified
from polytope3d.algos.hull_simple import _is_bounded
from polytope3d.polytope import DegenerateGeometry
from polytope3d.synthetic import Box, make_box_surface_cloud


def sphere_cloud(n=6000, radius=0.5, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return (radius * v / np.linalg.norm(v, axis=1, keepdims=True) + [0.0, 0.5, 3.0]).astype(
        np.float32
    )


def rotated_box_cloud(angle_deg=30.0, n=6000):
    pts = make_box_surface_cloud(Box((-0.5, -0.2, -0.3), (0.5, 0.2, 0.3)), n).astype(np.float64)
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (pts @ rot.T + np.array([0.0, 0.5, 3.0])).astype(np.float32)


@pytest.mark.parametrize("cloud", [sphere_cloud, rotated_box_cloud])
def test_contains_all_points_with_few_planes(cloud):
    pts = cloud()
    poly = fit_hull_simplified(pts, max_planes=16)
    assert poly.contains(pts).all()
    assert poly.n_planes <= 16
    assert poly.method == "hull_simple"
    assert poly.volume() >= fit_hull(pts).volume() - 1e-9


def test_big_reduction_on_sphere():
    pts = sphere_cloud()
    assert fit_hull(pts).n_planes > 1000
    assert fit_hull_simplified(pts, max_planes=16).n_planes <= 16


def test_close_to_exact_hull_on_box_and_better_than_aabb():
    pts = rotated_box_cloud()
    hull_v = fit_hull(pts).volume()
    simple_v = fit_hull_simplified(pts, max_planes=16).volume()
    assert simple_v < 1.2 * hull_v
    assert simple_v < fit_aabb(pts).volume()


def test_sphere_beats_aabb_and_is_deterministic():
    pts = sphere_cloud()
    a = fit_hull_simplified(pts, max_planes=16)
    b = fit_hull_simplified(pts, max_planes=16)
    np.testing.assert_array_equal(a.A, b.A)
    np.testing.assert_array_equal(a.b, b.b)
    assert a.volume() < fit_aabb(pts).volume()


def test_small_hull_is_returned_unchanged_in_volume():
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    poly = fit_hull_simplified(corners, max_planes=16)
    np.testing.assert_allclose(poly.volume(), 1.0, rtol=1e-6)


def test_boundedness_check():
    assert _is_bounded(np.vstack([np.eye(3), -np.eye(3)]))
    assert not _is_bounded(
        np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1, 1, 1] / np.sqrt(3)])
    )


def test_bad_arguments_and_degenerate_input():
    with pytest.raises(ValueError):
        fit_hull_simplified(sphere_cloud(), max_planes=3)
    flat = np.column_stack([np.random.default_rng(0).uniform(0, 1, (50, 2)), np.zeros(50)])
    with pytest.raises(DegenerateGeometry):
        fit_hull_simplified(flat)
