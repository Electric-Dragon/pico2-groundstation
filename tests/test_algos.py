import numpy as np
import pytest

from polytope3d.algos import ALGORITHMS, fit_aabb, fit_hull, fit_obb
from polytope3d.polytope import DegenerateGeometry
from polytope3d.synthetic import Box, make_box_surface_cloud

BOX = Box((-0.5, -0.2, -0.3), (0.5, 0.2, 0.3))


def rotated_box_cloud(angle_deg=30.0, n=20000):
    pts = make_box_surface_cloud(BOX, n).astype(np.float64)
    a = np.radians(angle_deg)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    return (pts @ rot.T + np.array([0.0, 0.5, 3.0])).astype(np.float32)


@pytest.mark.parametrize("name", list(ALGORITHMS))
def test_every_algorithm_contains_all_points(name):
    pts = rotated_box_cloud()
    poly = ALGORITHMS[name](pts, obstacle_id=7)
    assert poly.contains(pts).all()
    assert poly.method == name and poly.obstacle_id == 7 and poly.n_points == len(pts)
    assert np.isfinite(poly.A).all() and np.isfinite(poly.b).all()
    assert poly.volume() > 0


def test_aabb_of_axis_aligned_box_is_exact():
    pts = make_box_surface_cloud(BOX, 20000)
    poly = fit_aabb(pts)
    assert poly.n_planes == 6
    np.testing.assert_allclose(poly.volume(), 1.0 * 0.4 * 0.6, rtol=0.01)


def test_aabb_widens_flat_cluster():
    pts = np.column_stack([np.random.default_rng(0).uniform(0, 1, (50, 2)), np.full(50, 2.0)])
    poly = fit_aabb(pts[:, [0, 1, 2]], min_extent=0.02)
    assert poly.volume() > 0 and poly.contains(pts).all()


def test_obb_tight_on_rotated_box_and_aabb_is_not():
    pts = rotated_box_cloud(30.0)
    true_volume = 1.0 * 0.4 * 0.6
    assert fit_obb(pts).volume() == pytest.approx(true_volume, rel=0.05)
    assert fit_aabb(pts).volume() > 1.3 * true_volume


def test_obb_axes_do_not_depend_on_point_order():
    pts = rotated_box_cloud()
    shuffled = pts[np.random.default_rng(1).permutation(len(pts))]
    np.testing.assert_allclose(fit_obb(pts).A, fit_obb(shuffled).A, atol=1e-5)


def test_hull_is_no_larger_than_obb_or_aabb():
    pts = rotated_box_cloud()
    hull_v = fit_hull(pts).volume()
    assert hull_v <= fit_obb(pts).volume() + 1e-9
    assert hull_v <= fit_aabb(pts).volume() + 1e-9


def test_hull_merges_duplicate_facets():
    corners = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    poly = fit_hull(corners)
    assert poly.n_planes == 6
    np.testing.assert_allclose(poly.volume(), 1.0)


def test_degenerate_inputs():
    flat = np.column_stack([np.random.default_rng(0).uniform(0, 1, (50, 2)), np.zeros(50)])
    with pytest.raises(DegenerateGeometry):
        fit_hull(flat)
    with pytest.raises(DegenerateGeometry):
        fit_hull(flat[:3])
    with pytest.raises(DegenerateGeometry):
        fit_obb(flat[:3])
    assert fit_obb(flat).volume() > 0
    with pytest.raises(ValueError):
        fit_aabb(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        fit_aabb(np.array([[0.0, 0.0, np.nan]]))