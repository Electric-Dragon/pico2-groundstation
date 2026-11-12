import numpy as np
import pytest

from polytope3d.algos import MULTI_ALGORITHMS, fit_obb, fit_split
from polytope3d.metrics import containment_rate
from polytope3d.polytope import DegenerateGeometry
from polytope3d.synthetic import Box, make_box_surface_cloud


def l_shape_cloud(n=15000):
    arm_a, arm_b = Box((0, 0, 0), (1.0, 0.2, 0.2)), Box((0, 0, 0), (0.2, 1.0, 0.2))

    def strictly_inside(p, box, eps=1e-4):
        lo, hi = np.array(box.lo), np.array(box.hi)
        return np.all((p > lo + eps) & (p < hi - eps), axis=1)

    pa = make_box_surface_cloud(arm_a, n, seed=1)
    pb = make_box_surface_cloud(arm_b, n, seed=2)
    return np.vstack([pa[~strictly_inside(pa, arm_b)], pb[~strictly_inside(pb, arm_a)]])


def box_cloud():
    return make_box_surface_cloud(Box((0, 0, 0), (1.0, 0.4, 0.6)), 8000)


def test_l_shape_is_split_and_tighter_than_one_obb():
    pts = l_shape_cloud()
    parts = fit_split(pts, obstacle_id=5)
    assert len(parts) >= 2
    assert containment_rate(parts, pts) == 1.0
    assert sum(p.volume() for p in parts) < 0.7 * fit_obb(pts).volume()


def test_ids_and_names():
    parts = fit_split(l_shape_cloud(), obstacle_id=5)
    assert [p.part_id for p in parts] == list(range(len(parts)))
    assert all(p.obstacle_id == 5 and p.method == "split_obb" for p in parts)


def test_convex_box_is_not_split():
    pts = box_cloud()
    parts = fit_split(pts)
    assert len(parts) == 1 and parts[0].contains(pts).all()


def test_max_depth_zero_gives_single_part():
    assert len(fit_split(l_shape_cloud(), max_depth=0)) == 1


def test_more_depth_never_increases_part_volume_much():
    pts = l_shape_cloud()
    shallow = sum(p.volume() for p in fit_split(pts, max_depth=1))
    deep = sum(p.volume() for p in fit_split(pts, max_depth=3))
    assert deep <= shallow + 1e-9


def test_registry_and_bad_arguments():
    assert "split_obb" in MULTI_ALGORITHMS
    with pytest.raises(ValueError):
        fit_split(box_cloud(), max_depth=-1)
    with pytest.raises(ValueError):
        fit_split(box_cloud(), improvement=1.5)
    with pytest.raises(DegenerateGeometry):
        fit_split(box_cloud()[:3])
