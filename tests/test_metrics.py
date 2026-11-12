import numpy as np
import pytest

from polytope3d.algos import fit_aabb, fit_obb
from polytope3d.metrics import (
    boundary_gaps,
    containment_rate,
    estimated_bytes,
    frame_to_frame,
    gap_stats,
    total_volume,
)
from polytope3d.polytope import Polytope


def unit_cube(shift=0.0):
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0]) + np.array([shift, 0, 0, -shift, 0, 0])
    return Polytope(A, b, method="test")


def test_containment_rate_single_and_union():
    pts = np.array([[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [3.0, 0.5, 0.5]])
    assert containment_rate(unit_cube(), pts) == pytest.approx(1 / 3)
    assert containment_rate([unit_cube(), unit_cube(shift=1.0)], pts) == pytest.approx(2 / 3)
    assert containment_rate(unit_cube(), np.empty((0, 3))) == 1.0


def test_boundary_gaps():
    cube = unit_cube()
    pts = np.array([[0.5, 0.5, 0.5], [0.9, 0.5, 0.5], [1.2, 0.5, 0.5]])
    np.testing.assert_allclose(boundary_gaps(cube, pts), [0.5, 0.1, -0.2])
    stats = gap_stats(cube, pts[:2])
    assert stats["max"] == pytest.approx(0.5) and stats["mean"] == pytest.approx(0.3)


def test_total_volume_and_bytes():
    assert total_volume([unit_cube(), unit_cube(shift=1.0)]) == pytest.approx(2.0)
    pts = np.random.default_rng(0).uniform(0, 1, (200, 3))
    assert estimated_bytes(fit_aabb(pts)) == 6 * 4
    assert estimated_bytes(fit_obb(pts)) == 6 * 16


def test_frame_to_frame():
    same = frame_to_frame(unit_cube(), unit_cube())
    assert same["centre_shift"] == pytest.approx(0.0, abs=1e-9)
    assert same["volume_ratio"] == pytest.approx(1.0)
    moved = frame_to_frame(unit_cube(), unit_cube(shift=0.1))
    assert moved["centre_shift"] == pytest.approx(0.1)
