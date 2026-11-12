import numpy as np
import pytest

from polytope3d.polytope import DegenerateGeometry, Polytope


def unit_cube():
    A = np.vstack([np.eye(3), -np.eye(3)])
    b = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
    return Polytope(A, b, method="test")


def test_vertices_and_volume_of_cube():
    cube = unit_cube()
    assert cube.vertices().shape == (8, 3)
    np.testing.assert_allclose(cube.volume(), 1.0)


def test_contains_with_tolerance():
    cube = unit_cube()
    pts = np.array([[0.5, 0.5, 0.5], [1.0000005, 0.5, 0.5], [1.1, 0.5, 0.5]])
    assert cube.contains(pts).tolist() == [True, True, False]


def test_inflation_moves_planes_outward_and_accumulates():
    cube = unit_cube().inflated(0.1).inflated(0.05)
    assert cube.margin == pytest.approx(0.15)
    assert cube.contains(np.array([[1.14, 0.5, 0.5]])).all()
    assert not cube.contains(np.array([[1.16, 0.5, 0.5]])).any()
    with pytest.raises(ValueError):
        cube.inflated(-0.1)


def test_invalid_polytopes_rejected():
    A = np.vstack([np.eye(3), -np.eye(3)])
    with pytest.raises(ValueError):
        Polytope(2 * A, np.ones(6), method="bad")
    with pytest.raises(ValueError):
        Polytope(A, np.array([1, 1, 1, np.nan, 0, 0.0]), method="bad")
    with pytest.raises(ValueError):
        Polytope(A[:3], np.ones(3), method="bad")


def test_distance_lower_bound():
    cube = unit_cube()
    assert cube.distance_lower_bound([2.0, 0.5, 0.5]) == pytest.approx(1.0)
    corner = [2.0, 2.0, 2.0]
    assert cube.distance_lower_bound(corner) <= np.linalg.norm(np.array(corner) - 1.0)
    assert cube.distance_lower_bound([0.5, 0.5, 0.5]) < 0


def test_zero_volume_polytope_has_no_vertices():
    A = np.vstack([np.eye(3), -np.eye(3)])
    flat = Polytope(A, np.array([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]), method="flat")
    with pytest.raises(DegenerateGeometry):
        flat.vertices()
