import numpy as np
import pytest

from polytope3d.contracts import Plane


def test_signed_distance():
    plane = Plane(normal=np.array([0.0, -1.0, 0.0]), d=1.0)
    pts = np.array([[0, 1.0, 5], [0, 0.0, 5], [0, 2.0, 5]], dtype=np.float32)
    np.testing.assert_allclose(plane.signed_distance(pts), [0.0, 1.0, -1.0])


def test_non_unit_normal_rejected():
    with pytest.raises(ValueError):
        Plane(normal=np.array([0.0, -2.0, 0.0]), d=1.0)
