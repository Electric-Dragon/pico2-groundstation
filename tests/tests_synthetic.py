import numpy as np

from polytope3d.synthetic import make_box_cloud


def test_shape_and_dtype():
    pts = make_box_cloud((0, 0, 2), (1, 1, 1), 500)
    assert pts.shape == (500, 3)
    assert pts.dtype == np.float32


def test_points_inside_box():
    pts = make_box_cloud((0, 0, 2), (1, 2, 3), 500)
    assert np.all(pts >= np.array([-0.5, -1.0, 0.5]) - 1e-6)
    assert np.all(pts <= np.array([0.5, 1.0, 3.5]) + 1e-6)
