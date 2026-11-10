import numpy as np

from polytope3d.reconstruction import depth_to_points
from polytope3d.synthetic import render_depth

CAM = dict(height=240, width=320, fx=300.0, fy=300.0, cx=160.0, cy=120.0)
BOX = ((-0.5, 0.5, 2.0), (0.5, 1.0, 3.0))


def render():
    return render_depth(**CAM, cam_height=1.0, boxes=[BOX])


def test_sky_is_nan():
    assert np.isnan(render()[0, 0])


def test_floor_depth_at_bottom_left():
    # dy = (239 - 120) / 300, so depth = 1.0 / dy
    np.testing.assert_allclose(render()[239, 0], 1.0 / (119 / 300), rtol=1e-4)


def test_box_front_face_depth():
    np.testing.assert_allclose(render()[232, 160], 2.0, rtol=1e-4)


def test_cloud_matches_scene():
    d = render()
    pts = depth_to_points(d, 300.0, 300.0, 160.0, 120.0)
    assert np.all(pts[:, 1] <= 1.0 + 1e-4)
    box_pts = pts[pts[:, 1] < 0.999]
    assert len(box_pts) > 0
    assert box_pts[:, 2].min() >= 2.0 - 1e-3
    assert box_pts[:, 2].max() <= 3.0 + 1e-3