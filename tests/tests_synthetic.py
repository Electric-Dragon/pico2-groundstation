import numpy as np
import pytest

from polytope3d.synthetic import Box, make_box_surface_cloud, render_depth


def test_box_validation():
    with pytest.raises(ValueError):
        Box((0, 0, 0), (1, 0, 1))


def test_surface_cloud_lies_on_surface():
    box = Box((-0.5, -1.0, 0.5), (0.5, 1.0, 3.5))
    pts = make_box_surface_cloud(box, 2000)
    lo, hi = np.array(box.lo), np.array(box.hi)
    assert pts.shape == (2000, 3) and pts.dtype == np.float32
    assert np.all(pts >= lo - 1e-6) and np.all(pts <= hi + 1e-6)
    on_face = np.isclose(pts, lo, atol=1e-6) | np.isclose(pts, hi, atol=1e-6)
    assert on_face.any(axis=1).all()


def test_sky_is_nan(intrinsics, boxes):
    frame = render_depth(intrinsics, 1.0, boxes)
    assert np.isnan(frame.depth[0, 0])


def test_floor_depth_bottom_left(intrinsics, boxes):
    frame = render_depth(intrinsics, 1.0, boxes)
    np.testing.assert_allclose(frame.depth[239, 0], 1.0 / (119 / 300), rtol=1e-4)


def test_box_front_face_depth(intrinsics, boxes):
    frame = render_depth(intrinsics, 1.0, boxes)
    np.testing.assert_allclose(frame.depth[232, 160], 2.0, rtol=1e-4)


def test_box_with_corner_at_zero(intrinsics):
    frame = render_depth(intrinsics, 1.0, [Box((0.0, 0.5, 2.0), (0.5, 1.0, 3.0))])
    np.testing.assert_allclose(frame.depth[232, 170], 2.0, rtol=1e-4)
    assert frame.depth[232, 150] > 2.1


def test_noise_grows_with_distance(intrinsics, boxes):
    clean = render_depth(intrinsics, 1.0, boxes).depth
    noisy = render_depth(intrinsics, 1.0, boxes, noise_k=0.005).depth
    diff = noisy - clean
    near = np.isfinite(diff) & (clean < 3.0)
    far = np.isfinite(diff) & (clean > 6.0) & (clean < 30.0)
    assert diff[far].std() > 3 * diff[near].std()


def test_dropout_fraction(intrinsics):
    clean = render_depth(intrinsics, 1.0).valid_mask.sum()
    holey = render_depth(intrinsics, 1.0, dropout=0.1, seed=1).valid_mask.sum()
    assert abs(1 - holey / clean - 0.1) < 0.02
