import numpy as np
import pytest

from polytope3d.contracts import DepthFrame, Intrinsics

K = Intrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)


def test_camera_matrix():
    expected = [[300, 0, 160], [0, 300, 120], [0, 0, 1]]
    np.testing.assert_allclose(K.matrix(), expected)


def test_bad_intrinsics_rejected():
    with pytest.raises(ValueError):
        Intrinsics(fx=0.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)


def test_valid_frame_and_mask():
    depth = np.full((240, 320), 2.0, dtype=np.float32)
    depth[0, 0] = np.nan
    depth[0, 1] = 0.0
    frame = DepthFrame(depth=depth, intrinsics=K)
    assert frame.valid_mask.sum() == 240 * 320 - 2


def test_wrong_dtype_rejected():
    with pytest.raises(ValueError):
        DepthFrame(depth=np.ones((240, 320), dtype=np.float64), intrinsics=K)


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        DepthFrame(depth=np.ones((320, 240), dtype=np.float32), intrinsics=K)
