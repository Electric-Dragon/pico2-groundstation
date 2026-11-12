import pytest

from polytope3d.contracts import Intrinsics
from polytope3d.synthetic import Box, render_depth


@pytest.fixture
def intrinsics():
    return Intrinsics(fx=300.0, fy=300.0, cx=160.0, cy=120.0, width=320, height=240)


@pytest.fixture
def boxes():
    return [Box((-0.5, 0.5, 2.0), (0.5, 1.0, 3.0)), Box((1.0, 0.6, 3.5), (1.6, 1.0, 4.0))]


@pytest.fixture
def scene_frame(intrinsics, boxes):
    return render_depth(intrinsics, cam_height=1.0, boxes=boxes)
