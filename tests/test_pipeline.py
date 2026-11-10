import numpy as np

from polytope3d.clustering import euclidean_clusters
from polytope3d.polytopes import aabb, contains, convex_hull
from polytope3d.preprocessing import remove_plane_ransac
from polytope3d.reconstruction import depth_to_points
from polytope3d.synthetic import make_box_cloud, render_depth

CAM = dict(fx=300.0, fy=300.0, cx=160.0, cy=120.0)
BOXES = [((-0.5, 0.5, 2.0), (0.5, 1.0, 3.0)), ((1.0, 0.6, 3.5), (1.6, 1.0, 4.0))]


def scene_points():
    depth = render_depth(240, 320, **CAM, cam_height=1.0, boxes=BOXES)
    return depth_to_points(depth, **CAM)


def test_ransac_finds_floor():
    rest, floor, (normal, _) = remove_plane_ransac(scene_points())
    assert abs(abs(normal[1]) - 1.0) < 1e-2
    assert len(floor) > len(rest) > 0


def test_two_boxes_two_clusters():
    rest, _, _ = remove_plane_ransac(scene_points())
    assert len(euclidean_clusters(rest, eps=0.1)) == 2


def test_aabb_and_hull_contain_cluster():
    rest, _, _ = remove_plane_ransac(scene_points())
    for cluster in euclidean_clusters(rest, eps=0.1):
        A, b = aabb(cluster)
        assert contains(A, b, cluster).all()
        A, b, _ = convex_hull(cluster)
        assert contains(A, b, cluster).all()


def test_hull_of_box_cloud_is_tighter_than_nothing():
    pts = make_box_cloud((0, 0, 2), (1, 1, 1), 500)
    A, b, verts = convex_hull(pts)
    assert contains(A, b, pts).all()
    assert len(verts) >= 4
