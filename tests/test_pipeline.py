import numpy as np

from polytope3d.clustering import euclidean_clusters
from polytope3d.polytopes import aabb, contains, convex_hull
from polytope3d.preprocessing import remove_plane_ransac
from polytope3d.reconstruction import depth_to_points
from polytope3d.synthetic import Box, make_box_surface_cloud


def test_ransac_finds_floor(scene_frame):
    rest, floor, (normal, _) = remove_plane_ransac(depth_to_points(scene_frame))
    assert abs(abs(normal[1]) - 1.0) < 1e-2
    assert len(floor) > len(rest) > 0


def test_two_boxes_two_clusters(scene_frame):
    rest, _, _ = remove_plane_ransac(depth_to_points(scene_frame))
    assert len(euclidean_clusters(rest, eps=0.1)) == 2


def test_aabb_and_hull_contain_cluster(scene_frame):
    rest, _, _ = remove_plane_ransac(depth_to_points(scene_frame))
    for cluster in euclidean_clusters(rest, eps=0.1):
        A, b = aabb(cluster)
        assert contains(A, b, cluster).all()
        A, b, _ = convex_hull(cluster)
        assert contains(A, b, cluster).all()


def test_hull_of_box_surface_cloud():
    pts = make_box_surface_cloud(Box((-0.5, -0.5, 1.5), (0.5, 0.5, 2.5)), 500)
    A, b, verts = convex_hull(pts)
    assert contains(A, b, pts).all()
    assert len(verts) >= 4
