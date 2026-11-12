from polytope3d.clustering import euclidean_labels, group_clusters
from polytope3d.polytopes import aabb, contains, convex_hull
from polytope3d.preprocessing import fit_floor_plane, split_by_plane
from polytope3d.reconstruction import depth_to_points


def test_aabb_and_hull_contain_cluster(scene_frame):
    pts = depth_to_points(scene_frame)
    above, _, _ = split_by_plane(pts, fit_floor_plane(pts))
    for cluster in group_clusters(above, euclidean_labels(above, 0.1)):
        A, b = aabb(cluster)
        assert contains(A, b, cluster).all()
        A, b, _ = convex_hull(cluster)
        assert contains(A, b, cluster).all()