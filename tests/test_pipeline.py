from polytope3d.clustering import cluster_indices, euclidean_labels
from polytope3d.polytopes import aabb, contains, convex_hull
from polytope3d.preprocessing import fit_floor_plane, split_by_plane
from polytope3d.reconstruction import depth_to_points


def test_aabb_and_hull_contain_cluster(scene_frame):
    pts = depth_to_points(scene_frame)
    above, _, _ = split_by_plane(pts, fit_floor_plane(pts))
    labels = euclidean_labels(above, eps=0.1, min_points=20)
    parts = cluster_indices(labels)
    assert len(parts) == 2
    for idx in parts:
        cluster = above[idx]
        A, b = aabb(cluster)
        assert contains(A, b, cluster).all()
        A, b, _ = convex_hull(cluster)
        assert contains(A, b, cluster).all()
