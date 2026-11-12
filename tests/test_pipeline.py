from polytope3d.algos import ALGORITHMS
from polytope3d.clustering import cluster_indices, euclidean_labels
from polytope3d.preprocessing import fit_floor_plane, split_by_plane
from polytope3d.reconstruction import depth_to_points


def test_every_algorithm_contains_every_scene_cluster(scene_frame):
    pts = depth_to_points(scene_frame)
    above, _, _ = split_by_plane(pts, fit_floor_plane(pts))
    parts = cluster_indices(euclidean_labels(above, eps=0.1, min_points=20))
    assert len(parts) == 2
    for obstacle_id, idx in enumerate(parts):
        cluster = above[idx]
        for fit in ALGORITHMS.values():
            assert fit(cluster, obstacle_id=obstacle_id).contains(cluster).all()
