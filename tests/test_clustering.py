import numpy as np
import pytest
from scipy.sparse.csgraph import connected_components

from polytope3d.clustering import dbscan_labels, euclidean_labels, group_clusters
from polytope3d.preprocessing import fit_floor_plane, split_by_plane
from polytope3d.reconstruction import depth_to_points


def two_blobs_with_bridge():
    rng = np.random.default_rng(0)
    left = rng.uniform(0.0, 0.4, size=(4000, 3))
    right = rng.uniform(0.0, 0.4, size=(4000, 3)) + [1.0, 0.0, 0.0]
    bridge = np.column_stack([np.linspace(0.42, 0.98, 15), np.full(15, 0.2), np.full(15, 0.2)])
    return np.vstack([left, right, bridge]).astype(np.float32)


def test_bridge_merges_euclidean_but_not_dbscan():
    pts = two_blobs_with_bridge()
    assert len(group_clusters(pts, euclidean_labels(pts, 0.05), 100)) == 1
    assert len(group_clusters(pts, dbscan_labels(pts, 0.05, 15), 100)) == 2


def test_dbscan_matches_brute_force():
    rng = np.random.default_rng(1)
    pts = np.vstack(
        [rng.normal(0, 0.1, (150, 3)), rng.normal(2, 0.1, (150, 3)), rng.uniform(-1, 3, (30, 3))]
    )
    eps, m = 0.15, 5
    labels = dbscan_labels(pts, eps, m)
    dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    adj = dist <= eps
    core = adj.sum(axis=1) >= m
    noise = ~core & ~(adj[:, core]).any(axis=1)
    np.testing.assert_array_equal(labels == -1, noise)
    same = labels[:, None] == labels[None, :]
    comp = connected_components(adj[np.ix_(core, core)], directed=False)[1]
    np.testing.assert_array_equal(same[np.ix_(core, core)], comp[:, None] == comp[None, :])


def test_group_clusters_sorted_and_filtered():
    pts = np.zeros((10, 3), dtype=np.float32)
    labels = np.array([1, 1, 1, 0, 0, -1, 2, 1, 1, 1], dtype=np.int32)
    groups = group_clusters(pts, labels, min_points=2)
    assert [len(g) for g in groups] == [6, 2]


def test_empty_input():
    empty = np.empty((0, 3), dtype=np.float32)
    assert euclidean_labels(empty, 0.1).shape == (0,)
    assert dbscan_labels(empty, 0.1, 3).shape == (0,)
    assert group_clusters(empty, np.empty(0, dtype=np.int32)) == []


def test_bad_parameters():
    pts = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        euclidean_labels(pts, 0.0)
    with pytest.raises(ValueError):
        dbscan_labels(pts, 0.1, 0)


@pytest.mark.parametrize("method", ["euclidean", "dbscan"])
def test_scene_has_two_obstacles(scene_frame, method):
    pts = depth_to_points(scene_frame)
    above, _, _ = split_by_plane(pts, fit_floor_plane(pts))
    if method == "euclidean":
        labels = euclidean_labels(above, 0.1)
    else:
        labels = dbscan_labels(above, 0.1, 10)
    clusters = group_clusters(above, labels, min_points=50)
    assert len(clusters) == 2
    assert len(clusters[0]) > len(clusters[1])
