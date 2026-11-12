import numpy as np
import pytest

from polytope3d.clustering import (
    cluster_indices,
    dbscan_labels,
    euclidean_labels,
    propagate_labels,
)
from polytope3d.preprocessing import fit_floor_plane, split_by_plane, voxel_downsample
from polytope3d.reconstruction import depth_to_points


def blob(center, n=1500, size=0.2, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-size / 2, size / 2, size=(n, 3)) + center).astype(np.float32)


def two_blobs_with_bridge():
    a = blob((0.0, 0.0, 0.0), seed=1)
    b = blob((1.1, 0.0, 0.0), seed=2)
    x = np.arange(0.1, 1.0, 0.03)
    bridge = np.column_stack([x, np.zeros_like(x), np.zeros_like(x)]).astype(np.float32)
    return np.vstack([a, b, bridge])


def test_two_separate_blobs():
    pts = np.vstack([blob((0, 0, 0), seed=1), blob((2, 0, 0), n=900, seed=2)])
    for fn in (euclidean_labels, dbscan_labels):
        labels = fn(pts, eps=0.05, min_points=10)
        assert labels.max() == 1 and (labels >= 0).all()
        assert np.bincount(labels).tolist() == [1500, 900]


def test_euclidean_chains_through_bridge_but_dbscan_does_not():
    pts = two_blobs_with_bridge()
    assert euclidean_labels(pts, eps=0.05, min_points=10).max() == 0
    labels = dbscan_labels(pts, eps=0.05, min_points=10)
    assert labels.max() == 1
    assert len(set(labels[:1500]) - {-1}) == 1 and len(set(labels[1500:3000]) - {-1}) == 1
    assert labels[0] != labels[1500]


def test_small_group_is_noise():
    pts = np.vstack([blob((0, 0, 0)), blob((3, 0, 0), n=5, seed=3)])
    labels = euclidean_labels(pts, eps=0.05, min_points=20)
    assert (labels[1500:] == -1).all() and labels.max() == 0


def test_isolated_point_is_noise_and_border_point_joins():
    pts = np.vstack([blob((0, 0, 0)), [[5.0, 5.0, 5.0]], [[0.13, 0.0, 0.0]]]).astype(np.float32)
    labels = dbscan_labels(pts, eps=0.05, min_points=10)
    assert labels[1500] == -1
    assert labels[0] >= 0 and labels[1501] == labels[0]


def test_empty_and_bad_arguments():
    assert len(euclidean_labels(np.empty((0, 3), dtype=np.float32))) == 0
    assert len(dbscan_labels(np.empty((0, 3), dtype=np.float32))) == 0
    with pytest.raises(ValueError):
        euclidean_labels(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        dbscan_labels(np.zeros((4, 3)), eps=0.0)


def test_cluster_indices():
    labels = np.array([0, 1, -1, 0, 1, 1])
    parts = cluster_indices(labels)
    assert [p.tolist() for p in parts] == [[0, 3], [1, 4, 5]]


def test_scene_gives_two_clusters_and_labels_propagate(scene_frame):
    pts = depth_to_points(scene_frame)
    above, _, _ = split_by_plane(pts, fit_floor_plane(pts))
    coarse = voxel_downsample(above, 0.03)
    for fn, min_points in ((euclidean_labels, 10), (dbscan_labels, 3)):
        coarse_labels = fn(coarse, eps=0.06, min_points=min_points)
        assert coarse_labels.max() == 1
        fine = propagate_labels(coarse, coarse_labels, above, max_dist=0.06)
        assert (fine >= 0).all()
        for k in range(2):
            assert (fine == k).sum() > (coarse_labels == k).sum()
