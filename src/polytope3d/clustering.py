"""Group obstacle points into clusters. Labels are integers per point; -1 means noise."""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _check(points, eps, min_points):
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if eps <= 0 or min_points < 1:
        raise ValueError("need eps > 0 and min_points >= 1")


def _components(n, i, j):
    """Connected-component label of each of n nodes, given edges (i[k], j[k])."""
    graph = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    return connected_components(graph, directed=False)[1]


def _relabel_by_size(labels, min_size):
    """Renumber clusters 0..K-1 from largest to smallest; clusters below min_size become -1."""
    valid = labels >= 0
    if not valid.any():
        return np.full(len(labels), -1, dtype=np.int64)
    counts = np.bincount(labels[valid])
    keep = np.flatnonzero(counts >= min_size)
    order = keep[np.argsort(-counts[keep], kind="stable")]
    mapping = np.full(len(counts), -1, dtype=np.int64)
    mapping[order] = np.arange(len(order))
    return np.where(valid, mapping[np.where(valid, labels, 0)], -1)


def euclidean_labels(points, eps=0.05, min_points=20):
    """Euclidean clustering: points closer than eps are linked; linked groups are clusters.

    Groups smaller than min_points are labelled -1 (noise).
    """
    _check(points, eps, min_points)
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    labels = _components(n, pairs[:, 0], pairs[:, 1])
    return _relabel_by_size(labels, min_points)


def dbscan_labels(points, eps=0.05, min_points=10):
    """DBSCAN: only dense (core) points can connect clusters, so thin bridges do not merge them.

    A core point has at least min_points points (itself included) within eps. Clusters are the
    connected groups of core points. Non-core points within eps of a core point join the
    cluster of their nearest core point; all others are noise (-1).
    """
    _check(points, eps, min_points)
    n = len(points)
    if n == 0:
        return np.empty(0, dtype=np.int64)
    tree = cKDTree(points)
    core = tree.query_ball_point(points, eps, return_length=True) >= min_points
    pairs = tree.query_pairs(eps, output_type="ndarray")
    i, j = pairs[:, 0], pairs[:, 1]

    both_core = core[i] & core[j]
    labels = np.full(n, -1, dtype=np.int64)
    if core.any():
        comp = _components(n, i[both_core], j[both_core])
        labels[core] = comp[core]

    mixed = core[i] != core[j]
    if mixed.any():
        c = np.where(core[i[mixed]], i[mixed], j[mixed])
        b = np.where(core[i[mixed]], j[mixed], i[mixed])
        dist = np.linalg.norm(points[c] - points[b], axis=1)
        order = np.lexsort((dist, b))
        b, c = b[order], c[order]
        first = np.r_[True, b[1:] != b[:-1]]
        labels[b[first]] = labels[c[first]]
    return _relabel_by_size(labels, 1)


def cluster_indices(labels):
    """List of index arrays, one per cluster (label 0, 1, 2, ...), noise excluded."""
    n_clusters = labels.max() + 1 if len(labels) else 0
    return [np.flatnonzero(labels == k) for k in range(n_clusters)]


def propagate_labels(coarse_points, coarse_labels, fine_points, max_dist):
    """Give each fine point the label of its nearest coarse point, or -1 if farther than max_dist.

    Use it to carry labels from a downsampled cloud back to the full-resolution cloud.
    """
    if len(coarse_points) == 0:
        return np.full(len(fine_points), -1, dtype=np.int64)
    dist, idx = cKDTree(coarse_points).query(fine_points)
    return np.where(dist <= max_dist, coarse_labels[idx], -1)
