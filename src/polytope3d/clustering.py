"""Split a point cloud into obstacle clusters.

Every method returns an int32 label per point: 0, 1, 2, ... for clusters and -1 for noise.
`group_clusters` turns labels into a list of (N_j, 3) arrays, largest cluster first.
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _components(points, eps):
    """Connected components of the graph that links every pair of points closer than eps."""
    n = len(points)
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(graph, directed=False)[1].astype(np.int32)


def euclidean_labels(points, eps):
    """Euclidean clustering: points closer than eps (metres) end up in the same cluster."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    if len(points) == 0:
        return np.empty(0, dtype=np.int32)
    return _components(points, eps)


def dbscan_labels(points, eps, min_samples):
    """DBSCAN. A core point has >= min_samples points (itself included) within eps.

    Clusters are the connected groups of core points; a non-core point within eps of a core
    point joins that cluster (a border point); everything else is noise (-1).
    """
    if eps <= 0 or min_samples < 1:
        raise ValueError("need eps > 0 and min_samples >= 1")
    labels = np.full(len(points), -1, dtype=np.int32)
    if len(points) == 0:
        return labels
    counts = cKDTree(points).query_ball_point(points, eps, return_length=True)
    core_idx = np.flatnonzero(counts >= min_samples)
    if len(core_idx) == 0:
        return labels
    core_tree = cKDTree(points[core_idx])
    core_labels = _components(points[core_idx], eps)
    labels[core_idx] = core_labels
    rest = np.flatnonzero(counts < min_samples)
    if len(rest):
        dist, nearest = core_tree.query(points[rest], k=1, distance_upper_bound=eps)
        joined = np.isfinite(dist)
        labels[rest[joined]] = core_labels[nearest[joined]]
    return labels


def group_clusters(points, labels, min_points=20):
    """List of (N_j, 3) arrays, one per cluster with >= min_points points, largest first."""
    if len(points) != len(labels):
        raise ValueError("points and labels must have the same length")
    clustered = labels >= 0
    if not clustered.any():
        return []
    ids, counts = np.unique(labels[clustered], return_counts=True)
    big = counts >= min_points
    ids, counts = ids[big], counts[big]
    order = np.argsort(-counts, kind="stable")
    return [points[labels == i] for i in ids[order]]
