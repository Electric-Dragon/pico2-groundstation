import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def euclidean_clusters(points, eps=0.05, min_points=20):
    """Group points: two points are linked if closer than `eps` metres.

    Returns a list of (N_j, 3) arrays, one per cluster with >= min_points points.
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"
    if len(points) == 0:
        return []
    pairs = cKDTree(points).query_pairs(eps, output_type="ndarray")
    n = len(points)
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, labels = connected_components(graph, directed=False)
    clusters = [points[labels == k] for k in range(labels.max() + 1)]
    return [c for c in clusters if len(c) >= min_points]