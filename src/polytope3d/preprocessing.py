import numpy as np


def remove_plane_ransac(points, threshold=0.02, iterations=200, seed=0):
    """Find the dominant plane (the floor) with RANSAC and split it from the rest.

    Returns (rest, plane_points, (normal, d)) where the plane is normal . x + d = 0
    and `normal` has unit length. `threshold` is the max distance (metres) for a
    point to count as lying on the plane.
    """
    assert points.ndim == 2 and points.shape[1] == 3, "points must be (N, 3)"
    assert len(points) >= 3, "need at least 3 points"
    rng = np.random.default_rng(seed)
    best_mask, best_model = None, None
    for _ in range(iterations):
        a, b, c = points[rng.choice(len(points), size=3, replace=False)].astype(np.float64)
        normal = np.cross(b - a, c - a)
        length = np.linalg.norm(normal)
        if length < 1e-9:
            continue
        normal /= length
        d = -normal @ a
        mask = np.abs(points @ normal + d) < threshold
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask, best_model = mask, (normal, d)
    assert best_mask is not None, "no valid plane found (all samples collinear)"
    return points[~best_mask], points[best_mask], best_model