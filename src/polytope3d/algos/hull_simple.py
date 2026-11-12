"""Conservative simplified hull: few planes, offsets recomputed so every point stays inside."""

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from polytope3d.polytope import DegenerateGeometry, Polytope, check_points


def _spherical_kmeans(normals, weights, k, iterations=25):
    """Cluster unit vectors by angle (area-weighted). Deterministic: no random start."""
    centers = [normals[np.argmax(weights)]]
    for _ in range(k - 1):
        similarity = (normals @ np.array(centers).T).max(axis=1)
        centers.append(normals[np.argmin(similarity)])
    centers = np.array(centers)
    labels = np.zeros(len(normals), dtype=np.int64)
    for _ in range(iterations):
        labels = np.argmax(normals @ centers.T, axis=1)
        updated = centers.copy()
        for g in range(k):
            members = labels == g
            if members.any():
                total = (weights[members, None] * normals[members]).sum(axis=0)
                length = np.linalg.norm(total)
                if length > 1e-12:
                    updated[g] = total / length
        if np.allclose(updated, centers):
            break
        centers = updated
    return centers, labels


def _is_bounded(directions):
    """The planes enclose a finite region iff the origin lies strictly inside the hull of
    their normals (the normals point in all directions)."""
    try:
        hull = ConvexHull(directions)
    except QhullError:
        return False
    return bool(np.all(hull.equations[:, 3] < -1e-6))


def fit_hull_simplified(points, obstacle_id=0, max_planes=16):
    """Convex hull reduced to at most about max_planes planes, still enclosing every point.

    Face normals of the exact hull are grouped by angle (weighted by face area). Each group
    becomes one plane whose offset is the largest projection of any hull vertex, so the
    result contains the hull and therefore every input point. If the grouped directions do
    not enclose a finite region, the six axis directions are added (up to 6 extra planes).
    """
    if max_planes < 4:
        raise ValueError("max_planes must be at least 4")
    pts = check_points(points, min_count=4)
    try:
        hull = ConvexHull(pts)
    except QhullError as exc:
        raise DegenerateGeometry("points are flat, collinear or otherwise degenerate") from exc

    vertices = pts[hull.vertices]
    normals = hull.equations[:, :3]
    tri = pts[hull.simplices]
    areas = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)

    k = min(max_planes, len(normals))
    centers, labels = _spherical_kmeans(normals, areas, k)
    centers = centers[np.unique(labels)]
    _, first = np.unique(np.round(centers, 6), axis=0, return_index=True)
    centers = centers[np.sort(first)]
    if len(centers) < 4 or not _is_bounded(centers):
        centers = np.vstack([centers, np.eye(3), -np.eye(3)])

    b = (vertices @ centers.T).max(axis=0)
    return Polytope(centers, b, method="hull_simple", obstacle_id=obstacle_id, n_points=len(pts))