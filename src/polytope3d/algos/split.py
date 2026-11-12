"""Recursive splitting: several small polytopes for a cluster that is not convex."""

from dataclasses import replace

import numpy as np

from polytope3d.algos.obb import fit_obb, pca_axes
from polytope3d.polytope import DegenerateGeometry, check_points


def _volume(poly):
    """Box volume from its extents (cheap); other shapes use the exact polytope volume."""
    if poly.method in ("aabb", "obb"):
        return float(np.prod(poly.b[:3] + poly.b[3:]))
    return poly.volume()


def _best_split(pts, base, min_points):
    """Try cutting at the centroid, perpendicular to each principal axis; keep the best cut."""
    axes = pca_axes(pts)
    centre = pts.mean(axis=0)
    best = None
    for k in range(3):
        right_side = (pts - centre) @ axes[:, k] > 0
        left, right = pts[~right_side], pts[right_side]
        if len(left) < min_points or len(right) < min_points:
            continue
        try:
            poly_left, poly_right = base(left), base(right)
            total = _volume(poly_left) + _volume(poly_right)
        except DegenerateGeometry:
            continue
        if best is None or total < best[0]:
            best = (total, left, right, poly_left, poly_right)
    return best


def _recurse(pts, poly, base, depth, max_depth, min_points, improvement):
    if depth >= max_depth or len(pts) < 2 * min_points:
        return [poly]
    best = _best_split(pts, base, min_points)
    if best is None or best[0] >= improvement * _volume(poly):
        return [poly]
    _, left, right, poly_left, poly_right = best
    return _recurse(left, poly_left, base, depth + 1, max_depth, min_points, improvement) + _recurse(
        right, poly_right, base, depth + 1, max_depth, min_points, improvement
    )


def fit_split(
    points, base=fit_obb, obstacle_id=0, max_depth=3, min_points=30, improvement=0.8
):
    """Fit several polytopes to one cluster by recursively cutting it in two.

    A cut at the centroid, perpendicular to a principal axis, is kept only if the two halves'
    polytopes together have less than `improvement` times the volume of the undivided one.
    The halves partition the points, so the union of the returned polytopes contains every
    point. All parts share `obstacle_id`; `part_id` numbers them 0, 1, 2, ...
    """
    if max_depth < 0 or min_points < 4 or not 0 < improvement <= 1:
        raise ValueError("need max_depth >= 0, min_points >= 4 and 0 < improvement <= 1")
    pts = check_points(points, min_count=4)
    root = base(pts)
    parts = _recurse(pts, root, base, 0, max_depth, min_points, improvement)
    return [
        replace(p, method=f"split_{p.method}", obstacle_id=obstacle_id, part_id=i)
        for i, p in enumerate(parts)
    ]