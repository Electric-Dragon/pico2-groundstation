"""The Polytope object. The H-representation (A x <= b) is the stored truth."""

from dataclasses import dataclass, replace

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, QhullError


class DegenerateGeometry(ValueError):
    """Raised when points or planes cannot form a valid, non-flat 3D polytope."""


def check_points(points, min_count=1):
    """Validate a cluster and return it as a float64 (N, 3) array."""
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must be (N, 3)")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points contain NaN or infinity")
    if len(pts) < min_count:
        raise DegenerateGeometry(f"need at least {min_count} points, got {len(pts)}")
    return pts.astype(np.float64)


@dataclass(frozen=True, eq=False)
class Polytope:
    """Convex polytope {x : A x <= b}. Every row of A is a unit normal, so b is in metres."""

    A: np.ndarray
    b: np.ndarray
    method: str
    obstacle_id: int = 0
    margin: float = 0.0
    n_points: int = 0

    def __post_init__(self):
        if self.A.ndim != 2 or self.A.shape[1] != 3 or self.b.shape != (self.A.shape[0],):
            raise ValueError("A must be (M, 3) and b must be (M,)")
        if self.A.shape[0] < 4:
            raise ValueError("a bounded 3D polytope needs at least 4 planes")
        if not (np.all(np.isfinite(self.A)) and np.all(np.isfinite(self.b))):
            raise ValueError("A and b must be finite")
        if np.any(np.abs(np.linalg.norm(self.A, axis=1) - 1.0) > 1e-6):
            raise ValueError("every row of A must have unit length")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")

    @property
    def n_planes(self):
        return self.A.shape[0]

    def contains(self, points, tol=1e-6):
        """Boolean mask: True for points inside (or within tol metres of) every plane."""
        return np.all(np.asarray(points, dtype=np.float64) @ self.A.T <= self.b + tol, axis=1)

    def inflated(self, margin):
        """A copy with every plane moved outward by `margin` metres (rows are unit length)."""
        if margin < 0:
            raise ValueError("margin must be >= 0")
        return replace(self, b=self.b + margin, margin=self.margin + margin)

    def distance_lower_bound(self, point):
        """max_i (a_i . p - b_i). For a point outside, a lower bound on its distance to the
        polytope; for a point inside, this value is <= 0."""
        return float(np.max(self.A @ np.asarray(point, dtype=np.float64) - self.b))

    def interior_point(self):
        """Centre of the largest inscribed sphere, and its radius (a linear programme)."""
        c = np.array([0.0, 0.0, 0.0, -1.0])
        a_ub = np.hstack([self.A, np.ones((self.n_planes, 1))])
        bounds = [(None, None)] * 3 + [(0.0, None)]
        res = linprog(c, A_ub=a_ub, b_ub=self.b, bounds=bounds, method="highs")
        if res.status != 0 or res.x[3] < 1e-9:
            raise DegenerateGeometry("polytope is empty, unbounded or has zero volume")
        return res.x[:3], float(res.x[3])

    def vertices(self):
        """Corner points, derived from the planes (for drawing and volume only)."""
        centre, _ = self.interior_point()
        try:
            hs = HalfspaceIntersection(np.hstack([self.A, -self.b[:, None]]), centre)
            v = hs.intersections
            return v[ConvexHull(v).vertices]
        except QhullError as exc:
            raise DegenerateGeometry("could not compute vertices") from exc

    def volume(self):
        return float(ConvexHull(self.vertices()).volume)
