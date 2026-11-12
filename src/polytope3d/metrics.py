"""Quality metrics for polytopes. All distances are in metres."""

import numpy as np

from polytope3d.polytope import Polytope

FIXED_DIRECTION_METHODS = frozenset({"aabb", "kdop14", "kdop26"})


def _as_list(polytopes):
    return [polytopes] if isinstance(polytopes, Polytope) else list(polytopes)


def containment_rate(polytopes, points, tol=1e-6):
    """Fraction of points inside at least one of the polytopes (1.0 is required)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return 1.0
    inside = np.zeros(len(pts), dtype=bool)
    for poly in _as_list(polytopes):
        inside |= poly.contains(pts, tol)
    return float(inside.mean())


def boundary_gaps(poly, points):
    """Distance from each point to the nearest face of the polytope (negative = outside)."""
    slack = poly.b - np.asarray(points, dtype=np.float64) @ poly.A.T
    return slack.min(axis=1)


def gap_stats(poly, points):
    """Mean, 95th percentile and maximum boundary gap: how loosely the polytope wraps."""
    gaps = boundary_gaps(poly, points)
    return {
        "mean": float(gaps.mean()),
        "p95": float(np.percentile(gaps, 95)),
        "max": float(gaps.max()),
    }


def total_volume(polytopes):
    """Sum of volumes. For several overlapping parts this over-counts the union."""
    return float(sum(p.volume() for p in _as_list(polytopes)))


def estimated_bytes(polytopes):
    """Rough float32 payload size. Fixed-direction methods send one offset per plane; the
    others send a normal and an offset (4 floats). A sizing estimate, not a packet format."""
    return int(
        sum(
            p.n_planes * (4 if p.method in FIXED_DIRECTION_METHODS else 16)
            for p in _as_list(polytopes)
        )
    )


def frame_to_frame(previous, current):
    """Change between the same obstacle's polytopes in consecutive frames (jitter)."""
    shift = np.linalg.norm(current.vertices().mean(axis=0) - previous.vertices().mean(axis=0))
    return {"centre_shift": float(shift), "volume_ratio": current.volume() / previous.volume()}
