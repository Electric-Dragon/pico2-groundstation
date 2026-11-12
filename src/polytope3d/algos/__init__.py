"""Polytope-fitting algorithms. Each takes an (N, 3) cluster and returns a Polytope."""

from polytope3d.algos.aabb import fit_aabb
from polytope3d.algos.hull import fit_hull
from polytope3d.algos.obb import fit_obb

ALGORITHMS = {"aabb": fit_aabb, "obb": fit_obb, "hull": fit_hull}

__all__ = ["ALGORITHMS", "fit_aabb", "fit_hull", "fit_obb"]
