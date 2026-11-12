"""Polytope-fitting algorithms. Single-polytope fitters return one Polytope; multi-polytope
fitters return a list."""

from polytope3d.algos.aabb import fit_aabb
from polytope3d.algos.hull import fit_hull
from polytope3d.algos.hull_simple import fit_hull_simplified
from polytope3d.algos.kdop import fit_kdop, fit_kdop14, fit_kdop26
from polytope3d.algos.obb import fit_obb
from polytope3d.algos.split import fit_split

ALGORITHMS = {
    "aabb": fit_aabb,
    "obb": fit_obb,
    "kdop14": fit_kdop14,
    "kdop26": fit_kdop26,
    "hull": fit_hull,
    "hull_simple": fit_hull_simplified,
}

MULTI_ALGORITHMS = {"split_obb": fit_split}

__all__ = [
    "ALGORITHMS",
    "MULTI_ALGORITHMS",
    "fit_aabb",
    "fit_hull",
    "fit_hull_simplified",
    "fit_kdop",
    "fit_kdop14",
    "fit_kdop26",
    "fit_obb",
    "fit_split",
]
