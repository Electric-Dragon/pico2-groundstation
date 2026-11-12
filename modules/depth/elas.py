"""ELAS, Efficient LArge-scale Stereo (Geiger, Roser & Urtasun, ACCV 2010), in numpy.

ELAS first finds a sparse set of confident "support" matches, triangulates
them, and uses the resulting piecewise-planar surface as a prior for dense
matching. That keeps low-texture areas (common on the HM01B0) close to their
surroundings instead of letting them match noise. Steps, following libelas:
  1. descriptor per pixel: 16 Sobel responses from a 5x5 neighbourhood;
     matching cost = L1 distance between descriptors
  2. support points on a coarse grid: unique, textured, left/right-consistent
     matches; isolated ones are dropped, and image corners are added
  3. Delaunay triangulation of the support points; one disparity plane per triangle
  4. dense matching: each pixel picks the d minimising
     cost(d) + prior(|d - plane(x, y)|), over disparities near the plane plus
     the support disparities seen in its grid cell
  5. the same for the right image, left/right check, speckle removal

Differences from libelas: costs for all disparities are computed up front
(simpler in numpy, and cheap at HM01B0 resolution); redundant support-point
removal, gap interpolation and the adaptive-mean filter are left out; a
subpixel parabola is added. Defaults are libelas's ROBOTICS setting.
"""
import logging

import cv2 as cv
import numpy as np
from scipy.spatial import Delaunay, QhullError, cKDTree

from .. import config
from .base import StereoMatcher
from .common import lr_check, remove_speckles, right_view, winner_takes_all

log = logging.getLogger(__name__)

## (dy, dx) taps of libelas's descriptor: 12 horizontal-gradient samples, 4 vertical.
_DU_TAPS = [(-2, 0), (-1, -2), (-1, 0), (-1, 2), (0, -1), (0, 0), (0, 0), (0, 1),
            (1, -2), (1, 0), (1, 2), (2, 0)]
_DV_TAPS = [(-1, 0), (0, -1), (0, 1), (1, 0)]
_BIG = np.iinfo(np.int32).max // 4  # cost where there is no right pixel to match
_MARGIN = 2  # descriptor radius: pixels closer to the edge than this aren't matched


def descriptors(img: np.ndarray) -> np.ndarray:
    """(H, W, 16) int16 descriptor of Sobel responses, scaled to about ±128 as in libelas."""
    du = np.clip(cv.Sobel(img, cv.CV_16S, 1, 0, ksize=3) // 4, -128, 127)
    dv = np.clip(cv.Sobel(img, cv.CV_16S, 0, 1, ksize=3) // 4, -128, 127)
    H, W = img.shape
    pu = np.pad(du, _MARGIN, mode="edge")
    pv = np.pad(dv, _MARGIN, mode="edge")
    taps = [pu[_MARGIN + dy:_MARGIN + dy + H, _MARGIN + dx:_MARGIN + dx + W] for dy, dx in _DU_TAPS]
    taps += [pv[_MARGIN + dy:_MARGIN + dy + H, _MARGIN + dx:_MARGIN + dx + W] for dy, dx in _DV_TAPS]
    return np.stack(taps, axis=2).astype(np.int16)


def descriptor_costs(desc_l: np.ndarray, desc_r: np.ndarray, D: int) -> np.ndarray:
    """(H, W, D) int32 L1 descriptor distance; _BIG where x - d leaves the matchable area."""
    H, W, _ = desc_l.shape
    C = np.full((H, W, D), _BIG, np.int32)
    for d in range(D):
        C[:, d:, d] = np.abs(desc_l[:, d:] - desc_r[:, :W - d]).sum(axis=2, dtype=np.int32)
    ## Edge rows/columns have no full descriptor; libelas doesn't match them.
    C[_edge_mask(H, W)] = _BIG
    for d in range(D):  # right pixel x - d must also be away from the edge
        C[:, :min(W, d + _MARGIN), d] = _BIG
    return C


def _edge_mask(H: int, W: int) -> np.ndarray:
    m = np.ones((H, W), bool)
    m[_MARGIN:H - _MARGIN, _MARGIN:W - _MARGIN] = False
    return m


def _add_corners(pts: np.ndarray, W: int, H: int) -> np.ndarray:
    """Add the image corners with the disparity of the nearest support point, so
    the triangulation covers the whole image."""
    corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]])
    _, nearest = cKDTree(pts[:, :2]).query(corners)
    return np.vstack([pts, np.column_stack([corners, pts[nearest, 2]])])


class ELAS(StereoMatcher):
    """Parameters (libelas names):
    support_threshold  best/second-best cost ratio a support match must beat
    support_texture    minimum descriptor texture for a support point
    candidate_step     spacing of the support-point candidate grid (px)
    incon_window / incon_threshold / incon_min_support
                       a support point needs this many others within
                       incon_window grid steps whose disparity is within
                       incon_threshold of its own, or it is dropped
    grid_size          cell size (px) for collecting support disparities
    beta, gamma, sigma prior: cost(d) - log(gamma + exp(-(d - plane)^2 / 2 sigma^2)) / beta
    sradius            disparities within this of the plane are always tried
    match_texture      minimum descriptor texture for a dense match
    lr_threshold       left/right consistency tolerance (px)
    speckle_size / speckle_sim_threshold   speckle removal
    """
    name = "elas"

    def __init__(self, num_disparities: int = config.NUM_DISPARITIES,
                 support_threshold: float = 0.85, support_texture: int = 10,
                 candidate_step: int = 5, incon_window: int = 5, incon_threshold: int = 5,
                 incon_min_support: int = 5, grid_size: int = 20,
                 beta: float = 0.02, gamma: float = 3, sigma: float = 1, sradius: int = 2,
                 match_texture: int = 1, lr_threshold: float = 2,
                 speckle_size: int = 200, speckle_sim_threshold: float = 1):
        super().__init__(num_disparities)
        self.support_threshold = support_threshold
        self.support_texture = support_texture
        self.candidate_step = candidate_step
        self.incon_window, self.incon_threshold = incon_window, incon_threshold
        self.incon_min_support = incon_min_support
        self.grid_size = grid_size
        self.beta, self.gamma, self.sigma, self.sradius = beta, gamma, sigma, sradius
        self.match_texture = match_texture
        self.lr_threshold = lr_threshold
        self.speckle_size, self.speckle_sim_threshold = speckle_size, speckle_sim_threshold
        ## Prior in cost units for |d - plane| = 0, 1, ..., sradius; zero further out.
        delta = np.arange(sradius + 1)
        self._prior = ((np.log(gamma) - np.log(gamma + np.exp(-delta ** 2 / (2 * sigma ** 2)))) / beta
                       ).astype(np.float32)

    def _compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        D = self.num_disparities
        desc_l, desc_r = descriptors(left), descriptors(right)
        tex_l = np.abs(desc_l).sum(axis=2, dtype=np.int32)
        tex_r = np.abs(desc_r).sum(axis=2, dtype=np.int32)
        C = descriptor_costs(desc_l, desc_r, D)
        Cr = right_view(C, _BIG)

        support = self.support_points(C, Cr, tex_l)
        if len(support) < 3:
            log.debug("ELAS: only %d support points, no disparity this frame", len(support))
            return np.full(left.shape, np.nan, np.float32)
        support_r = support.copy()
        support_r[:, 0] -= support[:, 2]  # the same points seen in the right image
        H, W = left.shape
        support, support_r = _add_corners(support, W, H), _add_corners(support_r, W, H)

        disp = self.dense(C, tex_l, support)
        disp_r = self.dense(Cr, tex_r, support_r, subpixel=False)
        disp = lr_check(disp, disp_r, self.lr_threshold)
        return remove_speckles(disp, self.speckle_size, self.speckle_sim_threshold)

    ## ---- support points ----------------------------------------------------

    def support_points(self, C: np.ndarray, Cr: np.ndarray, tex: np.ndarray) -> np.ndarray:
        """(N, 3) int array of (x, y, d) in left-image coordinates."""
        H, W, _ = C.shape
        step = self.candidate_step
        ys = np.arange(step, H - _MARGIN, step)
        xs = np.arange(step, W - _MARGIN, step)
        d_l = self._unique_match(C[np.ix_(ys, xs)])
        d_l[tex[np.ix_(ys, xs)] < self.support_texture] = -1

        ## Match the same point from the right image back to the left.
        xr = xs[None, :] - np.maximum(d_l, 0)
        d_r = self._unique_match(Cr[ys[:, None], xr])
        d_l[(d_r < 0) | (np.abs(d_l - d_r) > self.lr_threshold)] = -1

        d_l = self._remove_inconsistent(d_l)
        gy, gx = np.nonzero(d_l >= 0)
        return np.stack([xs[gx], ys[gy], d_l[gy, gx]], axis=1)

    def _unique_match(self, costs: np.ndarray) -> np.ndarray:
        """Best d per row of costs (..., D), or -1 unless best < threshold * second best."""
        order = np.argpartition(costs, 1, axis=-1)
        best = np.take_along_axis(costs, order[..., :1], axis=-1)[..., 0]
        second = np.take_along_axis(costs, order[..., 1:2], axis=-1)[..., 0]
        ok = (best < _BIG) & (second < _BIG) & (best < self.support_threshold * second)
        return np.where(ok, order[..., 0], -1)

    def _remove_inconsistent(self, d: np.ndarray) -> np.ndarray:
        valid = d >= 0
        r = self.incon_window
        padded = np.pad(np.where(valid, d, -10 ** 6), r, constant_values=-10 ** 6)
        count = np.zeros(d.shape, np.int32)
        h, w = d.shape
        for oy in range(2 * r + 1):
            for ox in range(2 * r + 1):
                count += np.abs(padded[oy:oy + h, ox:ox + w] - d) <= self.incon_threshold
        return np.where(valid & (count >= self.incon_min_support), d, -1)

    ## ---- dense matching ----------------------------------------------------

    def dense(self, C: np.ndarray, tex: np.ndarray, support: np.ndarray,
              subpixel: bool = True) -> np.ndarray:
        H, W, D = C.shape
        plane = self._plane_prior(support, W, H)
        if plane is None:
            return np.full((H, W), np.nan, np.float32)

        ## Disparities to try: the support disparities in each grid cell...
        g = self.grid_size
        allowed = self._disparity_grid(support, W, H, D)[(np.arange(H) // g)[:, None], (np.arange(W) // g)[None, :]]
        ## ...plus those within sradius of the plane, with the prior bonus.
        delta = np.abs(np.arange(D)[None, None, :] - np.rint(plane)[..., None])
        near = delta <= self.sradius
        E = C.astype(np.float32)
        E[near] += self._prior[delta[near].astype(np.intp)]
        E[~(allowed | near) | (C >= _BIG)] = np.inf

        disp = winner_takes_all(E, subpixel=subpixel)
        disp[(tex < self.match_texture) | np.isnan(plane)] = np.nan
        return disp

    def _plane_prior(self, support: np.ndarray, W: int, H: int) -> np.ndarray | None:
        """Disparity of the support triangle's plane at every pixel; NaN outside the hull."""
        xy = support[:, :2].astype(np.float64)
        try:
            tri = Delaunay(xy)
        except (QhullError, ValueError):
            return None
        ## Plane d = a x + b y + c through each triangle's three corners.
        A = np.concatenate([xy[tri.simplices], np.ones((*tri.simplices.shape, 1))], axis=2)
        b = support[tri.simplices, 2].astype(np.float64)
        ok = np.abs(np.linalg.det(A)) > 1e-9
        planes = np.zeros((len(A), 3))
        planes[ok] = np.linalg.solve(A[ok], b[ok][..., None])[..., 0]
        planes[~ok, 2] = b[~ok].mean(axis=1)

        yy, xx = np.mgrid[0:H, 0:W]
        idx = tri.find_simplex(np.column_stack([xx.ravel(), yy.ravel()])).reshape(H, W)
        p = planes[np.maximum(idx, 0)]
        plane = (p[..., 0] * xx + p[..., 1] * yy + p[..., 2]).astype(np.float32)
        plane[idx < 0] = np.nan
        return plane

    def _disparity_grid(self, support: np.ndarray, W: int, H: int, D: int) -> np.ndarray:
        """(gh, gw, D) bool: disparities within ±1 of a support point in this cell
        or one of its 8 neighbours."""
        g = self.grid_size
        gh, gw = -(-H // g), -(-W // g)
        grid = np.zeros((gh + 2, gw + 2, D), bool)  # one cell of padding each side
        cy = np.clip(support[:, 1] // g, 0, gh - 1) + 1
        cx = np.clip(support[:, 0] // g, 0, gw - 1) + 1
        for dd in (-1, 0, 1):
            d = np.clip(support[:, 2] + dd, 0, D - 1)
            for oy in (-1, 0, 1):
                for ox in (-1, 0, 1):
                    grid[cy + oy, cx + ox, d] = True
        return grid[1:-1, 1:-1]
