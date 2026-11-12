"""Cost-volume helpers shared by the numpy matchers.

A cost volume C has shape (H, W, D): C[y, x, d] is the cost of matching left
pixel (x, y) with right pixel (x - d, y). Lower is better.
"""
import cv2 as cv
import numpy as np


def right_view(C: np.ndarray, fill) -> np.ndarray:
    """The same costs indexed from the right image: Cr[y, xr, d] = C[y, xr + d, d].
    Entries whose left pixel would fall off the image get `fill`."""
    H, W, D = C.shape
    Cr = np.full_like(C, fill)
    for d in range(D):
        Cr[:, :W - d, d] = C[:, d:, d]
    return Cr


def winner_takes_all(C: np.ndarray, uniqueness: float = 0.0, subpixel: bool = True) -> np.ndarray:
    """Lowest-cost disparity per pixel, as float32.

    uniqueness (percent, like OpenCV's uniquenessRatio): a pixel is NaN unless
    the best cost beats every disparity more than one step away from it by this
    margin. subpixel fits a parabola through the best cost and its neighbours.
    Non-finite costs (e.g. inf for disallowed disparities) never win.
    """
    D = C.shape[2]
    best = np.argmin(C, axis=2)
    c0 = np.take_along_axis(C, best[..., None], axis=2)[..., 0].astype(np.float32)
    disp = best.astype(np.float32)

    if uniqueness > 0:
        other = C.astype(np.float32)  # copy
        for k in (-1, 0, 1):
            idx = np.clip(best + k, 0, D - 1)[..., None]
            np.put_along_axis(other, idx, np.inf, axis=2)
        c1 = other.min(axis=2)
        disp[~(c1 * 100 > c0 * (100 + uniqueness))] = np.nan

    if subpixel:
        inner = (best > 0) & (best < D - 1)
        lo = np.take_along_axis(C, np.clip(best - 1, 0, D - 1)[..., None], axis=2)[..., 0].astype(np.float32)
        hi = np.take_along_axis(C, np.clip(best + 1, 0, D - 1)[..., None], axis=2)[..., 0].astype(np.float32)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            denom = lo - 2 * c0 + hi
            offset = (lo - hi) / (2 * denom)
        ok = inner & np.isfinite(offset) & (denom > 0)
        disp[ok] += np.clip(offset[ok], -0.5, 0.5)

    disp[~np.isfinite(c0)] = np.nan
    return disp


def lr_check(disp_l: np.ndarray, disp_r: np.ndarray, max_diff: float = 1.0) -> np.ndarray:
    """Copy of disp_l with NaN wherever the right image's disparity at the matched
    pixel disagrees by more than max_diff (occlusions and mismatches)."""
    H, W = disp_l.shape
    xs = np.arange(W)[None, :] - np.nan_to_num(disp_l, nan=W)
    xr = np.rint(xs).astype(np.intp)
    inside = (xr >= 0) & (xr < W)
    d_r = np.take_along_axis(disp_r, np.clip(xr, 0, W - 1), axis=1)
    out = disp_l.copy()
    with np.errstate(invalid="ignore"):
        out[~(inside & (np.abs(disp_l - d_r) <= max_diff))] = np.nan
    return out


def remove_speckles(disp: np.ndarray, max_size: int, max_diff: float) -> np.ndarray:
    """NaN out connected blobs of fewer than max_size pixels whose disparities differ
    from their surroundings by more than max_diff (cv.filterSpeckles)."""
    if max_size <= 0:
        return disp
    fixed = np.where(np.isfinite(disp), np.rint(disp * 16), -16).astype(np.int16)
    cv.filterSpeckles(fixed, -16, max_size, int(round(max_diff * 16)))
    out = disp.copy()
    out[fixed == -16] = np.nan
    return out
