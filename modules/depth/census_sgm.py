"""Census-cost semi-global matching (Hirschmüller 2008, with a census cost as in
Zabih & Woodfill 1994), in numpy.

Census compares each pixel with its neighbours and keeps only the order, so it
copes with the two cameras having different gain or exposure, which a raw
intensity cost does not. Steps:
  1. census bit string per pixel; cost = Hamming distance (H, W, D volume)
  2. SGM: aggregate costs along 4 or 8 straight paths, penalising disparity
     changes of 1 by P1 and larger jumps by P2
  3. winner-takes-all with uniqueness check and subpixel parabola
  4. left/right consistency check, then speckle removal
"""
import numpy as np

from .. import config
from .base import StereoMatcher
from .common import lr_check, remove_speckles, right_view, winner_takes_all

## Direction a path travels in, as (dy, dx). Each pixel's predecessor is p - (dy, dx).
_PATHS4 = [(0, 1), (0, -1), (1, 0), (-1, 0)]
_PATHS8 = _PATHS4 + [(1, 1), (1, -1), (-1, 1), (-1, -1)]


def parse_window(s: str) -> tuple[int, int]:
    w, h = (int(v) for v in s.lower().split("x"))
    if w % 2 == 0 or h % 2 == 0 or w * h - 1 > 64:
        raise ValueError(f"census window must be odd x odd with at most 65 pixels, got {s!r}")
    return w, h


def census(img: np.ndarray, window: tuple[int, int]) -> np.ndarray:
    """uint64 code per pixel: one bit per neighbour, set if the neighbour is darker."""
    w, h = window
    rx, ry = w // 2, h // 2
    H, W = img.shape
    padded = np.pad(img, ((ry, ry), (rx, rx)), mode="edge")
    code = np.zeros((H, W), np.uint64)
    for dy in range(h):
        for dx in range(w):
            if dy == ry and dx == rx:
                continue
            code = (code << np.uint64(1)) | (padded[dy:dy + H, dx:dx + W] < img)
    return code


def census_costs(left: np.ndarray, right: np.ndarray, D: int, window: tuple[int, int]) -> np.ndarray:
    cl, cr = census(left, window), census(right, window)
    H, W = left.shape
    C = np.full((H, W, D), window[0] * window[1] - 1, np.uint8)  # max distance where x - d < 0
    for d in range(D):
        C[:, d:, d] = np.bitwise_count(cl[:, d:] ^ cr[:, :W - d])
    return C


def sgm_aggregate(C: np.ndarray, P1: int, P2: int, paths: int = 8) -> np.ndarray:
    """Sum over paths r of L_r(p, d) = C(p, d) + min(L_r(p-r, d), L_r(p-r, d±1) + P1,
    min_k L_r(p-r, k) + P2) - min_k L_r(p-r, k)."""
    if paths not in (4, 8):
        raise ValueError(f"paths must be 4 or 8, got {paths}")
    S = np.zeros(C.shape, np.int32)
    for dy, dx in _PATHS8 if paths == 8 else _PATHS4:
        c, s = C, S
        if dy == 0:  # horizontal path: sweep columns instead of rows
            c, s = c.transpose(1, 0, 2), s.transpose(1, 0, 2)
            dy, dx = dx, 0
        if dy < 0:
            c, s = c[::-1], s[::-1]
        _sweep(c, s, dx, P1, P2)
    return S


def _sweep(c: np.ndarray, s: np.ndarray, dx: int, P1: int, P2: int) -> None:
    """One path, top row to bottom; (i, j)'s predecessor is (i - 1, j - dx).
    Adds the path costs into s in place (s may be a transposed/flipped view)."""
    prev = c[0].astype(np.int32)
    s[0] += prev
    for i in range(1, c.shape[0]):
        m = prev.min(axis=1, keepdims=True)
        t = np.minimum(prev, m + P2)
        np.minimum(t[:, 1:], prev[:, :-1] + P1, out=t[:, 1:])
        np.minimum(t[:, :-1], prev[:, 1:] + P1, out=t[:, :-1])
        t -= m
        cur = c[i].astype(np.int32)
        if dx == 1:
            cur[1:] += t[:-1]
        elif dx == -1:
            cur[:-1] += t[1:]
        else:
            cur += t
        s[i] += cur
        prev = cur


class CensusSGM(StereoMatcher):
    """window: census window "WxH" (W*H - 1 <= 64 bits). P1/P2: SGM penalties in
    Hamming-distance units. paths: 4 is roughly twice as fast as 8."""
    name = "census-sgm"

    def __init__(self, num_disparities: int = config.NUM_DISPARITIES,
                 window: str = config.CENSUS_WINDOW, P1: int = config.SGM_P1,
                 P2: int = config.SGM_P2, paths: int = config.SGM_PATHS,
                 uniqueness: float = 5, lr_max_diff: float = 1,
                 speckle_size: int = 100, speckle_range: float = 2):
        super().__init__(num_disparities)
        self.window = parse_window(window)
        if not 0 < P1 < P2:
            raise ValueError(f"need 0 < P1 < P2, got P1={P1} P2={P2}")
        if paths not in (4, 8):
            raise ValueError(f"paths must be 4 or 8, got {paths}")
        self.P1, self.P2, self.paths = P1, P2, paths
        self.uniqueness = uniqueness
        self.lr_max_diff = lr_max_diff
        self.speckle_size, self.speckle_range = speckle_size, speckle_range

    def _compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        C = census_costs(left, right, self.num_disparities, self.window)
        S = sgm_aggregate(C, self.P1, self.P2, self.paths)
        disp = winner_takes_all(S, self.uniqueness)
        disp_r = winner_takes_all(right_view(S, np.iinfo(np.int32).max), subpixel=False)
        disp = lr_check(disp, disp_r, self.lr_max_diff)
        return remove_speckles(disp, self.speckle_size, self.speckle_range)
