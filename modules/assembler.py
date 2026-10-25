"""Reassembles UDP chunks into full frames and pairs left/right frames by frame_id.

The cameras are hardware-synced, so both halves of a pair share a frame_id.
Output is in frame_id order: once a pair is emitted, anything older is dropped.
"""
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from . import config
from .protocol import CAM_LEFT, CAM_RIGHT, ChunkHeader, ProtocolError, parse_chunk

log = logging.getLogger(__name__)

## A frame_id this far behind the last emitted one means the Pico restarted
## (or the uint32 counter wrapped), not a late packet.
RESET_WINDOW = 1000


@dataclass(slots=True)
class Frame:
    cam_id: int
    frame_id: int
    t_capture_us: int
    image: np.ndarray  # (height, width) uint8


@dataclass(slots=True)
class StereoPair:
    frame_id: int
    left: Frame
    right: Frame

    @property
    def skew_us(self) -> int:
        return abs(self.left.t_capture_us - self.right.t_capture_us)


@dataclass(slots=True)
class Stats:
    chunks: int = 0
    malformed: int = 0   # failed to parse
    rejected: int = 0    # unknown cam_id, size mismatch, duplicate or stale
    pairs: int = 0
    incomplete: int = 0  # frames dropped with rows missing
    unpaired: int = 0    # complete frames whose partner never completed
    skewed: int = 0      # pairs dropped for exceeding max_skew_us
    resets: int = 0


class _Partial:
    __slots__ = ("hdr", "image", "rows", "missing", "first_seen")

    def __init__(self, hdr: ChunkHeader, now: float):
        self.hdr = hdr
        self.image = np.empty((hdr.height, hdr.width), np.uint8)
        self.rows = np.zeros(hdr.height, bool)
        self.missing = hdr.height
        self.first_seen = now

    def add(self, hdr: ChunkHeader, payload: memoryview) -> None:
        rows = slice(hdr.row_start, hdr.row_start + hdr.row_count)
        self.missing -= hdr.row_count - np.count_nonzero(self.rows[rows])
        self.rows[rows] = True
        self.image[rows] = np.frombuffer(payload, np.uint8).reshape(hdr.row_count, hdr.width)


class StereoAssembler:
    """Feed it datagrams with push(); it returns a StereoPair when one completes."""

    def __init__(self, timeout_s: float = config.FRAME_TIMEOUT_S, max_skew_us: int = config.MAX_SKEW_US,
                 clock: Callable[[], float] = time.monotonic):
        self.timeout_s = timeout_s
        self.max_skew_us = max_skew_us
        self.stats = Stats()
        self._clock = clock
        self.reset()

    def reset(self) -> None:
        """Drop all in-flight frames. Stats are kept."""
        self._partials: dict[tuple[int, int], _Partial] = {}
        self._complete: dict[tuple[int, int], tuple[Frame, float]] = {}
        self._last_emitted: int | None = None

    def push(self, datagram: bytes) -> StereoPair | None:
        now = self._clock()
        self.stats.chunks += 1
        self._drop(lambda fid, t: now - t <= self.timeout_s)

        try:
            hdr, payload = parse_chunk(datagram)
        except ProtocolError as e:
            self.stats.malformed += 1
            log.debug("malformed chunk: %s", e)
            return None
        if hdr.cam_id not in (CAM_LEFT, CAM_RIGHT):
            self.stats.rejected += 1
            return None

        if self._last_emitted is not None and hdr.frame_id <= self._last_emitted:
            if self._last_emitted - hdr.frame_id < RESET_WINDOW:
                self.stats.rejected += 1  # late chunk for an emitted or superseded frame
                return None
            log.warning("frame_id jumped back from %d to %d, assuming Pico restart",
                        self._last_emitted, hdr.frame_id)
            self.stats.resets += 1
            self.reset()

        key = (hdr.cam_id, hdr.frame_id)
        if key in self._complete:
            self.stats.rejected += 1
            return None
        part = self._partials.get(key)
        if part is None:
            part = self._partials[key] = _Partial(hdr, now)
        elif (hdr.width, hdr.height) != (part.hdr.width, part.hdr.height):
            self.stats.rejected += 1
            return None

        part.add(hdr, payload)
        if part.missing:
            return None

        del self._partials[key]
        frame = Frame(hdr.cam_id, hdr.frame_id, part.hdr.t_capture_us, part.image)
        other_cam = CAM_RIGHT if hdr.cam_id == CAM_LEFT else CAM_LEFT
        other = self._complete.pop((other_cam, hdr.frame_id), None)
        if other is None:
            self._complete[key] = (frame, now)
            return None

        left, right = (frame, other[0]) if hdr.cam_id == CAM_LEFT else (other[0], frame)
        return self._emit(StereoPair(hdr.frame_id, left, right))

    def _emit(self, pair: StereoPair) -> StereoPair | None:
        self._drop(lambda fid, t: fid > pair.frame_id)
        self._last_emitted = pair.frame_id
        if pair.skew_us > self.max_skew_us:
            self.stats.skewed += 1
            log.warning("frame %d: left/right capture times differ by %d us, sync broken?",
                        pair.frame_id, pair.skew_us)
            return None
        self.stats.pairs += 1
        return pair

    def _drop(self, keep: Callable[[int, float], bool]) -> None:
        """Discard in-flight frames for which keep(frame_id, timestamp) is false."""
        for key in [k for k, p in self._partials.items() if not keep(k[1], p.first_seen)]:
            del self._partials[key]
            self.stats.incomplete += 1
        for key in [k for k, (_, t) in self._complete.items() if not keep(k[1], t)]:
            del self._complete[key]
            self.stats.unpaired += 1
