"""Wire format of the frame chunks the Pico 2 W sends over UDP.

Mirrors the firmware's packed `chunk_hdr_t`. The RP2350 is little-endian.
"""
import struct
from dataclasses import dataclass

from . import config

HEADER_FMT = "<HBBIQHHHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 24

## Set via GCS_MAGIC / GCS_VERSION in .env; None skips the check.
MAGIC = config.MAGIC
VERSION = config.VERSION

CAM_LEFT = config.CAM_LEFT
CAM_RIGHT = config.CAM_RIGHT


class ProtocolError(ValueError):
    """Datagram is not a valid chunk."""


@dataclass(frozen=True, slots=True)
class ChunkHeader:
    magic: int
    version: int
    cam_id: int
    frame_id: int
    t_capture_us: int
    width: int
    height: int
    row_start: int
    row_count: int

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FMT, self.magic, self.version, self.cam_id, self.frame_id,
            self.t_capture_us, self.width, self.height, self.row_start, self.row_count,
        )


def parse_chunk(datagram: bytes) -> tuple[ChunkHeader, memoryview]:
    """Split a datagram into its header and pixel payload (row_count * width bytes)."""
    if len(datagram) < HEADER_SIZE:
        raise ProtocolError(f"datagram too short: {len(datagram)} bytes")

    hdr = ChunkHeader(*struct.unpack_from(HEADER_FMT, datagram))
    if MAGIC is not None and hdr.magic != MAGIC:
        raise ProtocolError(f"bad magic 0x{hdr.magic:04x}")
    if VERSION is not None and hdr.version != VERSION:
        raise ProtocolError(f"unsupported version {hdr.version}")
    if hdr.width == 0 or hdr.height == 0 or hdr.row_count == 0:
        raise ProtocolError("zero width, height or row_count")
    if hdr.row_start + hdr.row_count > hdr.height:
        raise ProtocolError(
            f"rows {hdr.row_start}+{hdr.row_count} exceed height {hdr.height}")

    payload = memoryview(datagram)[HEADER_SIZE:]
    if len(payload) != hdr.row_count * hdr.width:
        raise ProtocolError(
            f"payload is {len(payload)} bytes, expected {hdr.row_count * hdr.width}")
    return hdr, payload
