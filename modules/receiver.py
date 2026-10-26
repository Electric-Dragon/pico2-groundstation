"""UDP receive loop: reads chunks from the Pico on a background thread and
queues completed stereo pairs for whatever consumes them.

Run directly to check the link:  uv run python -m modules.receiver [--port N] [--show]
"""
import argparse
import logging
import os
import queue
import signal
import socket
import sys
import threading
import time

import numpy as np

from . import config
from .assembler import StereoAssembler, StereoPair

log = logging.getLogger(__name__)


def no_display() -> bool:
    """True when no window can be opened (e.g. Linux in Docker), so CLIs go headless."""
    return sys.platform.startswith("linux") and not (
        os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class Receiver:
    """Owns the socket and assembler. Use as a context manager, then call get()."""

    def __init__(self, port: int = config.PORT, host: str = config.HOST,
                 source: str | None = config.SOURCE, rcvbuf: int = config.RCVBUF,
                 assembler: StereoAssembler | None = None):
        self.asm = assembler or StereoAssembler()
        self.source = source
        self.dropped = 0  # pairs discarded because the consumer fell behind

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        ## A small OS buffer silently loses chunks whenever Python stalls briefly.
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, rcvbuf)
        actual = self.sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        if actual < rcvbuf:
            log.warning("receive buffer is %d bytes, asked for %d (OS limit?)", actual, rcvbuf)
        self.sock.bind((host, port))
        self.sock.settimeout(0.5)  # lets the thread notice stop()

        self._pairs: queue.Queue[StereoPair] = queue.Queue(maxsize=2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="receiver", daemon=True)

    def start(self) -> "Receiver":
        self._thread.start()
        log.info("listening on %s:%d", *self.sock.getsockname())
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        self.sock.close()

    def __enter__(self) -> "Receiver":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def get(self, timeout: float | None = None) -> StereoPair | None:
        """Latest completed pair, or None if none arrives within timeout."""
        try:
            return self._pairs.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                data, (addr, _) = self.sock.recvfrom(65535)
            except TimeoutError:
                continue
            if self.source is not None and addr != self.source:
                continue
            try:
                pair = self.asm.push(data)
            except Exception:
                log.exception("assembler failed on a chunk from %s", addr)
                continue
            if pair is not None:
                self._put_latest(pair)

    def _put_latest(self, pair: StereoPair) -> None:
        ## Live processing wants the newest pair, not a backlog.
        if self._pairs.full():
            try:
                self._pairs.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
        self._pairs.put_nowait(pair)


def main() -> None:
    parser = argparse.ArgumentParser(description="Receive stereo frames from the Pico and report link health.")
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--source", default=config.SOURCE, help="only accept chunks from this IP")
    parser.add_argument("--show", action="store_true", help="preview left|right in a window")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ## `docker stop` sends SIGTERM, which PID 1 otherwise ignores.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))

    if args.show:
        import cv2 as cv

    with Receiver(args.port, args.host, args.source) as rx:
        count, last, latest, fps = 0, time.monotonic(), None, 0.0
        try:
            while True:
                pair = rx.get(timeout=0.1)
                if pair is not None:
                    count += 1
                    latest = pair
                    if args.show:
                        view = np.hstack([pair.left.image, pair.right.image])
                        label = f"{fps:.1f} fps"
                        cv.putText(view, label, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, 0, 3, cv.LINE_AA)
                        cv.putText(view, label, (8, 22), cv.FONT_HERSHEY_SIMPLEX, 0.6, 255, 1, cv.LINE_AA)
                        cv.imshow("left | right", view)
                if args.show and cv.waitKey(1) & 0xFF == ord("q"):
                    break

                now = time.monotonic()
                if now - last >= 1.0:
                    size = f"{latest.left.image.shape[1]}x{latest.left.image.shape[0]}" if latest else "-"
                    skew = f"{latest.skew_us}us" if latest else "-"
                    fps = count / (now - last)
                    log.info("%.1f pairs/s  size %s  skew %s  consumer-dropped %d  %s",
                             fps, size, skew, rx.dropped, rx.asm.stats)
                    count, last = 0, now
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
