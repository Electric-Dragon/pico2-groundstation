# pico groundstation

Ground station for the Pico 2 W and Arducam HM01B0 stereo vision system.

The Pico captures from two hardware-synced HM01B0 cameras and streams raw 8-bit
grayscale frames over UDP, split into row chunks. This repo is the receiving end.
It rebuilds the frames, pairs left and right, calibrates the stereo rig and
rectifies the pairs for stereo matching.

```
Pico 2 W ──UDP chunks──▶ receiver ─▶ assembler ─▶ StereoPair ─▶ rectify ─▶ (depth: not built yet)
                                                      │
                                                      └─▶ calibrate capture ─▶ data/calib/*.png
                                                            calibrate solve ─▶ data/calib.npz
```

## Structure

```
modules/
  config.py      settings with defaults, overridable from .env / environment
  protocol.py    chunk header format (chunk_hdr_t) parsing and validation
  assembler.py   chunks → full frames → left/right pairs by frame_id
  receiver.py    UDP socket + background thread, hands out the latest pair
  calibrate.py   capture board views from the live stream; solve the calibration
  rectify.py     builds rectification maps from calib.npz and applies them
main.py          entry point (not implemented yet)
data/            runtime output: calibration images, calib.npz, snapshots (gitignored)
.env.example     every setting with its default
Dockerfile, docker-compose.yml
```

### Wire format

Each UDP datagram is a 24-byte packed little-endian header followed by
`row_count * width` pixel bytes:

```c
typedef struct __attribute__((packed)) {
    uint16_t magic;        uint8_t  version;   uint8_t cam_id;
    uint32_t frame_id;     uint64_t t_capture_us;
    uint16_t width, height, row_start, row_count;
} chunk_hdr_t;
```

A frame counts as complete only when every row has arrived. Frames with missing
chunks are dropped after a timeout. Left and right are paired by `frame_id`, and a
pair is rejected if its two `t_capture_us` values differ by more than
`GCS_MAX_SKEW_US`. The firmware should record `t_capture_us` at the start of the
frame (VSYNC).

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env    # optional: uncomment and edit the settings you need
```

### Configuration

Every setting has a default in `modules/config.py` and can be overridden with a
`GCS_*` variable in `.env`. The most important ones must match the firmware:

| Variable | Default | |
|---|---|---|
| `GCS_PORT` | `5005` | UDP port the Pico sends to |
| `GCS_MAGIC` / `GCS_VERSION` | unset (not checked) | header magic/version, hex allowed (`0x5053`) |
| `GCS_CAM_LEFT` / `GCS_CAM_RIGHT` | `0` / `1` | which `cam_id` is which camera |
| `GCS_SOURCE` | unset (any) | only accept packets from this IP |
| `GCS_PATTERN` | `8x5` | calibration board inner corners |
| `GCS_SQUARE_MM` | unset | measured board square size |

See `.env.example` for the rest (receive buffer, timeouts, data folder,
rectification crop). Precedence: command-line flag > environment variable >
`.env` > default.

## Running (natively, e.g. on a Mac)

Get the IP address for the Pico system through: `ipconfig getifaddr en0`

Run from the repo root. `calibrate.py` and `rectify.py` can also be run as files,
e.g. `uv run modules/calibrate.py ...`.

**1. Check the link.** Logs pairs/s and drop counters every second.
`--show` opens a left | right preview.

```bash
uv run python -m modules.receiver --show
```

`incomplete` counts frames lost to dropped chunks, `unpaired` counts frames whose
partner never arrived, and `skewed` counts pairs with a left/right timing
mismatch (a sync problem).

**2. Capture calibration images.** A preview window shows the detected corners
and where earlier saves had the board. Keys: `space` saves, `a` toggles
auto-capture, `q` quits. Pairs go to `data/calib/`.

```bash
uv run python -m modules.calibrate capture
```

**3. Solve the calibration.** Writes `data/calib.npz` and `data/rectify_check.png`.

```bash
uv run python -m modules.calibrate solve --square-mm 30   # or set GCS_SQUARE_MM
```

**4. View the rectified stream.** With `--pattern`, a board held in view shows the
live row mismatch, which is a quick way to check the calibration still holds.

```bash
uv run python -m modules.rectify --pattern 8x5
```

### Calibration tips

- Print a chessboard with **9×6 squares** (8×5 inner corners), mount it on
  something rigid and flat, and **measure** the squares after printing.
- Mount the cameras rigidly and fix the lens focus *before* calibrating. Moving
  either afterwards means recalibrating.
- Capture 20–40 views: tilted, at different distances, and covering the
  edges and corners of each camera's image.
- Good results: RMS reprojection error below about 0.5 px, a baseline that matches the
  rig, and a rectified row mismatch well under 1 px. `solve` names
  outlier images to delete.
- A lower streaming resolution doesn't need recalibration. `rectify` rescales the
  calibration automatically. Cropping or windowing the sensor does.

## Running with Docker

The container is headless: no windows. Everything else is the same. `.env` is
passed in as environment variables, and `./data` is mounted at `/app/data`, so
calibration images and results are shared with the host.

```bash
docker compose build

# link check (default command)
docker compose up

# calibration capture: auto-saves still, new board poses (stops after --target, default 40)
docker compose run --rm --service-ports gcs python -m modules.calibrate capture --headless

# solve: no network needed
docker compose run --rm gcs python -m modules.calibrate solve

# rectified stream: logs, and rewrites data/rectified.png every 5 s
docker compose run --rm --service-ports gcs python -m modules.rectify --headless --pattern 8x5
```

Notes:
- The Pico must send to the **host's** LAN IP. The UDP port is published from
  `GCS_PORT`.
- `docker compose run` publishes ports only with `--service-ports`.
- With Docker Desktop, packets may appear to come from Docker's gateway rather
  than the Pico, so `GCS_SOURCE` may filter out everything. Leave it unset if
  nothing arrives.
- Headless capture can't show a preview. Use the logged "not yet covered" regions
  to decide where to hold the board next.
