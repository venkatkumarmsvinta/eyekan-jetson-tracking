# eyekan-jetson-tracking

People detection and tracking for store CCTV. Every person gets an ID that stays the same while they move,
get hidden behind a rack, or walk out of the frame and come back. Output: an annotated video, a CSV with one
row per person per frame (box, ID, face confidence, facing, floor position), and a top-down floor map.

Runs on a Mac (Apple GPU / MPS) for development and on an NVIDIA Jetson (CUDA) in the store.

## How it works

```
video frame
   │
   ▼
1. RF-DETR            person boxes (posters showing people are ignored)
   │
   ▼
2. ViTPose            17 keypoints per box (nose, eyes, ears, shoulders, ... ankles)
   │                  duplicate boxes on one person are removed
   ▼
3. BoT-SORT           a tracker ID per box, frame to frame (by position/motion only)
   │                  a hidden person is kept for HOLD_S = 10 s
   ▼
4. ReLink             when the tracker starts a NEW ID: is this someone we lost?
   │                  → appearance (OSNet re-ID), position, torso colour
   │                  → if yes, the person gets their old ID back
   ▼
5. Output             video + CSV + floor map (metres with a floor calibration, else pixels)
```

### How IDs are kept (ReLink)

- While a person is tracked, a small crop of them is kept **in memory** every 2 s (max 15, ~1.5 MB per
  person), only when their box overlaps nobody and the torso is visible. No model runs for this.
- The **re-ID model (OSNet) runs only when the tracker gives a new ID** and someone lost could match it:
  on the new person's crop and once on each lost person's stored crops. In a 12-minute test it ran on
  ~650 crops in total.
- A new ID gets an old ID back when the appearance similarity is
  - **≥ 0.82** — anywhere (walked out of the frame and came back elsewhere), or
  - **≥ 0.70** — and near where the lost person was last seen.
- Torso colour can only **block** a match (clearly different clothes), never make one.
- New box overlapping another person (its crop shows two people): colour + position only, no re-ID.
- Torso hidden (only the head visible): position only, and only for someone lost ≤ 2 s ago within 1 box height.
- A new ID keeps being checked by appearance for its first 5 s (people walk back in only partly visible).
- **Hand-over** (two people overlapped and the tracker started a new ID while the old one was alive):
  colour + position only, no re-ID. The CSV and floor map are corrected afterwards.

Known limits: two people in identical clothes (e.g. staff uniforms) look alike to the re-ID model; a person
who looks very different when they come back (e.g. was holding a big garment in front of them) gets a new ID —
on purpose, a new ID is safer than a wrong one.

## Folder structure

```
track.py                 entry point: python track.py <video>
calibrate_floor.py       one-time floor calibration per camera (chessboard -> metres)
requirements.txt
tools/
├── sync_clocks.py       frame offset between two recordings from the burned-in clock
└── calibrate_3d.py      3D camera calibration from measured store points (multi-camera)
tracking/
├── camera.py            calibrated camera: store metres <-> pixels (head/feet -> floor position)
├── config.py            ALL settings (thresholds, poster zones, model names, CSV columns)
├── pipeline.py          the per-frame loop: detect → pose → track → ReLink → outputs
├── detection.py         RF-DETR models, poster filter, duplicate-box removal
├── pose.py              ViTPose keypoints
├── keypoints.py         facing, torso colour, crop for the re-ID model
├── tracker.py           BoT-SORT tracker + OSNet re-ID model loader
├── relink.py            ReLink (keeps IDs) + CSV/floor-map correction after renames
├── floor.py             pixel → floor position, floor map image
└── drawing.py           drawing a person on the output video
```

Not in git (see `.gitignore`): `videos/`, `output/`, model weights (`*.pth`, `*.pt`), `venv*/`, `calibration/`.

## Setup

### Mac / Linux PC

```bash
git clone git@github.com:venkatkumarmsvinta/eyekan-jetson-tracking.git
cd eyekan-jetson-tracking
python3 -m venv venv-rfdetr
source venv-rfdetr/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir videos        # put the camera videos here, e.g. videos/camera1.mp4
```

### NVIDIA Jetson

`torch` and `torchvision` from PyPI do not use the Jetson GPU. Install NVIDIA's builds for your JetPack
version first, then the rest:

```bash
cat /etc/nv_tegra_release            # shows the JetPack / L4T version
python3 -m venv --system-site-packages venv-rfdetr
source venv-rfdetr/bin/activate
# 1. install torch + torchvision wheels matching your JetPack (NVIDIA "PyTorch for Jetson" page)
# 2. then everything else, without replacing torch:
grep -v -E '^(torch|torchvision)==' requirements.txt > /tmp/req-jetson.txt
pip install -r /tmp/req-jetson.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
```

On the Jetson the code picks `cuda` automatically and runs the re-ID model in half precision. For more speed,
set `REID_WEIGHTS = "osnet_x0_25_msmt17.pt"` in `tracking/config.py` (~2x faster, slightly less accurate).

### Model weights (downloaded automatically on first run)

| Model | Used for | Where it goes |
|---|---|---|
| RF-DETR nano (`rf-detr-nano.pth`, ~350 MB) | person boxes | current folder |
| ViTPose-B (`usyd-community/vitpose-base-simple`) | keypoints | Hugging Face cache (`~/.cache/huggingface`) |
| OSNet (`osnet_ain_x1_0_msmt17.pt`, 17 MB) | re-ID in ReLink | boxmot's `engine/weights/` folder inside the venv |

The first run needs internet; after that it works offline.

## Run

Always run from the repository folder (outputs and calibrations are relative to it).

```bash
source venv-rfdetr/bin/activate
python track.py videos/camera1.mp4 --show
```

| Option | Default | Meaning |
|---|---|---|
| `--size` | `nano` | RF-DETR size: `nano`, `small`, `medium`, `base`, `large` (bigger = more accurate, slower) |
| `--conf` | `0.25` | minimum detection confidence |
| `--show` | off | show the video and floor map live (`q` to quit) |
| `--max-frames N` | `0` | stop after N frames (0 = whole video) |

Progress is printed every 100 frames: fps, boxes, IDs, re-linked, handed over, duplicates, re-ID crops.

### Floor calibration (optional, once per camera)

Without it, floor positions are in image pixels. With it, in metres.

1. Lay a printed chessboard flat on the floor where the camera sees it; record a short video.
2. Run (cols/rows = **inner** corners, square = side of one square in metres):
   ```bash
   python calibrate_floor.py videos/camera1_chessboard.mp4 --name camera1 --cols 9 --rows 6 --square 0.10
   ```
3. Check `calibration/camera1_check.jpg`: the drawn 0.5 m grid must lie flat on the floor.

`track.py` loads `calibration/<video name>.json` automatically for the video with the same name.

## Multi-camera preparation (branch `multicamera-tracking`)

Goal: the same person keeps the same ID across cameras that see the same area (now: camera1 + camera6).
Two things are needed first: the videos in sync, and every camera calibrated in one store coordinate system.

### 1. Sync the recordings (burned-in clock)

```bash
python tools/sync_clocks.py videos/camera1.mp4 videos/camera6.mp4
# read the two times on output/sync/*_first_tick.png, then:
python tools/sync_clocks.py videos/camera1.mp4 videos/camera6.mp4 --times 04:01:46 04:00:52
# -> RESULT: camera6 frame = camera1 frame +805 ; check output/sync/camera1_camera6_check.jpg
```

Put the result in `CAMERA_GROUPS` in `tracking/config.py` (camera1 + camera6 is already there).

### 2. Calibrate each camera in 3D

1. Pick one origin (0, 0) on the store floor (e.g. a wall corner). Measure points the cameras see, in metres:
   x and y along the floor, z = height. At least 6 per camera, at least 2 of them **not** on the floor
   (counter top corners, poster corners, top of a sign). Write them in `calibration/store_points.csv`:
   ```
   name,x,y,z
   A1_counter_top_near_trial_room,2.10,3.40,0.95
   E1_floor_tile_corner,4.00,1.20,0
   ```
   One file for all cameras. The cameras do not need to see the same points, only the same origin.
2. Click the points each camera sees, on a frame without people in front of them (zoom window helps;
   `n` = skip a point the camera can't see, `u` = undo, `q` = save):
   ```bash
   python tools/calibrate_3d.py click camera1 --frame 3375
   python tools/calibrate_3d.py click camera6 --frame 4180
   ```
3. Compute (prints the error per point; points over 15 px are probably a wrong measurement or click):
   ```bash
   python tools/calibrate_3d.py solve camera1 --frame 3375 --distortion
   python tools/calibrate_3d.py solve camera6 --frame 4180 --distortion
   ```
   Check `calibration/<camera>_check.jpg`: the 0.5 m floor grid must lie flat on the floor and the red 1.7 m
   posts must look as tall as a person. The JSON also contains a floor homography, so `track.py` gives floor
   positions in metres for these cameras too. Use `--distortion` with 8+ points.

## Output (`output/<video name>/`)

| File | Content |
|---|---|
| `rfdetr.mp4` | video with box + ID, skeleton, face confidence/facing, floor position, dot where the person stands |
| `rfdetr.csv` | one row per person per frame (columns below) |
| `floor_map.png` | top-down map with every person's path in their ID colour |

CSV columns:

| Column | Meaning |
|---|---|
| `frame`, `time_s` | frame number, seconds from the start |
| `track_id` | **the person's ID** (after ReLink) — use this one |
| `tracker_id` | raw BoT-SORT ID (changes when the tracker loses someone) |
| `x1, y1, x2, y2`, `conf` | box in pixels, detection confidence |
| `face_conf`, `facing` | mean confidence of nose/eyes/ears; `front` / `side` / `back` / `unknown` |
| `head_x`, `head_y` | head position in pixels (empty if not visible) |
| `foot_x_px`, `foot_y_px`, `foot_from` | where the person stands, from `ankles` or the `box` bottom |
| `floor_x`, `floor_y`, `floor_unit` | floor position, in `m` (calibrated) or `px` |

## Tuning (`tracking/config.py`)

| Setting | Default | Raise it → | Lower it → |
|---|---|---|---|
| `SAME` | 0.82 | fewer "came back anywhere" links, fewer wrong ones | more links, more risk of joining two people |
| `LIKELY` | 0.70 | fewer links near the lost spot | more links, more risk |
| `COLOUR_VETO` | 0.6 | colour blocks fewer matches | colour blocks more matches |
| `REMEMBER_S` | 60 s | people can return later and keep their ID | lost people are forgotten sooner |
| `HOLD_S` | 10 s | tracker keeps hidden people longer | tracker gives new IDs sooner |
| `RETRY_S` | 5 s | longer re-check of new IDs by appearance | shorter |
| `BLIND_S`, `BLIND_DIST` | 2 s, 1.0 | position-only matching for hidden torsos is more lenient (risk of wrong IDs) | stricter (more new IDs) |
| `GALLERY`, `GALLERY_EVERY_S` | 15, 2 s | more / older views per person (more RAM) | fewer views |
| `POSTER_ZONES` | camera1 | add `"cameraX": [(x1, y1, x2, y2), ...]` for posters with people in a new camera | |

Change one setting at a time and compare the `IDs` / `re-linked` numbers and the video.
