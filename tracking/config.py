"""All settings in one place. Change values here, not in the other modules."""
import os
from pathlib import Path

# ---------- paths (every file the code reads or writes lives under DATA_DIR) ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("EYEKAN_DATA_DIR", ROOT / "data"))  # git-ignored; set EYEKAN_DATA_DIR to move it
VIDEO_DIR = DATA_DIR / "videos"        # recordings: <camera>.mp4
CALIB_DIR = DATA_DIR / "calibration"   # per-camera calibration: <camera>.json, store_points.csv
PHOTO_DIR = DATA_DIR / "photos"        # local copy of the staff face photos (from Supabase)
MODEL_DIR = DATA_DIR / "models"        # RF-DETR weights (downloaded on first use)
OUTPUT_DIR = DATA_DIR / "output"       # results: <video name>/rfdetr.mp4, rfdetr.csv, floor_map.png
LOG_DIR = DATA_DIR / "logs"            # cron logs (staff photo sync)

# ---------- models ----------
POSE_MODEL = "usyd-community/vitpose-base-simple"  # ViTPose-B, COCO 17 keypoints, downloaded on first use
REID_WEIGHTS = "osnet_ain_x1_0_msmt17.pt"  # boxmot OSNet re-ID model; osnet_x0_25_msmt17.pt is ~2x faster
CROP_W, CROP_H = 128, 256                   # re-ID model input size

# ---------- tracker / ID memory ----------
HOLD_S = 10.0      # seconds the tracker keeps a hidden person's box (its predicted position drifts after that)
REMEMBER_S = 60.0  # seconds ReLink remembers a lost person (e.g. walked out of the frame and came back)
YOUNG_S = 50.0     # hand-over: an ID younger than this can be renamed to an ID that just ended

# ---------- ReLink (see tracking/relink.py for how they are used) ----------
SAME = 0.82            # appearance similarity: this sure -> same person wherever they reappear
LIKELY = 0.70          # this similar AND near where they were lost -> same person
COLOUR_VETO = 0.6      # torso colours further apart (Bhattacharyya) -> never the same person (colour never matches)
COLOUR_MAX = 0.5       # colour-only rules (overlapping box, hand-over): no re-ID model
GALLERY = 15           # crops kept per person (~1.5 MB in RAM: 15 x 128x256 px)
GALLERY_EVERY_S = 2.0  # at most one new crop per this many seconds
DIST_START = 1.5       # box heights allowed right after the person is lost
DIST_PER_S = 2.0       # extra box heights allowed per second they are gone
DIST_MAX = 8.0         # never more than this many box heights
HANDOVER_DIST = 0.5    # hand-over: young ID within this many box heights of where the old ID ended
RETRY_S = 5.0          # a new ID can still be re-linked (by appearance) during its first RETRY_S seconds
ANYWHERE_S = 50.0      # gone longer than this: position is not checked
BLIND_S = 2.0          # no appearance (torso hidden): only people lost in the last BLIND_S seconds
BLIND_DIST = 1.0       # ... and only within this many box heights of where they were lost
BLIND_GAP_COST = 0.5   # ... and per second gone this much is added to the score (most recent loss wins)

# ---------- per camera ----------
# Posters showing people, per video (key = file name without .mp4), as [x1, y1, x2, y2] in pixels.
POSTER_ZONES = {
    "camera1": [(1400, 200, 1595, 590),   # ROOVY poster
                (1020, 200, 1125, 380)],  # trial-room poster
}

# ---------- COCO keypoints ----------
SEEN = 0.5  # a keypoint counts as visible above this confidence
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
FACE_POINTS = (0, 1, 2, 3, 4)
TORSO_POINTS = (5, 6, 11, 12)  # shoulders, hips
UPPER_POINTS = (0, 1, 2, 3, 4, 5, 6, 11, 12)  # face, shoulders, hips: used to find duplicate boxes
ANKLES = (15, 16)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

# ---------- output ----------
CSV_COLUMNS = ["frame", "time_s", "track_id", "tracker_id", "x1", "y1", "x2", "y2", "conf",
               "face_conf", "facing", "head_x", "head_y", "foot_x_px", "foot_y_px", "foot_from",
               "floor_x", "floor_y", "floor_unit", "staff", "staff_score"]

# ---------- multi-camera ----------
# Frame offset per camera so that the same frame index = the same moment, from the burned-in clocks
# (tools/sync_clocks.py): camera1 frame 6 = 04:01:46, camera6 frame 1 = 04:00:52
# -> camera6 frame = camera1 frame + 805 (+/-1 frame). Both record at exactly 15 fps (no drift).
CAMERA_GROUPS = {
    "store_test": {"camera1": 0, "camera6": 805},
}

# ---------- database (Supabase, see tracking/cloud/db.py) ----------
STAFF_PHOTO_BUCKET = "staff-photos"   # private bucket: {org_id}/{staff_id}/{n}.jpg
STAFF_PHOTO_DIR = PHOTO_DIR           # {emp_id}/{n}.jpg + embeddings.json

# ---------- face embeddings (InsightFace, see tracking/staff/faces.py) ----------
FACE_MODEL = "buffalo_l"     # SCRFD detector + ArcFace R50; weights are non-commercial only
FACE_DET_SIZE = 320          # photos are scaled so the longer side is this many pixels before face detection
STAFF_EMBED_FILE = STAFF_PHOTO_DIR / "embeddings.json"  # embeddings of the staff photos, next to them
FACE_EVERY_S = 0.33        # check a track's face at most this often (seconds)
FACE_MAX_PER_TRACK = 30    # faces collected per track, then its label is stable
FACE_MIN_CONF = 0.7        # ViTPose face confidence needed before trying
FACE_MIN_BOX_H = 250       # px: smaller people have too few face pixels
STAFF_THRESHOLD = 0.25     # cosine similarity for "staff"; camera1 test: staff >= 0.30, customer/poster <= 0.09
STAFF_MARGIN = 0.10        # best staff member must beat the second best by this much
