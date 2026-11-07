"""People detection + tracking for store CCTV with RF-DETR (Roboflow, Apache 2.0).

Per frame:
    1. RF-DETR finds the person boxes (posters showing people are ignored).
    2. ViTPose (Hugging Face transformers, Apache 2.0) finds 17 keypoints in every box:
       nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles.
       Duplicate boxes on one person (same head/shoulder keypoints) are reduced to the most confident one.
    3. BoT-SORT (boxmot) gives every box a tracker ID. A hidden person is remembered for HOLD_S seconds.
    4. ReLink: when the tracker starts a new ID (often after two people overlapped, or someone came back into
       the frame), it is matched against the people lost in the last REMEMBER_S seconds by appearance (OSNet
       re-ID model from boxmot, run only at that moment on a few stored crops), body position, and torso colour
       as a veto. If they match, the person gets their old ID back.
       Hand-over: when an ID ends while a young ID (< YOUNG_S old) is at its place with the same clothing colour,
       the young ID is renamed to the old one (the tracker started it during an overlap while the
       old ID was still alive). The CSV and floor map are corrected for the frames before the rename too.
    5. Floor position = where the person stands: between the ankles, or the bottom-centre of the box if the
       ankles are hidden. With a floor calibration (calibration/<video_name>.json from calibrate_floor.py)
       it is in metres, otherwise in image pixels.

Run (in the RF-DETR environment):
    venv-rfdetr/bin/python detect_rfdetr.py videos/camera1.mp4 --show
Options: --size nano|small|medium|base|large  --conf 0.25  --max-frames N  --show (q to quit)

Outputs (in output/<video_name>/):
    rfdetr.mp4     - video: box + ID, skeleton, face confidence, dot where the person stands
    rfdetr.csv     - one row per person per frame (columns: see CSV_COLUMNS)
    floor_map.png  - top-down x, y graph of the floor with every person's path in their ID colour
"""
import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.util.coco_classes import COCO_CLASSES

MODELS = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "base": RFDETRBase, "large": RFDETRLarge}
PERSON = next(k for k, v in COCO_CLASSES.items() if v == "person")  # RF-DETR uses COCO category ids
POSE_MODEL = "usyd-community/vitpose-base-simple"  # ViTPose-B, COCO 17 keypoints, downloaded on first use
HOLD_S = 10.0  # seconds the tracker keeps a hidden person's box (its predicted position drifts after that)
REMEMBER_S = 60.0  # seconds ReLink remembers a lost person (e.g. walked out of the frame and came back)
YOUNG_S = 50.0  # hand-over: an ID younger than this can be renamed to an ID that just ended
REID_WEIGHTS = "osnet_ain_x1_0_msmt17.pt"  # boxmot OSNet re-ID model; osnet_x0_25_msmt17.pt is ~2x faster
CROP_W, CROP_H = 128, 256                   # re-ID model input size

# Posters showing people, per video (key = file name without .mp4), as [x1, y1, x2, y2] in pixels.
POSTER_ZONES = {
    "camera1": [(1400, 200, 1595, 590),   # ROOVY poster
                (1020, 200, 1125, 380)],  # trial-room poster
}

# COCO keypoints
SEEN = 0.5  # a keypoint counts as visible above this confidence
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SHOULDER, R_SHOULDER = 0, 1, 2, 3, 4, 5, 6
FACE_POINTS = (0, 1, 2, 3, 4)
TORSO_POINTS = (5, 6, 11, 12)  # shoulders, hips
UPPER_POINTS = (0, 1, 2, 3, 4, 5, 6, 11, 12)  # face, shoulders, hips: used to find duplicate boxes
ANKLES = (15, 16)
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]

CSV_COLUMNS = ["frame", "time_s", "track_id", "tracker_id", "x1", "y1", "x2", "y2", "conf",
               "face_conf", "facing", "head_x", "head_y", "foot_x_px", "foot_y_px", "foot_from",
               "floor_x", "floor_y", "floor_unit"]


# ---------- detection helpers ----------

def on_poster(box, zones, overlap=0.6):
    """True if most of the box lies inside a poster area (a real person walking past is mostly outside it)."""
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for zx1, zy1, zx2, zy2 in zones:
        iw, ih = min(x2, zx2) - max(x1, zx1), min(y2, zy2) - max(y1, zy1)
        if iw > 0 and ih > 0 and iw * ih / area >= overlap:
            return True
    return False


class Pose:
    """ViTPose: keypoints for given person boxes (top-down: one crop per person)."""

    def __init__(self, device):
        print('self',self);
        print('device',device)
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(POSE_MODEL)
        self.model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device).eval()

    def __call__(self, rgb, xyxy):
        """RGB image, boxes (N, 4) x1 y1 x2 y2 -> keypoints xy (N, 17, 2), confidence (N, 17)."""
        if len(xyxy) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        boxes = np.asarray(xyxy, np.float32).copy()
        boxes[:, 2:] -= boxes[:, :2]  # ViTPose wants x, y, width, height
        inputs = self.processor(rgb, boxes=[boxes], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_pose_estimation(out, boxes=[boxes])[0]
        return (np.stack([r["keypoints"].cpu().numpy() for r in res]),
                np.stack([r["scores"].cpu().numpy() for r in res]))


def iou(a, b):
    iw, ih = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def drop_duplicates(det, kp_xy, kp_conf, min_iou=0.5, max_kp_dist=0.1):
    """RF-DETR sometimes puts two boxes on one person, and the tracker then makes a second ID for them.
    Two boxes are the same person when they overlap (IoU >= min_iou) AND their head/shoulder/hip keypoints
    seen in both (at least 3) are on average closer than max_kp_dist box heights. Keeps the most confident box.
    Two real people standing close have different heads and shoulders, so they are not merged."""
    keep = []
    for i in np.argsort(-det.confidence):
        same = False
        for j in keep:
            if iou(det.xyxy[i], det.xyxy[j]) < min_iou:
                continue
            common = [p for p in UPPER_POINTS if kp_conf[i][p] > SEEN and kp_conf[j][p] > SEEN]
            if len(common) < 3:
                continue
            h = max(det.xyxy[i][3] - det.xyxy[i][1], det.xyxy[j][3] - det.xyxy[j][1], 1)
            if np.linalg.norm(kp_xy[i][common] - kp_xy[j][common], axis=1).mean() / h < max_kp_dist:
                same = True
                break
        if not same:
            keep.append(i)
    keep = np.sort(np.array(keep, dtype=int))
    return det[keep], kp_xy[keep], kp_conf[keep], len(det) - len(keep)


def mean_point(kp_xy, kp_conf, idx):
    """Mean of the visible keypoints in idx, None if none is visible."""
    vis = [i for i in idx if kp_conf[i] > SEEN]
    return tuple(float(v) for v in kp_xy[vis].mean(axis=0)) if vis else None


def face_info(kp_conf):
    """(face confidence = mean of nose/eyes/ears, facing: front / side / back / unknown)."""
    face = float(np.mean(kp_conf[list(FACE_POINTS)]))
    nose, eyes = kp_conf[NOSE] > SEEN, max(kp_conf[L_EYE], kp_conf[R_EYE]) > SEEN
    ears = max(kp_conf[L_EAR], kp_conf[R_EAR]) > SEEN
    if nose and eyes:
        return face, "front"
    if ears or nose or eyes:
        return face, "side"
    if kp_conf[L_SHOULDER] > SEEN or kp_conf[R_SHOULDER] > SEEN:  # shoulders but no face: turned away
        return face, "back"
    return face, "unknown"


def torso_colour(frame, kp_xy, kp_conf):
    """Hue/saturation histogram of the torso (rectangle around the visible shoulders and hips, shrunk a bit
    so it stays on the clothes). None when fewer than 3 of those points are visible."""
    vis = [i for i in TORSO_POINTS if kp_conf[i] > SEEN]
    if len(vis) < 3:
        return None
    pts = kp_xy[vis]
    (x1, y1), (x2, y2) = pts.min(axis=0), pts.max(axis=0)
    dx, dy = 0.15 * (x2 - x1), 0.10 * (y2 - y1)
    x1, x2, y1, y2 = int(x1 + dx), int(x2 - dx), int(y1 + dy), int(y2 - dy)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None
    hsv = cv2.cvtColor(frame[max(y1, 0):y2, max(x1, 0):x2], cv2.COLOR_BGR2HSV)
    if hsv.size == 0:
        return None
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def torso_visible(kp_conf):
    """At least 3 of shoulders/hips seen: the box shows enough of the body to judge appearance."""
    return sum(kp_conf[i] > SEEN for i in TORSO_POINTS) >= 3


def person_crop(frame, box):
    """The person's box cut out at the re-ID input size (cheap: no model runs here)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    c = frame[max(y1, 0):min(y2, h), max(x1, 0):min(x2, w)]
    return cv2.resize(c, (CROP_W, CROP_H)) if c.size else None


def overlaps_other(k, boxes):
    """True if box k overlaps any other box (its colour then may contain the other person)."""
    x1, y1, x2, y2 = boxes[k]
    for j, (a1, b1, a2, b2) in enumerate(boxes):
        if j != k and min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1):
            return True
    return False


# ---------- tracking ----------

def make_tracker(fps, device):
    """BoT-SORT without appearance model (fewest ID changes in our tests). track_buffer is given for
    30 fps and scaled by boxmot, so this keeps lost tracks for HOLD_S seconds at any frame rate."""
    from boxmot import BotSort
    return BotSort(reid_weights=Path("unused.pt"), device=device, half=False, with_reid=False,
                   track_high_thresh=0.25, new_track_thresh=0.25, track_low_thresh=0.1,
                   track_buffer=int(HOLD_S * 30), match_thresh=0.8, frame_rate=round(fps),
                   cmc_method="sof")  # fixed cameras: camera-motion correction finds nothing, cheapest method


class ReLink:
    """Keeps a person's ID when the tracker gives them a new one.

    Appearance: while a person is tracked, a small crop of them is kept every GALLERY_EVERY_S seconds (only when
       their box overlaps nobody and the torso is visible), at most GALLERY crops (front, side, back ...), with
       its torso colour. The re-ID model (OSNet) does NOT run during normal tracking. It runs only when the
       tracker gives a NEW ID whose box overlaps nobody and someone lost could match: on the new person's crop,
       and once on each lost candidate's stored crops (cached). A new box that overlaps someone (its crop shows
       two people) uses torso colour + position as before (colour distance <= COLOUR_MAX).
       sim = best cosine similarity between the two sets of crops. Colour only vetoes: torso colours further
       apart than COLOUR_VETO are never the same person.
    Re-link: a new tracker ID gets the ID of a person lost in the last REMEMBER_S seconds, when
       - sim >= SAME, wherever they are (walked out of the frame and came back elsewhere), or
       - sim >= LIKELY and the body point is close to where the lost person was last seen (allowed distance
         grows with the time they were gone: DIST_START box heights + DIST_PER_S per second, at most DIST_MAX;
         not checked at all after ANYWHERE_S seconds).
       If the torso is mostly hidden (e.g. bent behind a rack, only the head visible), there is no appearance:
       only people lost in the last BLIND_S seconds qualify, by position alone, the most recently lost first.
       Tried again during the new ID's first RETRY_S seconds, because a person walking back into the frame is
       only partly visible at first.
    Hand-over: when an ID ends and a young ID (born after it, < YOUNG_S old) is within HANDOVER_DIST box
       heights of where it ended with a similar torso colour (<= COLOUR_MAX, or unknown), the young ID becomes
       the old ID. This is the overlap case: the tracker started the new ID while the old one was still alive.
       No re-ID model here: during an overlap a crop can show two people and swap the IDs."""

    SAME = 0.82         # appearance similarity: this sure -> same person wherever they reappear
    LIKELY = 0.70       # this similar AND near where they were lost -> same person
    COLOUR_VETO = 0.6   # torso colours further apart (Bhattacharyya) -> never the same person (colour never matches)
    COLOUR_MAX = 0.5    # colour-only rules (overlapping box, hand-over): as before, no re-ID model
    GALLERY = 15        # crops kept per person (~1.5 MB in RAM: 15 x 128x256 px)
    GALLERY_EVERY_S = 2.0  # at most one new crop per this many seconds
    DIST_START = 1.5  # box heights allowed right after the person is lost
    DIST_PER_S = 2.0  # extra box heights allowed per second they are gone
    DIST_MAX = 8.0    # never more than this many box heights
    HANDOVER_DIST = 0.5
    RETRY_S = 5.0     # a new ID can still be re-linked (by appearance) during its first RETRY_S seconds
    ANYWHERE_S = 50.0  # gone longer than this: position is not checked
    BLIND_S = 2.0      # no appearance (torso hidden): only people lost in the last BLIND_S seconds
    BLIND_DIST = 1.0   # ... and only within this many box heights of where they were lost
    BLIND_GAP_COST = 0.5  # ... and per second gone this much is added to the score (most recent loss wins)

    def __init__(self, fps, reid):
        self.fps, self.hold, self.young = fps, int(REMEMBER_S * fps), int(YOUNG_S * fps)
        self.retry = int(self.RETRY_S * fps)
        self.every = max(1, int(self.GALLERY_EVERY_S * fps))
        self.reid = reid
        self.final = {}      # tracker ID -> ID shown
        self.last = {}       # ID shown -> {"frame", "point", "height", "views": [(crop, colour)], "added"}
        self.vecs = {}       # ID shown -> ("added" frame, appearance vectors of its views): computed only when needed
        self.born = {}       # brand-new ID shown -> frame it first appeared
        self.handovers = []  # (young ID, old ID, frame the young ID was born), in order
        self.relinked = self.handed_over = 0
        self.n_embedded = 0  # crops the re-ID model has processed (its real cost)

    def near(self, dist, gap_s):
        """Close enough to where the lost person was last seen, for the time they were gone."""
        return gap_s >= self.ANYWHERE_S or dist <= min(self.DIST_MAX, self.DIST_START + self.DIST_PER_S * gap_s)

    def embed(self, crops):
        """THE ONLY PLACE THE RE-ID MODEL RUNS: appearance vectors of the crops, in one batch."""
        if not crops:
            return []
        strip = np.hstack(crops)  # crops side by side, one box each
        boxes = np.array([[i * CROP_W, 0, (i + 1) * CROP_W, CROP_H] for i in range(len(crops))], np.float32)
        self.n_embedded += len(crops)
        return list(self.reid.get_features(boxes, strip))

    def gallery(self, fid):
        """(appearance vectors, colours) of a person's stored views. The vectors are computed on first need and
        cached; a lost person's views don't change, so that happens once per lost person."""
        s = self.last.get(fid)
        if not s or not s["views"]:
            return [], []
        cached = self.vecs.get(fid)
        if cached is None or cached[0] != s["added"]:
            cached = self.vecs[fid] = (s["added"], self.embed([crop for crop, _ in s["views"]]))
        return cached[1], [colour for _, colour in s["views"]]

    def colours(self, fid):
        """Torso colours of a person's stored views (no model)."""
        s = self.last.get(fid)
        return [c for _, c in s["views"]] if s else []

    @staticmethod
    def colour_dist(a, b):
        """Distance between the AVERAGE torso colour of two lists of colours (like the old averaged colour;
        the closest pair is too lenient: some pair is always similar by chance). None if unknown."""
        a, b = [x for x in a if x is not None], [y for y in b if y is not None]
        if not a or not b:
            return None
        return cv2.compareHist(np.mean(a, axis=0).astype(np.float32), np.mean(b, axis=0).astype(np.float32),
                               cv2.HISTCMP_BHATTACHARYYA)

    def current(self, k, people, cache):
        """(appearance vectors, colours) of person k's view in this frame; embedded at most once per frame.
        Box overlapping someone: the crop shows two people, so NO re-ID, colour only."""
        if k not in cache:
            view, clean = people[k][3], people[k][4]
            if view is None:
                cache[k] = ([], [])
            elif not clean:
                cache[k] = ([], [view[1]])
            else:
                cache[k] = (self.embed([view[0]]), [view[1]])
        return cache[k]

    def young_views(self, young, k, people, cache):
        """A young ID's stored views, or its current view if none is stored yet."""
        g = self.gallery(young)
        return g if g[0] else self.current(k, people, cache)

    @staticmethod
    def match(a, b):
        """(best appearance similarity, lowest torso-colour distance) between two (vectors, colours); None = unknown."""
        sim = float(np.max(np.stack(a[0]) @ np.stack(b[0]).T)) if len(a[0]) and len(b[0]) else None
        ds = [cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA)
              for x in a[1] if x is not None for y in b[1] if y is not None]
        return sim, (min(ds) if ds else None)

    def same(self, sim, cdist, dist, gap_s):
        """Same person? Colour can only say no; appearance (+ position when less sure) says yes."""
        if cdist is not None and cdist > self.COLOUR_VETO:
            return False
        return sim >= self.SAME or (sim >= self.LIKELY and self.near(dist, gap_s))

    def __call__(self, frame_idx, people):
        """people: [(tracker_id, body_point, box_height, view or None, clean), ...] -> [ID shown, ...]
        view = (crop for the re-ID model, torso colour or None); None when the torso is mostly hidden."""
        ids = [self.final.get(p[0]) for p in people]
        in_use = {i for i in ids if i is not None}
        lost = {fid: s for fid, s in self.last.items()
                if fid not in in_use and 0 < frame_idx - s["frame"] <= self.hold}
        cache = {}  # k -> this frame's appearance of person k (computed only if someone lost could match)

        # re-link: score every (new tracker ID, lost person) pair that passes the checks, best pairs first
        pairs = []
        for k, (tid, point, height, view, _) in enumerate(people):
            if ids[k] is not None:
                continue
            for fid, s in lost.items():
                gap_s = (frame_idx - s["frame"]) / self.fps
                dist = np.hypot(point[0] - s["point"][0], point[1] - s["point"][1]) / max(height, s["height"], 1)
                mine = self.current(k, people, cache)
                if mine[0] and s["views"]:  # new ID, clean box: appearance (re-ID model)
                    sim, cdist = self.match(mine, self.gallery(fid))
                    if self.same(sim, cdist, dist, gap_s):
                        pairs.append((dist + 2 * (1 - sim), k, fid))
                    continue
                cdist = self.colour_dist(mine[1], self.colours(fid))
                if cdist is None:
                    # partial view (e.g. bent behind a rack, only the head visible): no appearance, so
                    # position alone decides. Only people lost a moment ago, the most recently lost first, so
                    # the box does not take the ID of someone else who was hidden at the same spot earlier.
                    if gap_s <= self.BLIND_S and dist <= self.BLIND_DIST:
                        pairs.append((dist + self.BLIND_GAP_COST * gap_s, k, fid))
                    continue
                if self.near(dist, gap_s) and cdist <= self.COLOUR_MAX:  # overlapping box: colour, as before
                    pairs.append((dist + cdist, k, fid))
        for _, k, fid in sorted(pairs):
            if ids[k] is None and fid in lost:
                ids[k] = fid
                self.final[people[k][0]] = fid
                del lost[fid]
                self.relinked += 1
        # brand-new people keep their tracker ID (always higher than any earlier one, so never a clash)
        for k, (tid, *_rest) in enumerate(people):
            if ids[k] is None:
                ids[k] = self.final[tid] = tid
                self.born[tid] = frame_idx

        # late re-link: a person coming back into the frame is only partly visible in their first frames, so
        # keep trying for RETRY_S seconds once their appearance is known. Renamed like a hand-over (CSV fixed later).
        in_use = set(ids)
        pairs = []
        for k, (tid, point, height, view, _) in enumerate(people):
            young = ids[k]
            born = self.born.get(young)
            if born is None or born == frame_idx or frame_idx - born > self.retry:
                continue
            candidates = [fid for fid, s in self.last.items()
                          if fid not in in_use and s["views"] and 0 < born - s["frame"] <= self.hold]
            if not candidates:  # nobody lost: the re-ID model does not run
                continue
            mine = self.young_views(young, k, people, cache)
            if not mine[0]:  # no appearance yet: too risky to match on position alone
                continue
            for fid in candidates:
                s = self.last[fid]
                gap_s = (frame_idx - s["frame"]) / self.fps
                dist = np.hypot(point[0] - s["point"][0], point[1] - s["point"][1]) / max(height, s["height"], 1)
                sim, cdist = self.match(mine, self.gallery(fid))
                if self.same(sim, cdist, dist, gap_s):
                    pairs.append((dist + 2 * (1 - sim), k, fid))
        done = set()
        for _, k, old in sorted(pairs):
            young = ids[k]
            if k in done or old in done:
                continue
            for t, f in self.final.items():
                if f == young:
                    self.final[t] = old
            self.handovers.append((young, old, self.born.pop(young)))
            self.last.pop(young, None)
            self.vecs.pop(young, None)
            ids[k] = old
            done.update((k, old))
            self.relinked += 1

        # hand-over: IDs that were there in the previous frame and are gone now
        for old in [fid for fid, s in self.last.items() if s["frame"] == frame_idx - 1 and fid not in ids]:
            s = self.last[old]
            best = None
            for k, (tid, point, height, view, _) in enumerate(people):
                young = ids[k]
                born = self.born.get(young)
                if born is None or frame_idx - born > self.young or born <= self.born.get(old, -1):
                    continue
                dist = np.hypot(point[0] - s["point"][0], point[1] - s["point"][1]) / max(height, s["height"], 1)
                # overlap case: colour + position only, no re-ID model (a crop here may show two people)
                cdist = self.colour_dist(self.colours(young) or [view[1] if view else None], self.colours(old))
                cdist = 0.0 if cdist is None else cdist  # colour unknown: position decides (as before)
                if dist <= self.HANDOVER_DIST and cdist <= self.COLOUR_MAX and (best is None or dist + cdist < best[0]):
                    best = (dist + cdist, k)
            if best is not None:
                young = ids[best[1]]
                for t, f in self.final.items():
                    if f == young:
                        self.final[t] = old
                self.handovers.append((young, old, self.born.pop(young)))
                self.last.pop(young, None)
                self.vecs.pop(young, None)
                ids[best[1]] = old
                self.handed_over += 1

        # keep a crop now and then (no model runs here), only from boxes that overlap nobody
        for k, (tid, point, height, view, clean) in enumerate(people):
            s = self.last.get(ids[k], {"views": [], "added": -self.every})
            views, added = s["views"], s["added"]
            if clean and view is not None and frame_idx - added >= self.every:
                views, added = (views + [view])[-self.GALLERY:], frame_idx
            self.last[ids[k]] = {"frame": frame_idx, "point": point, "height": height, "views": views, "added": added}
        # forget people gone too long to be re-linked (their crops would only use memory)
        for fid in [f for f, s in self.last.items() if frame_idx - s["frame"] > self.hold + self.retry]:
            del self.last[fid]
            self.vecs.pop(fid, None)
        return ids


def apply_handovers(csv_path, floor_map, handovers):
    """Rename every young ID to the old ID it was handed over to, also in the rows written before the
    hand-over. From the young ID's birth on, the young rows replace the old ID's rows (in the overlap the old
    box had already slid off the person)."""
    if not handovers:
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    ti, fi = header.index("track_id"), header.index("frame")
    for young, old, born in handovers:
        rows = [r for r in rows if not (int(r[ti]) == old and int(r[fi]) >= born)]
        for r in rows:
            if int(r[ti]) == young:
                r[ti] = str(old)
        floor_map.paths[old].extend(floor_map.paths.pop(young, []))
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)


# ---------- floor ----------

class Floor:
    """Pixel -> floor position, in metres with a calibration (calibrate_floor.py), otherwise in pixels."""

    def __init__(self, name):
        path = Path("calibration") / "{}.json".format(name)
        self.H = np.array(json.loads(path.read_text())["H_image_to_floor"]) if path.exists() else None
        self.unit = "m" if self.H is not None else "px"
        self.source = str(path) if self.H is not None else "none (positions in image pixels)"

    def __call__(self, x, y):
        if self.H is None:
            return float(x), float(y)
        fx, fy = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float64), self.H)[0, 0]
        return float(fx), float(fy)


def id_color(tid):
    """A fixed, clearly visible colour (BGR) per ID."""
    hue = (tid * 47) % 180
    return tuple(int(c) for c in cv2.cvtColor(np.uint8([[[hue, 220, 255]]]), cv2.COLOR_HSV2BGR)[0, 0])


class FloorMap:
    """Top-down x, y graph of the floor with every person's path."""

    def __init__(self, unit, size=800, margin=60):
        self.unit, self.size, self.margin = unit, size, margin
        self.paths = defaultdict(list)  # ID -> [(x, y), ...]

    def add(self, tid, x, y):
        self.paths[tid].append((x, y))

    def draw(self, now=None):
        """The map image. `now` = {ID: (x, y)} of people in the current frame, drawn as big dots."""
        img = np.full((self.size, self.size, 3), 255, np.uint8)
        pts = [p for path in self.paths.values() for p in path]
        if not pts:
            cv2.putText(img, "no positions yet", (self.margin, self.size // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
            return img
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        min_span = 4.0 if self.unit == "m" else 400.0  # show at least 4 m / 400 px
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        span = max(max(xs) - min(xs), max(ys) - min(ys), min_span) * 1.1
        x0, y0 = cx - span / 2, cy - span / 2
        scale = (self.size - 2 * self.margin) / span

        def to_px(x, y):  # x to the right, y down (same direction as in the camera image / on the chessboard)
            return int(self.margin + (x - x0) * scale), int(self.margin + (y - y0) * scale)

        step = 1.0 if self.unit == "m" else 100.0  # grid every 1 m / 100 px, labelled
        while span / step > 12:
            step *= 2
        g = np.floor(x0 / step) * step
        while g <= x0 + span:
            px = to_px(g, y0)[0]
            cv2.line(img, (px, self.margin), (px, self.size - self.margin), (225, 225, 225), 1)
            cv2.putText(img, "{:g}".format(g), (px - 10, self.size - self.margin + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
            g += step
        g = np.floor(y0 / step) * step
        while g <= y0 + span:
            py = to_px(x0, g)[1]
            cv2.line(img, (self.margin, py), (self.size - self.margin, py), (225, 225, 225), 1)
            cv2.putText(img, "{:g}".format(g), (5, py + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (90, 90, 90), 1)
            g += step
        cv2.rectangle(img, (self.margin, self.margin), (self.size - self.margin, self.size - self.margin), (120, 120, 120), 1)
        cv2.putText(img, "x ({})".format(self.unit), (self.size // 2 - 20, self.size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "y ({})".format(self.unit), (5, self.margin - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        for tid, path in self.paths.items():
            line = np.array([to_px(x, y) for x, y in path], np.int32)
            cv2.polylines(img, [line], False, id_color(tid), 2)
            cv2.putText(img, str(tid), tuple(line[-1] + (5, -5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, id_color(tid), 2)
        for tid, (x, y) in (now or {}).items():
            cv2.circle(img, to_px(x, y), 8, id_color(tid), -1)
            cv2.circle(img, to_px(x, y), 8, (0, 0, 0), 1)
        return img


# ---------- drawing ----------

def draw_person(frame, box, tid, conf, kp_xy, kp_conf, face, facing, foot, pos_text):
    x1, y1, x2, y2 = box
    color = id_color(tid)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    for a, b in SKELETON:
        if kp_conf[a] > SEEN and kp_conf[b] > SEEN:
            cv2.line(frame, tuple(map(int, kp_xy[a])), tuple(map(int, kp_xy[b])), color, 3)
    for i, (x, y) in enumerate(kp_xy):
        if kp_conf[i] > SEEN:
            cv2.circle(frame, (int(x), int(y)), 4, (255, 255, 255), -1)
            cv2.circle(frame, (int(x), int(y)), 4, color, 1)
    cv2.circle(frame, (int(foot[0]), int(foot[1])), 7, color, -1)  # where the person stands
    cv2.putText(frame, "id {} {:.2f}".format(tid, conf), (x1 + 2, max(y1 - 6, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, "face {:.2f} {}".format(face, facing), (x1 + 2, max(y1 - 30, 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, pos_text, (x1 + 2, min(y2 + 22, frame.shape[0] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--size", default="nano", choices=list(MODELS), help="RF-DETR model size")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    ap.add_argument("--show", action="store_true", help="show the video and the floor map live (q to quit)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit("Video not found: {}".format(video_path))
    out_dir = Path("output") / video_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video: {}  {}x{} @ {:.1f} fps, {} frames".format(video_path, w, h, fps, total))

    import torch
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    poster_zones = POSTER_ZONES.get(video_path.stem, [])
    floor = Floor(video_path.stem)
    floor_map = FloorMap(floor.unit)
    model = MODELS[args.size]()  # downloads the COCO-trained weights on first use
    pose = Pose(device)
    tracker = make_tracker(fps, device)
    from boxmot.reid.core.auto_backend import ReidAutoBackend
    from boxmot.utils import WEIGHTS
    reid = ReidAutoBackend(weights=WEIGHTS / REID_WEIGHTS,  # downloaded on first use; boxmot wants "0" for CUDA
                           device="0" if device == "cuda" else device, half=device == "cuda").model
    relink = ReLink(fps, reid)
    print("Model: RF-DETR {} | pose: ViTPose on {} | tracker: BoT-SORT, lost boxes kept {:g} s | ReLink remembers {:g} s, "
          "re-ID {} | posters ignored: {} | floor calibration: {}".format(args.size, device, HOLD_S, REMEMBER_S, REID_WEIGHTS,
                                                                        poster_zones or "none", floor.source))

    writer = cv2.VideoWriter(str(out_dir / "rfdetr.mp4"), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if not writer.isOpened():  # no H.264 encoder: mp4v (may not play in QuickTime)
        writer = cv2.VideoWriter(str(out_dir / "rfdetr.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    csv_file = open(out_dir / "rfdetr.csv", "w", newline="")
    csv_w = csv.writer(csv_file)
    csv_w.writerow(CSV_COLUMNS)

    ids_seen, n_boxes, n_dupes, frame_idx = set(), 0, 0, 0
    t0 = time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = frame[:, :, ::-1].copy()  # RF-DETR and ViTPose expect RGB
            det = model.predict(rgb, threshold=args.conf)
            det = det[det.class_id == PERSON]
            if poster_zones and len(det):
                det = det[np.array([not on_poster(b, poster_zones) for b in det.xyxy])]
            kp_xy, kp_conf = pose(rgb, det.xyxy)
            det, kp_xy, kp_conf, dropped = drop_duplicates(det, kp_xy, kp_conf)
            n_dupes += dropped

            dets = np.column_stack([det.xyxy, det.confidence, np.zeros(len(det))]).astype(np.float32).reshape(-1, 6)
            tracks = tracker.update(dets, frame)  # rows: x1, y1, x2, y2, tracker id, conf, class, detection index
            boxes = [tuple(int(v) for v in t[:4]) for t in tracks]

            people = []  # per track: everything ReLink needs
            for k, t in enumerate(tracks):
                d = int(t[7])
                x1, y1, x2, y2 = boxes[k]
                body = mean_point(kp_xy[d], kp_conf[d], TORSO_POINTS) or ((x1 + x2) / 2, (y1 + y2) / 2)
                view = None  # (crop, torso colour), only when enough of the body is visible
                if torso_visible(kp_conf[d]):
                    crop = person_crop(frame, boxes[k])
                    view = None if crop is None else (crop, torso_colour(frame, kp_xy[d], kp_conf[d]))
                people.append((int(t[4]), body, y2 - y1, view, not overlaps_other(k, boxes)))
            ids = relink(frame_idx, people)

            now = {}  # ID -> floor position, for the map
            for k, t in enumerate(tracks):
                d, tid, conf = int(t[7]), ids[k], float(t[5])
                x1, y1, x2, y2 = boxes[k]
                face, facing = face_info(kp_conf[d])
                head = mean_point(kp_xy[d], kp_conf[d], FACE_POINTS)
                ankles = mean_point(kp_xy[d], kp_conf[d], ANKLES)
                foot, foot_from = (ankles, "ankles") if ankles else (((x1 + x2) / 2, y2), "box")
                fx, fy = floor(*foot)
                ids_seen.add(tid)
                floor_map.add(tid, fx, fy)
                now[tid] = (fx, fy)
                nd = 3 if floor.unit == "m" else 0
                csv_w.writerow([frame_idx, round(frame_idx / fps, 2), tid, int(t[4]), x1, y1, x2, y2, round(conf, 3),
                                round(face, 3), facing, int(head[0]) if head else "", int(head[1]) if head else "",
                                int(foot[0]), int(foot[1]), foot_from, round(fx, nd), round(fy, nd), floor.unit])
                n_boxes += 1
                pos = "x {:.2f} y {:.2f} m".format(fx, fy) if floor.unit == "m" else "x {:.0f} y {:.0f} px".format(fx, fy)
                draw_person(frame, boxes[k], tid, conf, kp_xy[d], kp_conf[d], face, facing, foot, pos)
            writer.write(frame)

            if args.show:
                cv2.imshow("rf-detr", frame)
                cv2.imshow("floor map", floor_map.draw(now))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            if frame_idx % 100 == 0:
                print("frame {}/{}  {:.1f} fps  boxes: {}  IDs: {}  re-linked: {}  handed over: {}  duplicates: {}  "
                      "re-ID crops: {}".format(frame_idx, total, (frame_idx + 1) / max(time.time() - t0, 1e-6), n_boxes,
                                               len(ids_seen) - relink.handed_over, relink.relinked, relink.handed_over,
                                               n_dupes, relink.n_embedded))
            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        writer.release()
        csv_file.close()
        apply_handovers(out_dir / "rfdetr.csv", floor_map, relink.handovers)
        cv2.imwrite(str(out_dir / "floor_map.png"), floor_map.draw())
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    print("Done. {} frames in {:.0f} s ({:.1f} fps), boxes: {}, unique IDs: {}, re-linked: {}, handed over: {}, "
          "duplicate boxes removed: {}, re-ID ran on {} crops".format(
              frame_idx, elapsed, frame_idx / max(elapsed, 1e-6), n_boxes, len(ids_seen) - relink.handed_over,
              relink.relinked, relink.handed_over, n_dupes, relink.n_embedded))
    for young, old, born in relink.handovers:
        print("  hand-over at {:.2f} s: ID {} -> {}".format(born / fps, young, old))
    for name in ("rfdetr.mp4", "rfdetr.csv", "floor_map.png"):
        print("Saved: {}".format(out_dir / name))


if __name__ == "__main__":
    main()
