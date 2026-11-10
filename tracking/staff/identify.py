"""Who is this person? Staff recognition by face: now and then the face of each track is embedded (buffalo_l),
and the average of a track's faces is compared with the staff photo embeddings (data/photos/embeddings.json)."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from ..config import (FACE_DET_SIZE, FACE_EVERY_S, FACE_MAX_PER_TRACK, FACE_MIN_BOX_H, FACE_MIN_CONF, FACE_POINTS,
                     STAFF_EMBED_FILE, STAFF_MARGIN, STAFF_THRESHOLD)
from ..vision.keypoints import face_info, mean_point


def load_gallery():
    """emp_id -> mean embedding (length 1) of that person's staff photos; {} without embeddings.json."""
    try:
        data = json.loads(Path(STAFF_EMBED_FILE).read_text())
    except (OSError, ValueError):
        return {}
    by_emp = defaultdict(list)
    for key, photo in data.get("photos", {}).items():
        if photo["embedding"]:
            by_emp[key.split("/")[0]].append(photo["embedding"])
    gallery = {}
    for emp, embs in by_emp.items():
        c = np.mean(embs, axis=0)
        gallery[emp] = c / np.linalg.norm(c)
    return gallery


class StaffMatcher:
    """Collects face embeddings per track ID and tells which staff member (if any) the track is."""

    def __init__(self, fps):
        self.every = max(1, int(round(FACE_EVERY_S * fps)))
        self.staff = load_gallery()
        self.app = None
        if self.staff:  # no staff embeddings: don't load the face model at all
            from .faces import FaceEmbedder
            self.app = FaceEmbedder().app
        self.faces = defaultdict(list)  # track ID -> [(embedding, detector score)]
        self.last = {}                  # track ID -> frame of its last face check

    def update(self, frame_idx, frame, tid, box, kp_xy, kp_conf):
        """Embed the face of track `tid` in this (undrawn) frame, if it's time and the face is usable."""
        if self.app is None or len(self.faces[tid]) >= FACE_MAX_PER_TRACK:
            return
        if frame_idx - self.last.get(tid, -self.every) < self.every:
            return
        face, facing = face_info(kp_conf)
        head = mean_point(kp_xy, kp_conf, FACE_POINTS)
        h = box[3] - box[1]
        if facing != "front" or face < FACE_MIN_CONF or h < FACE_MIN_BOX_H or head is None:
            return
        self.last[tid] = frame_idx
        s = int(h * 0.4)  # square around the head
        x0, y0 = max(int(head[0]) - s // 2, 0), max(int(head[1]) - s // 2, 0)
        crop = frame[y0:y0 + s, x0:x0 + s]
        if crop.size == 0:
            return
        k = FACE_DET_SIZE / max(crop.shape[:2])
        hx, hy = (head[0] - x0) * k, (head[1] - y0) * k
        # the face must cover this person's head point: not the face of someone behind them
        own = [f for f in self.app.get(cv2.resize(crop, None, fx=k, fy=k))
               if f.bbox[0] <= hx <= f.bbox[2] and f.bbox[1] <= hy <= f.bbox[3]]
        if own:
            f = max(own, key=lambda f: f.det_score)
            self.faces[tid].append((f.normed_embedding, float(f.det_score)))

    def label(self, tid):
        """(emp_id, score) when the track's faces match one staff member clearly enough, else (None, best score)."""
        faces = self.faces.get(tid)
        if not faces:
            return None, 0.0
        w = np.array([s for _, s in faces])
        v = (np.stack([e for e, _ in faces]) * w[:, None]).sum(axis=0)
        v /= np.linalg.norm(v)
        scores = sorted(((float(v @ c), emp) for emp, c in self.staff.items()), reverse=True)
        best, emp = scores[0]
        second = scores[1][0] if len(scores) > 1 else 0.0
        if best >= STAFF_THRESHOLD and best - second >= STAFF_MARGIN:
            return emp, best
        return None, best
