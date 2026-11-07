"""What the keypoints tell about a person: points, facing, torso colour, crop for the re-ID model."""
import cv2
import numpy as np

from .config import (CROP_H, CROP_W, FACE_POINTS, L_EAR, L_EYE, L_SHOULDER, NOSE, R_EAR, R_EYE, R_SHOULDER,
                     SEEN, TORSO_POINTS)


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
