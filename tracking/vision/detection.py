"""Person detection with RF-DETR and box clean-up (posters, duplicate boxes)."""
import numpy as np
from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall
from rfdetr.main import HOSTED_MODELS
from rfdetr.util.coco_classes import COCO_CLASSES
from rfdetr.util.files import download_file

from ..config import MODEL_DIR, SEEN, UPPER_POINTS

MODELS = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "base": RFDETRBase, "large": RFDETRLarge}
PERSON = next(k for k, v in COCO_CLASSES.items() if v == "person")  # RF-DETR uses COCO category ids


def load_detector(size):
    """RF-DETR with its COCO weights in MODEL_DIR (downloaded there on first use).
    rfdetr itself only downloads to the current folder, so the download is done here."""
    weights = MODEL_DIR / "rf-detr-{}.pth".format(size)
    if not weights.exists():
        weights.parent.mkdir(parents=True, exist_ok=True)
        download_file(HOSTED_MODELS[weights.name], str(weights))
    return MODELS[size](pretrain_weights=str(weights))


def on_poster(box, zones, overlap=0.6):
    """True if most of the box lies inside a poster area (a real person walking past is mostly outside it)."""
    x1, y1, x2, y2 = box
    area = max(1.0, (x2 - x1) * (y2 - y1))
    for zx1, zy1, zx2, zy2 in zones:
        iw, ih = min(x2, zx2) - max(x1, zx1), min(y2, zy2) - max(y1, zy1)
        if iw > 0 and ih > 0 and iw * ih / area >= overlap:
            return True
    return False


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


def overlaps_other(k, boxes):
    """True if box k overlaps any other box (its colour then may contain the other person)."""
    x1, y1, x2, y2 = boxes[k]
    for j, (a1, b1, a2, b2) in enumerate(boxes):
        if j != k and min(x2, a2) > max(x1, a1) and min(y2, b2) > max(y1, b1):
            return True
    return False
