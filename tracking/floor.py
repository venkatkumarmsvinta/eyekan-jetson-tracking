"""Floor position: pixels -> metres with a calibration (calibrate_floor.py), and the top-down floor map."""
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


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
