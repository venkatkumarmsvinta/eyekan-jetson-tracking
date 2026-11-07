"""Drawing a person on the output video: box, ID, skeleton, face info, floor position."""
import cv2

from .config import SEEN, SKELETON
from .floor import id_color


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
