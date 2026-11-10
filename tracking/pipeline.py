"""The per-frame loop: detect -> pose -> track -> ReLink -> CSV, video and floor map."""
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from .config import ANKLES, CSV_COLUMNS, FACE_POINTS, HOLD_S, POSTER_ZONES, REID_WEIGHTS, REMEMBER_S, TORSO_POINTS
from .detection import MODELS, PERSON, drop_duplicates, on_poster, overlaps_other
from .drawing import draw_person
from .floor import Floor, FloorMap
from .identify import StaffMatcher
from .keypoints import face_info, mean_point, person_crop, torso_colour, torso_visible
from .pose import Pose
from .relink import ReLink, apply_handovers
from .tracker import load_reid, make_tracker


def run(video, size="nano", conf=0.25, show=False, max_frames=0):
    """Track the people in `video`. Results go to output/<video name>/ (rfdetr.mp4, rfdetr.csv, floor_map.png)."""
    video_path = Path(video)
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
    model = MODELS[size]()  # downloads the COCO-trained weights on first use
    pose = Pose(device)
    tracker = make_tracker(fps, device)
    relink = ReLink(fps, load_reid(device))
    matcher = StaffMatcher(fps)
    print("Model: RF-DETR {} | pose: ViTPose on {} | tracker: BoT-SORT, lost boxes kept {:g} s | ReLink remembers {:g} s, "
          "re-ID {} | posters ignored: {} | floor calibration: {} | staff faces: {} people".format(
              size, device, HOLD_S, REMEMBER_S, REID_WEIGHTS, poster_zones or "none", floor.source, len(matcher.staff)))

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
            det = model.predict(rgb, threshold=conf)
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
            for k, t in enumerate(tracks):  # faces first, while the frame has no drawings on it
                d = int(t[7])
                matcher.update(frame_idx, frame, ids[k], boxes[k], kp_xy[d], kp_conf[d])

            now = {}  # ID -> floor position, for the map
            for k, t in enumerate(tracks):
                d, tid, score = int(t[7]), ids[k], float(t[5])
                x1, y1, x2, y2 = boxes[k]
                face, facing = face_info(kp_conf[d])
                head = mean_point(kp_xy[d], kp_conf[d], FACE_POINTS)
                ankles = mean_point(kp_xy[d], kp_conf[d], ANKLES)
                foot, foot_from = (ankles, "ankles") if ankles else (((x1 + x2) / 2, y2), "box")
                fx, fy = floor(*foot)
                staff, staff_score = matcher.label(tid)
                ids_seen.add(tid)
                floor_map.add(tid, fx, fy)
                now[tid] = (fx, fy)
                nd = 3 if floor.unit == "m" else 0
                csv_w.writerow([frame_idx, round(frame_idx / fps, 2), tid, int(t[4]), x1, y1, x2, y2, round(score, 3),
                                round(face, 3), facing, int(head[0]) if head else "", int(head[1]) if head else "",
                                int(foot[0]), int(foot[1]), foot_from, round(fx, nd), round(fy, nd), floor.unit,
                                staff or "", round(staff_score, 3)])
                n_boxes += 1
                pos = "x {:.2f} y {:.2f} m".format(fx, fy) if floor.unit == "m" else "x {:.0f} y {:.0f} px".format(fx, fy)
                draw_person(frame, boxes[k], tid, score, kp_xy[d], kp_conf[d], face, facing, foot, pos, staff)
            writer.write(frame)

            if show:
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
            if max_frames and frame_idx >= max_frames:
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
    for tid in sorted(matcher.faces):
        staff, score = matcher.label(tid)
        print("  ID {}: {} ({} faces, best score {:.2f})".format(tid, staff or "not staff", len(matcher.faces[tid]), score))
    for name in ("rfdetr.mp4", "rfdetr.csv", "floor_map.png"):
        print("Saved: {}".format(out_dir / name))
