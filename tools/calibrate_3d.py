"""3D camera calibration from points measured in the store (no chessboard needed).

1. Measure points in the store, all from ONE origin (0, 0) on the floor; x and y along the floor, z = height,
   in metres. At least 6 per camera, at least 2 not on the floor (counter top, poster corners, sign tops).
   Write them in calibration/store_points.csv (one file for all cameras):
       name,x,y,z
       A1_counter_top_near_trial_room,2.10,3.40,0.95
       E1_floor_tile_corner,4.00,1.20,0
2. Click the points each camera can see (a zoom window helps to click exactly):
       python tools/calibrate_3d.py click camera1 --frame 3375
   left click = place the asked point, n = skip it (not visible), u = undo, q = save and quit.
   Saved in calibration/camera1_clicks.csv; run again to add points (already clicked ones are kept).
3. Compute the camera:
       python tools/calibrate_3d.py solve camera1 --frame 3375
   Prints the error per point, saves calibration/camera1.json (3x4 matrix, lens, camera position, floor
   homography for track.py) and calibration/camera1_check.jpg (0.5 m floor grid + 1.7 m posts: the grid must
   lie flat on the floor and the posts must look as tall as a person).
   Add --distortion for wide-angle cameras with bent straight lines (needs 8+ points).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.camera import Camera  # noqa: E402

CALIB_DIR = Path("calibration")


# ---------- files ----------

def read_points(path):
    """calibration/store_points.csv -> {name: (x, y, z)}"""
    with open(path, newline="") as f:
        return {r["name"].strip(): (float(r["x"]), float(r["y"]), float(r["z"])) for r in csv.DictReader(f)}


def read_clicks(path):
    """calibration/<camera>_clicks.csv -> {name: (u, v)}"""
    if not Path(path).exists():
        return {}
    with open(path, newline="") as f:
        return {r["name"]: (float(r["u"]), float(r["v"])) for r in csv.DictReader(f)}


def write_clicks(path, clicks):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "u", "v"])
        for name, (u, v) in clicks.items():
            w.writerow([name, round(u, 1), round(v, 1)])


def read_frame(camera, frame_no):
    cap = cv2.VideoCapture(str(Path("videos") / "{}.mp4".format(camera)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read frame {} of videos/{}.mp4".format(frame_no, camera))
    return img


# ---------- 1. clicking ----------

def click(camera, frame_no, points_path, redo=False):
    points = read_points(points_path)
    clicks_path = CALIB_DIR / "{}_clicks.csv".format(camera)
    clicks = {} if redo else read_clicks(clicks_path)
    todo = [n for n in points if n not in clicks]
    img = read_frame(camera, frame_no)
    h, w = img.shape[:2]
    scale = min(1440 / w, 810 / h)
    state = {"i": 0, "mouse": (w // 2, h // 2), "skipped": []}

    def on_mouse(event, x, y, flags, param):
        state["mouse"] = (int(x / scale), int(y / scale))
        if event == cv2.EVENT_LBUTTONDOWN and state["i"] < len(todo):
            clicks[todo[state["i"]]] = (x / scale, y / scale)
            state["i"] += 1

    cv2.namedWindow(camera)
    cv2.setMouseCallback(camera, on_mouse)
    while True:
        view = img.copy()
        for name, (u, v) in clicks.items():
            cv2.circle(view, (int(u), int(v)), 6, (0, 255, 0), 2)
            cv2.putText(view, name, (int(u) + 8, int(v) - 8), 0, 0.7, (0, 255, 0), 2)
        ask = todo[state["i"]] if state["i"] < len(todo) else None
        text = "click: {}   (n = skip, u = undo, q = save + quit)".format(ask) if ask else "all done - q = save + quit"
        view = cv2.resize(view, None, fx=scale, fy=scale)
        cv2.rectangle(view, (0, 0), (view.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(view, text, (8, 24), 0, 0.7, (0, 255, 255), 2)
        cv2.imshow(camera, view)
        # zoom window: 4x around the mouse, with a cross in the middle
        mx, my = state["mouse"]
        r = 40
        patch = cv2.copyMakeBorder(img, r, r, r, r, cv2.BORDER_CONSTANT)[my:my + 2 * r, mx:mx + 2 * r]
        patch = cv2.resize(patch, (320, 320), interpolation=cv2.INTER_NEAREST)
        cv2.line(patch, (160, 0), (160, 320), (0, 0, 255), 1)
        cv2.line(patch, (0, 160), (320, 160), (0, 0, 255), 1)
        cv2.imshow("zoom", patch)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("n") and ask:
            state["skipped"].append(ask)
            state["i"] += 1
        if key == ord("u") and state["i"] > 0:
            state["i"] -= 1
            clicks.pop(todo[state["i"]], None)
    cv2.destroyAllWindows()
    write_clicks(clicks_path, clicks)
    print("Saved {} clicks to {} (skipped: {})".format(len(clicks), clicks_path, ", ".join(state["skipped"]) or "none"))


# ---------- 2. solving ----------

def solve_camera(xyz, uv, image_size, distortion=False):
    """Lens + pose from >= 6 store points (metres) and their pixels. Principal point fixed at the image centre,
    square pixels; focal length (and k1 with distortion=True) estimated. Tries several focal guesses."""
    w, h = image_size
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO |
             cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    if not distortion:
        flags |= cv2.CALIB_FIX_K1
    obj = [np.asarray(xyz, np.float32).reshape(-1, 1, 3)]
    img = [np.asarray(uv, np.float32).reshape(-1, 1, 2)]
    best = None
    for f_guess in (0.5, 0.7, 1.0, 1.4, 2.0):
        K0 = np.array([[f_guess * w, 0, w / 2], [0, f_guess * w, h / 2], [0, 0, 1]], np.float64)
        try:
            rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj, img, (w, h), K0, np.zeros(5), flags=flags)
        except cv2.error:
            continue
        if best is None or rms < best[0]:
            best = (rms, K, dist, rvecs[0], tvecs[0])
    if best is None:
        raise SystemExit("Calibration failed: check the points (6+ per camera, not all on one line)")
    rms, K, dist, rvec, tvec = best
    return Camera(K, dist, rvec, tvec, (w, h)), rms


def draw_check(img, cam, names, xyz, uv):
    """0.5 m floor grid + 1.7 m posts around the points, clicked (green) vs re-projected (red) points."""
    out = img.copy()
    xyz = np.asarray(xyz)
    x0, y0 = np.floor(xyz[:, :2].min(axis=0)) - 1
    x1, y1 = np.ceil(xyz[:, :2].max(axis=0)) + 1

    def visible(p):  # in front of the camera
        return (cam.R @ np.asarray(p, float).reshape(3, 1) + cam.tvec)[2, 0] > 0.3

    def line(a, b, color, thick, n=20):
        pts = [a + (b - a) * t for t in np.linspace(0, 1, n)]
        pts = [p for p in pts if visible(p)]
        if len(pts) > 1:
            px = cam.project(np.array(pts)).astype(np.int32)
            cv2.polylines(out, [px], False, color, thick)

    for gx in np.arange(x0, x1 + 1e-6, 0.5):
        line(np.array([gx, y0, 0]), np.array([gx, y1, 0]), (255, 255, 0) if gx % 1 else (255, 128, 0), 1)
    for gy in np.arange(y0, y1 + 1e-6, 0.5):
        line(np.array([x0, gy, 0]), np.array([x1, gy, 0]), (255, 255, 0) if gy % 1 else (255, 128, 0), 1)
    for gx in np.arange(x0, x1 + 1e-6, 1.0):
        for gy in np.arange(y0, y1 + 1e-6, 1.0):
            line(np.array([gx, gy, 0]), np.array([gx, gy, 1.7]), (0, 0, 255), 2, n=2)
    proj = cam.project(xyz)
    for name, (u, v), (pu, pv) in zip(names, uv, proj):
        cv2.circle(out, (int(u), int(v)), 7, (0, 255, 0), 2)
        cv2.drawMarker(out, (int(pu), int(pv)), (0, 0, 255), cv2.MARKER_CROSS, 14, 2)
        cv2.putText(out, name, (int(u) + 9, int(v) - 9), 0, 0.6, (0, 255, 0), 2)
    cv2.putText(out, "green = clicked, red cross = calibration, blue = 0.5 m floor grid, red posts = 1.7 m",
                (10, out.shape[0] - 15), 0, 0.8, (0, 255, 255), 2)
    return out


def solve(camera, frame_no, points_path, distortion=False):
    points = read_points(points_path)
    clicks = read_clicks(CALIB_DIR / "{}_clicks.csv".format(camera))
    names = [n for n in clicks if n in points]
    if len(names) < 6:
        raise SystemExit("Need at least 6 clicked points with measurements, have {}".format(len(names)))
    xyz = np.array([points[n] for n in names], np.float64)
    uv = np.array([clicks[n] for n in names], np.float64)
    if np.ptp(xyz[:, 2]) < 0.3:
        print("WARNING: all points are at about the same height - head positions will be less accurate. "
              "Add points on the counter top, sign tops or poster corners.")
    img = read_frame(camera, frame_no)
    cam, rms = solve_camera(xyz, uv, (img.shape[1], img.shape[0]), distortion)
    cam.name = camera
    err = np.linalg.norm(cam.project(xyz) - uv, axis=1)

    print("\n{}: {} points, re-projection error RMS {:.1f} px".format(camera, len(names), rms))
    print("  focal length {:.0f} px, k1 {:+.3f}, camera at x {:.2f} y {:.2f} m, height {:.2f} m".format(
        cam.K[0, 0], cam.dist[0], *cam.position))
    for n, e in sorted(zip(names, err), key=lambda t: -t[1]):
        print("  {:<40} {:6.1f} px{}".format(n, e, "   <-- check this point (measurement or click)" if e > 15 else ""))
    if not 1.5 < cam.position[2] < 8:
        print("WARNING: camera height {:.2f} m looks wrong - check the measurements (units = metres?)".format(
            cam.position[2]))

    out_json = CALIB_DIR / "{}.json".format(camera)
    cam.save(out_json, {"rms_px": round(float(rms), 2), "points": {n: {"xyz": points[n], "uv": clicks[n],
                                                                        "error_px": round(float(e), 1)}
                                                                    for n, e in zip(names, err)},
                        "frame": frame_no, "distortion": distortion})
    out_img = CALIB_DIR / "{}_check.jpg".format(camera)
    cv2.imwrite(str(out_img), draw_check(img, cam, names, xyz, uv))
    print("Saved {} and {}".format(out_json, out_img))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["click", "solve"])
    ap.add_argument("camera", help="video name in videos/, e.g. camera1")
    ap.add_argument("--frame", type=int, default=0, help="frame to click on / draw the check on (pick an empty scene)")
    ap.add_argument("--points", default=str(CALIB_DIR / "store_points.csv"), help="measured points (name,x,y,z)")
    ap.add_argument("--redo", action="store_true", help="click: start again, forget earlier clicks")
    ap.add_argument("--distortion", action="store_true", help="solve: also estimate lens distortion k1 (8+ points)")
    args = ap.parse_args()
    if args.action == "click":
        click(args.camera, args.frame, args.points, args.redo)
    else:
        solve(args.camera, args.frame, args.points, args.distortion)


if __name__ == "__main__":
    main()
