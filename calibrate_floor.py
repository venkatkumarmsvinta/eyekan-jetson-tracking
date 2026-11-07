"""Floor calibration with a chessboard: image pixels -> floor position in metres.

1. Print a chessboard and lay it FLAT on the floor where the camera can see it (bigger board = better).
2. Record a short video or take a frame with the camera (it must not move afterwards).
3. Run:
       python calibrate_floor.py videos/camera1_chessboard.mp4 --name camera1 --cols 9 --rows 6 --square 0.10
   --cols / --rows = number of INNER corners (where 4 squares meet), not squares.
   A board with 10 x 7 squares has 9 x 6 inner corners. --square = side of one square in metres.
4. Check calibration/<name>_check.jpg: the drawn 0.5 m floor grid must lie flat on the floor.

Floor coordinates: (0, 0) is the first detected corner (marked on the check image); x runs along the board's
rows, y along its columns, in metres. Use --origin-x / --origin-y to shift it (e.g. to a corner of the store).

Saves calibration/<name>.json, which detect_rfdetr.py loads for the video with the same name.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def read_frame(source, frame_no):
    """One image from a picture file or from frame `frame_no` of a video."""
    img = cv2.imread(str(source))
    if img is not None:
        return img
    cap = cv2.VideoCapture(str(source))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Could not read {} (frame {})".format(source, frame_no))
    return img


def find_corners(img, cols, rows):
    """Inner chessboard corners in pixels, (rows*cols, 2), row by row; None if the board isn't found."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ok, corners = cv2.findChessboardCornersSB(gray, (cols, rows), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
    if not ok:  # older, more tolerant detector as a fallback
        ok, corners = cv2.findChessboardCorners(gray, (cols, rows), cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    return corners.reshape(-1, 2) if ok else None


def to_floor(H, points):
    """Pixel points (N, 2) -> floor points in metres (N, 2) with the image->floor homography H."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, np.asarray(H, dtype=np.float64)).reshape(-1, 2)


def draw_check(img, H, corners, extent, step=0.5):
    """Draw the detected corners and a floor grid (every `step` m) projected back into the image."""
    out = img.copy()
    H_inv = np.linalg.inv(np.asarray(H))
    (xmin, xmax), (ymin, ymax) = extent
    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)

    def to_img(pts):
        return cv2.perspectiveTransform(np.array(pts, dtype=np.float64).reshape(-1, 1, 2), H_inv).reshape(-1, 2)

    for x in xs:  # lines of constant x
        line = to_img([(x, y) for y in np.linspace(ymin, ymax, 50)]).astype(int)
        cv2.polylines(out, [line], False, (0, 255, 255) if abs(x) < 1e-9 else (0, 200, 0), 2)
    for y in ys:  # lines of constant y
        line = to_img([(x, y) for x in np.linspace(xmin, xmax, 50)]).astype(int)
        cv2.polylines(out, [line], False, (0, 255, 255) if abs(y) < 1e-9 else (0, 200, 0), 2)
    for x in xs:  # metre labels along y = 0
        if abs(x - round(x)) < 1e-9:
            px, py = to_img([(x, 0)])[0]
            cv2.putText(out, "{:g}m".format(x), (int(px) + 4, int(py) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    for p in corners.astype(int):
        cv2.circle(out, tuple(p), 4, (0, 0, 255), -1)
    o = corners[0].astype(int)
    cv2.circle(out, tuple(o), 10, (255, 0, 255), 3)
    cv2.putText(out, "(0,0)", (o[0] + 12, o[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image, or video with the chessboard lying on the floor")
    ap.add_argument("--name", help="camera name = video name the tracker uses (default: source file name)")
    ap.add_argument("--frame", type=int, default=0, help="frame number to use from a video")
    ap.add_argument("--cols", type=int, default=9, help="inner corners along the board's long side")
    ap.add_argument("--rows", type=int, default=6, help="inner corners along the board's short side")
    ap.add_argument("--square", type=float, default=0.10, help="side of one square in metres")
    ap.add_argument("--origin-x", type=float, default=0.0, help="floor x (m) of the first corner")
    ap.add_argument("--origin-y", type=float, default=0.0, help="floor y (m) of the first corner")
    ap.add_argument("--grid", type=float, default=3.0, help="check image: draw the grid this many metres around the board")
    args = ap.parse_args()

    source = Path(args.source)
    name = args.name or source.stem
    img = read_frame(source, args.frame)

    corners = find_corners(img, args.cols, args.rows)
    if corners is None:
        raise SystemExit("Chessboard ({} x {} inner corners) not found. Check --cols/--rows, that the whole board "
                         "is visible, flat and well lit, or try another --frame.".format(args.cols, args.rows))

    # floor position of each corner, in the same order OpenCV returns them (row by row)
    floor = np.array([(args.origin_x + c * args.square, args.origin_y + r * args.square)
                      for r in range(args.rows) for c in range(args.cols)], dtype=np.float64)
    H, _ = cv2.findHomography(corners.astype(np.float64), floor)
    err = np.linalg.norm(to_floor(H, corners) - floor, axis=1)

    out_dir = Path("calibration")
    out_dir.mkdir(exist_ok=True)
    calib = {
        "camera": name,
        "image_size": [img.shape[1], img.shape[0]],
        "pattern_inner_corners": [args.cols, args.rows],
        "square_m": args.square,
        "origin_m": [args.origin_x, args.origin_y],
        "corners_px": corners.round(2).tolist(),
        "H_image_to_floor": H.tolist(),
        "error_m": {"mean": round(float(err.mean()), 4), "max": round(float(err.max()), 4)},
    }
    (out_dir / "{}.json".format(name)).write_text(json.dumps(calib, indent=2))

    board_w, board_h = (args.cols - 1) * args.square, (args.rows - 1) * args.square
    extent = ((args.origin_x - args.grid, args.origin_x + board_w + args.grid),
              (args.origin_y - args.grid, args.origin_y + board_h + args.grid))
    cv2.imwrite(str(out_dir / "{}_check.jpg".format(name)), draw_check(img, H, corners, extent))

    print("Chessboard found: {} corners, board {:.2f} x {:.2f} m".format(len(corners), board_w, board_h))
    print("Error on the board: mean {:.1f} mm, max {:.1f} mm".format(err.mean() * 1000, err.max() * 1000))
    print("Saved: {}".format(out_dir / "{}.json".format(name)))
    print("Saved: {}  <- check that the green 0.5 m grid lies flat on the floor".format(out_dir / "{}_check.jpg".format(name)))


if __name__ == "__main__":
    main()
