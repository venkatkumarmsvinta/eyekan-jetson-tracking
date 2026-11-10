"""Synchronise two recordings with the clock burned into the video (top right, white digits).

Step 1 - find the clock ticks and save the clock at the first tick of each video:
    python tools/sync_clocks.py data/videos/camera1.mp4 data/videos/camera6.mp4
Step 2 - read the two times on the saved images and compute the frame offset:
    python tools/sync_clocks.py data/videos/camera1.mp4 data/videos/camera6.mp4 --times 04:01:46 04:00:52

Result: "camera6 frame = camera1 frame + N". Put N in CAMERA_GROUPS in tracking/config.py.
Only the seconds digits are watched, so fans or lights moving behind the clock are ignored.
Images are saved in data/output/sync/.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking.config import OUTPUT_DIR  # noqa: E402

SECONDS_BOX = (1715, 25, 1800, 80)  # x1, y1, x2, y2 of the seconds digits on a 1920x1080 frame
CLOCK_BOX = (1150, 15, 1900, 80)    # the whole date/time, for the saved images


def find_ticks(path, box=SECONDS_BOX):
    """Frames where the seconds digits change, the tick phase (frame mod fps) and the fps."""
    cap = cv2.VideoCapture(str(path))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    x1, y1, x2, y2 = box
    prev, diffs = None, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[1] != 1920:
            frame = cv2.resize(frame, (1920, 1080))
        digits = (cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) > 200).astype(np.int16)  # white digits only
        diffs.append(0 if prev is None else int(np.abs(digits - prev).sum()))
        prev = digits
    cap.release()
    d = np.array(diffs)
    thr = max(30, 0.3 * np.percentile(d, 99.5))
    ticks = [k for k in range(1, len(d)) if d[k] > thr and d[k] >= d[max(0, k - 2):k + 3].max()]
    counts = np.bincount(np.array(ticks) % fps, minlength=fps)
    # ticks may land on phase p or p+1 (frame timing jitter): pick the phase with the most ticks in p..p+1
    phase = int(np.argmax([counts[p] + counts[(p + 1) % fps] for p in range(fps)]))
    real = [t for t in ticks if (t - phase) % fps in (0, 1)]
    return real, phase, fps, len(d)


def clock_crop(path, frame_no):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, frame = cap.read()
    cap.release()
    x1, y1, x2, y2 = CLOCK_BOX
    return cv2.resize(frame, (1920, 1080))[y1:y2, x1:x2].copy(), cv2.resize(frame, (960, 540))


def seconds(hms):
    h, m, s = (int(v) for v in hms.split(":"))
    return h * 3600 + m * 60 + s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video_a")
    ap.add_argument("video_b")
    ap.add_argument("--times", nargs=2, metavar=("TIME_A", "TIME_B"),
                    help="HH:MM:SS shown at the first tick of each video (read from the saved images)")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "sync"), help="folder for the clock and check images")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    info = {}
    for key, path in (("a", args.video_a), ("b", args.video_b)):
        ticks, phase, fps, n = find_ticks(path)
        gaps = np.diff(ticks)
        regular = np.mean((gaps >= fps - 1) & (gaps <= fps + 1)) if len(gaps) else 0
        name = Path(path).stem
        print("{}: {} frames @ {} fps, {} ticks, phase {}, {:.0%} of ticks {}+/-1 frames apart, first tick frame {}"
              .format(name, n, fps, len(ticks), phase, regular, fps, ticks[0]))
        if regular < 0.9:
            print("  WARNING: clock ticks are irregular - check {} (dropped frames or wrong clock box)".format(name))
        crop, _ = clock_crop(path, ticks[0])
        cv2.imwrite(str(out / "{}_first_tick.png".format(name)), crop)
        info[key] = (name, path, ticks[0], fps)

    (na, pa, ta, fps), (nb, pb, tb, _) = info["a"], info["b"]
    if not args.times:
        print("Now read the time on {0}/{1}_first_tick.png and {0}/{2}_first_tick.png and run again with "
              "--times <time {1}> <time {2}>".format(out, na, nb))
        return
    offset = tb - ta + (seconds(args.times[0]) - seconds(args.times[1])) * fps
    print("RESULT: {} frame = {} frame {:+d}  ({:+.2f} s)".format(nb, na, offset, offset / fps))

    # check image: the same moment from both videos, at 3 points in time
    rows = []
    for frac in (0.2, 0.5, 0.8):
        fa = int(frac * 1000) if offset >= 0 else int(frac * 1000) - offset
        fa = max(0, fa)
        _, a = clock_crop(pa, fa)
        _, b = clock_crop(pb, fa + offset)
        cv2.putText(a, "{} f{}".format(na, fa), (10, 530), 0, 0.8, (0, 255, 255), 2)
        cv2.putText(b, "{} f{}".format(nb, fa + offset), (10, 530), 0, 0.8, (0, 255, 255), 2)
        rows.append(np.hstack([a, b]))
    cv2.imwrite(str(out / "{}_{}_check.jpg".format(na, nb)), np.vstack(rows))
    print("Check image: {} (both clocks must show the same time)".format(out / "{}_{}_check.jpg".format(na, nb)))


if __name__ == "__main__":
    main()
