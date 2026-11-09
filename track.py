"""Track the people in a store CCTV video.

    python track.py videos/camera1.mp4 --show

Results: output/<video name>/rfdetr.mp4, rfdetr.csv, floor_map.png. Settings: tracking/config.py.
"""
import argparse

from tracking.detection import MODELS
from tracking.db import sync
from tracking.pipeline import run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", help="path to input video")
    ap.add_argument("--size", default="nano", choices=list(MODELS), help="RF-DETR model size")
    ap.add_argument("--conf", type=float, default=0.25, help="minimum detection confidence")
    ap.add_argument("--show", action="store_true", help="show the video and the floor map live (q to quit)")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = whole video)")
    ap.add_argument("--no-sync", action="store_true", help="don't refresh the staff photos from Supabase at start")
    args = ap.parse_args()
    if not args.no_sync:
        sync()
    run(args.video, size=args.size, conf=args.conf, show=args.show, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
