"""ReLink: keeps a person's ID when the tracker gives them a new one (re-link, late re-link, hand-over).
apply_handovers() fixes the CSV and floor map afterwards for IDs that were renamed."""
import csv

import cv2
import numpy as np

from .. import config
from ..config import CROP_H, CROP_W


class ReLink:
    """Keeps a person's ID when the tracker gives them a new one.

    Appearance: while a person is tracked, a small crop of them is kept every GALLERY_EVERY_S seconds (only when
       their box overlaps nobody and the torso is visible), at most GALLERY crops (front, side, back ...), with
       its torso colour. The re-ID model (OSNet) does NOT run during normal tracking. It runs only when the
       tracker gives a NEW ID whose box overlaps nobody and someone lost could match: on the new person's crop,
       and once on each lost candidate's stored crops (cached). A new box that overlaps someone (its crop shows
       two people) uses torso colour + position as before (colour distance <= COLOUR_MAX).
       sim = best cosine similarity between the two sets of crops. Colour only vetoes: torso colours further
       apart than COLOUR_VETO are never the same person.
    Re-link: a new tracker ID gets the ID of a person lost in the last REMEMBER_S seconds, when
       - sim >= SAME, wherever they are (walked out of the frame and came back elsewhere), or
       - sim >= LIKELY and the body point is close to where the lost person was last seen (allowed distance
         grows with the time they were gone: DIST_START box heights + DIST_PER_S per second, at most DIST_MAX;
         not checked at all after ANYWHERE_S seconds).
       If the torso is mostly hidden (e.g. bent behind a rack, only the head visible), there is no appearance:
       only people lost in the last BLIND_S seconds qualify, by position alone, the most recently lost first.
       Tried again during the new ID's first RETRY_S seconds, because a person walking back into the frame is
       only partly visible at first.
    Hand-over: when an ID ends and a young ID (born after it, < YOUNG_S old) is within HANDOVER_DIST box
       heights of where it ended with a similar torso colour (<= COLOUR_MAX, or unknown), the young ID becomes
       the old ID. This is the overlap case: the tracker started the new ID while the old one was still alive.
       No re-ID model here: during an overlap a crop can show two people and swap the IDs."""

    # thresholds: see tracking/config.py
    SAME = config.SAME
    LIKELY = config.LIKELY
    COLOUR_VETO = config.COLOUR_VETO
    COLOUR_MAX = config.COLOUR_MAX
    GALLERY = config.GALLERY
    GALLERY_EVERY_S = config.GALLERY_EVERY_S
    DIST_START = config.DIST_START
    DIST_PER_S = config.DIST_PER_S
    DIST_MAX = config.DIST_MAX
    HANDOVER_DIST = config.HANDOVER_DIST
    RETRY_S = config.RETRY_S
    ANYWHERE_S = config.ANYWHERE_S
    BLIND_S = config.BLIND_S
    BLIND_DIST = config.BLIND_DIST
    BLIND_GAP_COST = config.BLIND_GAP_COST

    def __init__(self, fps, reid):
        self.fps, self.hold, self.young = fps, int(config.REMEMBER_S * fps), int(config.YOUNG_S * fps)
        self.retry = int(self.RETRY_S * fps)
        self.every = max(1, int(self.GALLERY_EVERY_S * fps))
        self.reid = reid
        self.final = {}      # tracker ID -> ID shown
        self.last = {}       # ID shown -> {"frame", "point", "height", "views": [(crop, colour)], "added"}
        self.vecs = {}       # ID shown -> ("added" frame, appearance vectors of its views): computed only when needed
        self.born = {}       # brand-new ID shown -> frame it first appeared
        self.handovers = []  # (young ID, old ID, frame the young ID was born), in order
        self.relinked = self.handed_over = 0
        self.n_embedded = 0  # crops the re-ID model has processed (its real cost)

    def near(self, dist, gap_s):
        """Close enough to where the lost person was last seen, for the time they were gone."""
        return gap_s >= self.ANYWHERE_S or dist <= min(self.DIST_MAX, self.DIST_START + self.DIST_PER_S * gap_s)

    def embed(self, crops):
        """THE ONLY PLACE THE RE-ID MODEL RUNS: appearance vectors of the crops, in one batch."""
        if not crops:
            return []
        strip = np.hstack(crops)  # crops side by side, one box each
        boxes = np.array([[i * CROP_W, 0, (i + 1) * CROP_W, CROP_H] for i in range(len(crops))], np.float32)
        self.n_embedded += len(crops)
        return list(self.reid.get_features(boxes, strip))

    def gallery(self, fid):
        """(appearance vectors, colours) of a person's stored views. The vectors are computed on first need and
        cached; a lost person's views don't change, so that happens once per lost person."""
        s = self.last.get(fid)
        if not s or not s["views"]:
            return [], []
        cached = self.vecs.get(fid)
        if cached is None or cached[0] != s["added"]:
            cached = self.vecs[fid] = (s["added"], self.embed([crop for crop, _ in s["views"]]))
        return cached[1], [colour for _, colour in s["views"]]

    def colours(self, fid):
        """Torso colours of a person's stored views (no model)."""
        s = self.last.get(fid)
        return [c for _, c in s["views"]] if s else []

    @staticmethod
    def colour_dist(a, b):
        """Distance between the AVERAGE torso colour of two lists of colours (like the old averaged colour;
        the closest pair is too lenient: some pair is always similar by chance). None if unknown."""
        a, b = [x for x in a if x is not None], [y for y in b if y is not None]
        if not a or not b:
            return None
        return cv2.compareHist(np.mean(a, axis=0).astype(np.float32), np.mean(b, axis=0).astype(np.float32),
                               cv2.HISTCMP_BHATTACHARYYA)

    def current(self, k, people, cache):
        """(appearance vectors, colours) of person k's view in this frame; embedded at most once per frame.
        Box overlapping someone: the crop shows two people, so NO re-ID, colour only."""
        if k not in cache:
            view, clean = people[k][3], people[k][4]
            if view is None:
                cache[k] = ([], [])
            elif not clean:
                cache[k] = ([], [view[1]])
            else:
                cache[k] = (self.embed([view[0]]), [view[1]])
        return cache[k]

    def young_views(self, young, k, people, cache):
        """A young ID's stored views, or its current view if none is stored yet."""
        g = self.gallery(young)
        return g if g[0] else self.current(k, people, cache)

    @staticmethod
    def match(a, b):
        """(best appearance similarity, lowest torso-colour distance) between two (vectors, colours); None = unknown."""
        sim = float(np.max(np.stack(a[0]) @ np.stack(b[0]).T)) if len(a[0]) and len(b[0]) else None
        ds = [cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA)
              for x in a[1] if x is not None for y in b[1] if y is not None]
        return sim, (min(ds) if ds else None)

    def same(self, sim, cdist, dist, gap_s):
        """Same person? Colour can only say no; appearance (+ position when less sure) says yes."""
        if cdist is not None and cdist > self.COLOUR_VETO:
            return False
        return sim >= self.SAME or (sim >= self.LIKELY and self.near(dist, gap_s))

    def __call__(self, frame_idx, people):
        """people: [(tracker_id, body_point, box_height, view or None, clean), ...] -> [ID shown, ...]
        view = (crop for the re-ID model, torso colour or None); None when the torso is mostly hidden."""
        ids = [self.final.get(p[0]) for p in people]
        in_use = {i for i in ids if i is not None}
        lost = {fid: s for fid, s in self.last.items()
                if fid not in in_use and 0 < frame_idx - s["frame"] <= self.hold}
        cache = {}  # k -> this frame's appearance of person k (computed only if someone lost could match)

        # re-link: score every (new tracker ID, lost person) pair that passes the checks, best pairs first
        pairs = []
        for k, (tid, point, height, view, _) in enumerate(people):
            if ids[k] is not None:
                continue
            for fid, s in lost.items():
                gap_s = (frame_idx - s["frame"]) / self.fps
                dist = np.hypot(point[0] - s["point"][0], point[1] - s["point"][1]) / max(height, s["height"], 1)
                mine = self.current(k, people, cache)
                if mine[0] and s["views"]:  # new ID, clean box: appearance (re-ID model)
                    sim, cdist = self.match(mine, self.gallery(fid))
                    if self.same(sim, cdist, dist, gap_s):
                        pairs.append((dist + 2 * (1 - sim), k, fid))
                    continue
                cdist = self.colour_dist(mine[1], self.colours(fid))
                if cdist is None:
                    # partial view (e.g. bent behind a rack, only the head visible): no appearance, so
                    # position alone decides. Only people lost a moment ago, the most recently lost first, so
                    # the box does not take the ID of someone else who was hidden at the same spot earlier.
                    if gap_s <= self.BLIND_S and dist <= self.BLIND_DIST:
                        pairs.append((dist + self.BLIND_GAP_COST * gap_s, k, fid))
                    continue
                if self.near(dist, gap_s) and cdist <= self.COLOUR_MAX:  # overlapping box: colour, as before
                    pairs.append((dist + cdist, k, fid))
        for _, k, fid in sorted(pairs):
            if ids[k] is None and fid in lost:
                ids[k] = fid
                self.final[people[k][0]] = fid
                del lost[fid]
                self.relinked += 1
        # brand-new people keep their tracker ID (always higher than any earlier one, so never a clash)
        for k, (tid, *_rest) in enumerate(people):
            if ids[k] is None:
                ids[k] = self.final[tid] = tid
                self.born[tid] = frame_idx

        # late re-link: a person coming back into the frame is only partly visible in their first frames, so
        # keep trying for RETRY_S seconds once their appearance is known. Renamed like a hand-over (CSV fixed later).
        in_use = set(ids)
        pairs = []
        for k, (tid, point, height, view, _) in enumerate(people):
            young = ids[k]
            born = self.born.get(young)
            if born is None or born == frame_idx or frame_idx - born > self.retry:
                continue
            candidates = [fid for fid, s in self.last.items()
                          if fid not in in_use and s["views"] and 0 < born - s["frame"] <= self.hold]
            if not candidates:  # nobody lost: the re-ID model does not run
                continue
            mine = self.young_views(young, k, people, cache)
            if not mine[0]:  # no appearance yet: too risky to match on position alone
                continue
            for fid in candidates:
                s = self.last[fid]
                gap_s = (frame_idx - s["frame"]) / self.fps
                dist = np.hypot(point[0] - s["point"][0], point[1] - s["point"][1]) / max(height, s["height"], 1)
                sim, cdist = self.match(mine, self.gallery(fid))
                if self.same(sim, cdist, dist, gap_s):
                    pairs.append((dist + 2 * (1 - sim), k, fid))
        done = set()
        for _, k, old in sorted(pairs):
            young = ids[k]
            if k in done or old in done:
                continue
            for t, f in self.final.items():
                if f == young:
                    self.final[t] = old
            self.handovers.append((young, old, self.born.pop(young)))
            self.last.pop(young, None)
            self.vecs.pop(young, None)
            ids[k] = old
            done.update((k, old))
            self.relinked += 1

        # hand-over: IDs that were there in the previous frame and are gone now
        for old in [fid for fid, s in self.last.items() if s["frame"] == frame_idx - 1 and fid not in ids]:
            s = self.last[old]
            best = None
            for k, (tid, point, height, view, _) in enumerate(people):
                young = ids[k]
                born = self.born.get(young)
                if born is None or frame_idx - born > self.young or born <= self.born.get(old, -1):
                    continue
                dist = np.hypot(point[0] - s["point"][0], point[1] - s["point"][1]) / max(height, s["height"], 1)
                # overlap case: colour + position only, no re-ID model (a crop here may show two people)
                cdist = self.colour_dist(self.colours(young) or [view[1] if view else None], self.colours(old))
                cdist = 0.0 if cdist is None else cdist  # colour unknown: position decides (as before)
                if dist <= self.HANDOVER_DIST and cdist <= self.COLOUR_MAX and (best is None or dist + cdist < best[0]):
                    best = (dist + cdist, k)
            if best is not None:
                young = ids[best[1]]
                for t, f in self.final.items():
                    if f == young:
                        self.final[t] = old
                self.handovers.append((young, old, self.born.pop(young)))
                self.last.pop(young, None)
                self.vecs.pop(young, None)
                ids[best[1]] = old
                self.handed_over += 1

        # keep a crop now and then (no model runs here), only from boxes that overlap nobody
        for k, (tid, point, height, view, clean) in enumerate(people):
            s = self.last.get(ids[k], {"views": [], "added": -self.every})
            views, added = s["views"], s["added"]
            if clean and view is not None and frame_idx - added >= self.every:
                views, added = (views + [view])[-self.GALLERY:], frame_idx
            self.last[ids[k]] = {"frame": frame_idx, "point": point, "height": height, "views": views, "added": added}
        # forget people gone too long to be re-linked (their crops would only use memory)
        for fid in [f for f, s in self.last.items() if frame_idx - s["frame"] > self.hold + self.retry]:
            del self.last[fid]
            self.vecs.pop(fid, None)
        return ids


def apply_handovers(csv_path, floor_map, handovers):
    """Rename every young ID to the old ID it was handed over to, also in the rows written before the
    hand-over. From the young ID's birth on, the young rows replace the old ID's rows (in the overlap the old
    box had already slid off the person)."""
    if not handovers:
        return
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, rows = rows[0], rows[1:]
    ti, fi = header.index("track_id"), header.index("frame")
    for young, old, born in handovers:
        rows = [r for r in rows if not (int(r[ti]) == old and int(r[fi]) >= born)]
        for r in rows:
            if int(r[ti]) == young:
                r[ti] = str(old)
        floor_map.paths[old].extend(floor_map.paths.pop(young, []))
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
