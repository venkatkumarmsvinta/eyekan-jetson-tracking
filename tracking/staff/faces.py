"""Face embeddings of the staff photos with InsightFace buffalo_l (SCRFD face detector + ArcFace R50).
The buffalo_l weights are licensed for non-commercial research only: get a licence before production use.
Model files are downloaded to ~/.insightface/models on first use."""
import json
from pathlib import Path

import cv2

from ..config import FACE_DET_SIZE, FACE_MODEL, STAFF_EMBED_FILE, STAFF_PHOTO_DIR


class FaceEmbedder:
    """BGR image -> 512-number face embedding (length 1) of the most confident face."""

    def __init__(self):
        import onnxruntime
        from insightface.app import FaceAnalysis
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                     if p in onnxruntime.get_available_providers()]
        self.app = FaceAnalysis(name=FACE_MODEL, providers=providers, allowed_modules=["detection", "recognition"])
        self.app.prepare(ctx_id=0, det_size=(FACE_DET_SIZE, FACE_DET_SIZE))

    def __call__(self, image):
        """Returns (embedding, detector score), or (None, 0.0) when no face is found."""
        if image is None:
            return None, 0.0
        s = FACE_DET_SIZE / max(image.shape[:2])  # small head crops are scaled up, large photos down
        faces = self.app.get(cv2.resize(image, None, fx=s, fy=s))
        if not faces:
            return None, 0.0
        face = max(faces, key=lambda f: f.det_score)
        return face.normed_embedding, float(face.det_score)


def embed_photos():
    """Keep STAFF_EMBED_FILE in step with the photos in STAFF_PHOTO_DIR: embed new and changed photos,
    drop removed ones. Returns (added, updated, removed, no_face)."""
    root = Path(STAFF_PHOTO_DIR)
    try:
        data = json.loads(Path(STAFF_EMBED_FILE).read_text())
    except (OSError, ValueError):
        data = {}
    if data.get("model") != FACE_MODEL:  # other model: its embeddings can't be compared, start again
        data = {"model": FACE_MODEL, "photos": {}}
    known = data["photos"]
    photos = {p.relative_to(root).as_posix(): p for p in sorted(root.glob("*/*.jpg"))}
    added = updated = no_face = 0
    embedder = None
    for key, path in photos.items():
        stamp = "{}-{}".format(path.stat().st_mtime_ns, path.stat().st_size)
        if key in known and known[key]["stamp"] == stamp:
            continue
        if embedder is None:
            embedder = FaceEmbedder()  # load the model only when there is work to do
        emb, score = embedder(cv2.imread(str(path)))
        if key in known:
            updated += 1
        else:
            added += 1
        no_face += emb is None
        known[key] = {"stamp": stamp, "score": round(score, 3),
                      "embedding": None if emb is None else [round(float(v), 6) for v in emb]}
    removed = [k for k in known if k not in photos]
    for k in removed:
        del known[k]
    if root.exists():
        Path(STAFF_EMBED_FILE).write_text(json.dumps(data))
    return added, updated, len(removed), no_face
