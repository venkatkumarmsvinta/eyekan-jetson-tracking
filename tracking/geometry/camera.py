"""A calibrated camera: store position in metres (x, y on the floor, z up) <-> image pixels.

Made by tools/calibrate_3d.py and saved as data/calibration/<camera>.json. Used by the multi-camera tracking to put
every person on one store floor map: from the head pixel at the person's head height, or from the feet at z = 0.
"""
import json
from pathlib import Path

import cv2
import numpy as np


class Camera:
    def __init__(self, K, dist, rvec, tvec, image_size, name=""):
        self.name = name
        self.K = np.asarray(K, np.float64).reshape(3, 3)
        self.dist = np.asarray(dist, np.float64).ravel()
        self.rvec = np.asarray(rvec, np.float64).reshape(3, 1)
        self.tvec = np.asarray(tvec, np.float64).reshape(3, 1)
        self.R = cv2.Rodrigues(self.rvec)[0]
        self.image_size = tuple(int(v) for v in image_size)  # (width, height)
        self.position = (-self.R.T @ self.tvec).ravel()      # camera centre in store metres (z = mounting height)

    # ---------- matrices ----------
    @property
    def P(self):
        """3x4 projection matrix, store metres -> undistorted pixels (MV3DT: projectionMatrix_3x4_w2p)."""
        return self.K @ np.hstack([self.R, self.tvec])

    def plane_homography(self, z=0.0):
        """3x3 homography undistorted pixels -> (x, y) on the horizontal plane at height z."""
        P = self.P
        H_world_to_img = np.column_stack([P[:, 0], P[:, 1], P[:, 2] * z + P[:, 3]])
        return np.linalg.inv(H_world_to_img)

    # ---------- points ----------
    def project(self, xyz):
        """Store points (N, 3) in metres -> pixels (N, 2) as seen in the (distorted) video."""
        pts, _ = cv2.projectPoints(np.asarray(xyz, np.float64).reshape(-1, 1, 3), self.rvec, self.tvec, self.K, self.dist)
        return pts.reshape(-1, 2)

    def undistort(self, uv):
        """Video pixels (N, 2) -> undistorted pixels (N, 2) (same image size, no lens distortion)."""
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        if not np.any(self.dist):
            return uv.reshape(-1, 2)
        return cv2.undistortPoints(uv, self.K, self.dist, P=self.K).reshape(-1, 2)

    def to_plane(self, uv, z=0.0):
        """Video pixels (N, 2) -> store (x, y) (N, 2) of the point at height z on that pixel's line of sight.
        z = 0 for feet, z = head height for a head."""
        p = self.undistort(uv)
        q = cv2.perspectiveTransform(p.reshape(-1, 1, 2), self.plane_homography(z))
        return q.reshape(-1, 2)

    # ---------- file ----------
    def to_dict(self, extra=None):
        d = {"camera": self.name, "image_size": list(self.image_size),
             "K": self.K.tolist(), "dist": self.dist.tolist(),
             "rvec": self.rvec.ravel().tolist(), "tvec": self.tvec.ravel().tolist(),
             "projectionMatrix_3x4_w2p": self.P.tolist(),
             "camera_position_m": self.position.round(3).tolist(),
             # floor homography for track.py (single camera, floor positions in metres); ignores lens distortion
             "H_image_to_floor": self.plane_homography(0.0).tolist()}
        d.update(extra or {})
        return d

    def save(self, path, extra=None):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(extra), indent=2))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["K"], d["dist"], d["rvec"], d["tvec"], d["image_size"], d.get("camera", Path(path).stem))
