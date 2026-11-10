"""ViTPose (Hugging Face transformers): 17 COCO keypoints per person box."""
import numpy as np

from ..config import POSE_MODEL


class Pose:
    """ViTPose: keypoints for given person boxes (top-down: one crop per person)."""

    def __init__(self, device):
        print('self',self);
        print('device',device)
        import torch
        from transformers import AutoProcessor, VitPoseForPoseEstimation
        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(POSE_MODEL)
        self.model = VitPoseForPoseEstimation.from_pretrained(POSE_MODEL).to(device).eval()

    def __call__(self, rgb, xyxy):
        """RGB image, boxes (N, 4) x1 y1 x2 y2 -> keypoints xy (N, 17, 2), confidence (N, 17)."""
        if len(xyxy) == 0:
            return np.zeros((0, 17, 2), np.float32), np.zeros((0, 17), np.float32)
        boxes = np.asarray(xyxy, np.float32).copy()
        boxes[:, 2:] -= boxes[:, :2]  # ViTPose wants x, y, width, height
        inputs = self.processor(rgb, boxes=[boxes], return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_pose_estimation(out, boxes=[boxes])[0]
        return (np.stack([r["keypoints"].cpu().numpy() for r in res]),
                np.stack([r["scores"].cpu().numpy() for r in res]))
