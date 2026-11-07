"""BoT-SORT tracker and the OSNet re-ID model (both from boxmot)."""
from pathlib import Path

from .config import HOLD_S, REID_WEIGHTS


def make_tracker(fps, device):
    """BoT-SORT without appearance model (fewest ID changes in our tests). track_buffer is given for
    30 fps and scaled by boxmot, so this keeps lost tracks for HOLD_S seconds at any frame rate."""
    from boxmot import BotSort
    return BotSort(reid_weights=Path("unused.pt"), device=device, half=False, with_reid=False,
                   track_high_thresh=0.25, new_track_thresh=0.25, track_low_thresh=0.1,
                   track_buffer=int(HOLD_S * 30), match_thresh=0.8, frame_rate=round(fps),
                   cmc_method="sof")  # fixed cameras: camera-motion correction finds nothing, cheapest method


def load_reid(device):
    """OSNet re-ID model used by ReLink (only when the tracker gives a new ID). Weights are downloaded on first
    use into boxmot's weights folder; boxmot wants "0" for the first CUDA GPU."""
    from boxmot.reid.core.auto_backend import ReidAutoBackend
    from boxmot.utils import WEIGHTS
    return ReidAutoBackend(weights=WEIGHTS / REID_WEIGHTS,
                           device="0" if device == "cuda" else device, half=device == "cuda").model
