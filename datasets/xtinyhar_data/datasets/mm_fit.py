
from .base import BaseRawSequenceDataset

class MM_FIT_Raw(BaseRawSequenceDataset):
    """
    MM-Fit raw dataset (no preprocessing).
    Expects a manifest.csv with columns:
      seq_id, subject, label, imu_path[, skeleton_path][, split]
    """
    pass
