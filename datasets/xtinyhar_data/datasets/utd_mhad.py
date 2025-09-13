
from .base import BaseRawSequenceDataset

class UTD_MHAD_Raw(BaseRawSequenceDataset):
    """
    UTD-MHAD raw dataset (no preprocessing).
    Expects a manifest.csv with columns:
      seq_id, subject, label, imu_path[, skeleton_path][, split]
    """
    pass
