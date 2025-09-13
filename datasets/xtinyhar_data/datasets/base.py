
from typing import Optional, Dict, Any, List
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

def _load_array(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    elif ext == ".csv":
        # Try to read CSV (numeric). If headers exist, select numeric columns only.
        try:
            return pd.read_csv(path, header=None).values
        except Exception:
            return pd.read_csv(path).select_dtypes(include=[np.number]).values
    else:
        raise ValueError(f"Unsupported file extension: {ext} ({path})")

class BaseRawSequenceDataset(Dataset):
    """
    Minimal base dataset that:
      - reads a manifest.csv
      - loads raw IMU and (optionally) skeleton arrays
      - returns sample dict with raw tensors and metadata
    Absolutely NO preprocessing here.
    """

    def __init__(
        self,
        manifest_path: str,
        split: Optional[str] = None,  # 'train'|'val'|'test' or None
    ):
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"manifest not found: {manifest_path}")
        self.df = pd.read_csv(manifest_path)
        required = ["seq_id", "subject", "label", "imu_path"]
        for col in required:
            if col not in self.df.columns:
                raise ValueError(f"manifest must contain column: {col}")
        # Optional columns: skeleton_path, split
        if split is not None:
            if "split" not in self.df.columns:
                raise ValueError("split requested but 'split' column is missing in manifest.")
            self.df = self.df[self.df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("No rows in manifest after split filtering (or empty manifest).")

    def __len__(self) -> int:
        return len(self.df)

    def _load_sample(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        imu = torch.from_numpy(_load_array(row["imu_path"]).astype(np.float32))  # (T, C_imu)
        skel = None
        skel_path = row.get("skeleton_path", None)
        if isinstance(skel_path, str) and len(skel_path) > 0 and os.path.exists(skel_path):
            skel = torch.from_numpy(_load_array(skel_path).astype(np.float32))  # (T, C_skel)
        meta = {
            "seq_id": str(row["seq_id"]),
            "subject": str(row["subject"]),
            "label": row["label"],
            "imu_path": row["imu_path"],
            "skeleton_path": skel_path if isinstance(skel_path, str) and len(skel_path) > 0 else None,
            "split": row.get("split", None),
        }
        return {"imu": imu, "skeleton": skel, "meta": meta}

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._load_sample(idx)

def collate_list(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    No-op collate to preserve variable-length sequences with NO padding/processing.
    Use batch_size=1 if you need strict raw handling, or handle padding in your model.
    """
    return batch
