
# XTinyHAR – Raw Datasets

This repository provides **preprocessing-free** PyTorch datasets for **UTD-MHAD** and **MM-Fit**.
It only loads raw files and returns tensors plus metadata—**no filtering, normalization, resampling, or windowing**.

## Manifest format

Provide a `manifest.csv` per dataset with the following columns (UTF-8, header required):

- `seq_id` (str): unique sequence id
- `subject` (str/int): subject id
- `label` (str/int): activity label (string or integer)
- `imu_path` (str): path to a raw IMU file (`.npy` or `.csv`)
- `skeleton_path` (str, optional): path to skeleton file (`.npy` or `.csv`)
- `split` (str, optional): 'train' | 'val' | 'test'

**Raw file expectations**
- IMU file shape: `(T, C_imu)` (e.g., columns: ax, ay, az, gx, gy, gz, ...)
- Skeleton file shape: `(T, C_skel)` (flat per frame or any numeric columns)
- If CSV has headers, non-numeric columns are ignored automatically.

### Example `manifest.csv`

```
seq_id,subject,label,imu_path,skeleton_path,split
S01_A01_T1,1,clap,imu/S01_A01_T1.npy,skel/S01_A01_T1.npy,train
S01_A02_T1,1,walk,imu/S01_A02_T1.npy,skel/S01_A02_T1.npy,val
S02_A01_T1,2,clap,imu/S02_A01_T1.npy,skel/S02_A01_T1.npy,test
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
python examples/inspect_first_batch.py   --utd_manifest /path/to/UTD-MHAD/manifest.csv   --mmfit_manifest /path/to/MM-Fit/manifest.csv   --split train
```

## Python usage

```python
from torch.utils.data import DataLoader
from xtinyhar_data.datasets import UTD_MHAD_Raw, MM_FIT_Raw
from xtinyhar_data.datasets.base import collate_list

utd = UTD_MHAD_Raw("/path/to/UTD-MHAD/manifest.csv", split="train")
loader = DataLoader(utd, batch_size=1, collate_fn=collate_list)

sample = next(iter(loader))[0]
print(sample["imu"].shape, sample["skeleton"], sample["meta"])
```

## Notes
- This package intentionally does **not** include any preprocessing to match the manuscript's *Data Description* section.
- You can layer your own preprocessing/segmentation in the training pipeline.
