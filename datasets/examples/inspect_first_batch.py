
import argparse
from torch.utils.data import DataLoader
from xtinyhar_data.datasets import UTD_MHAD_Raw, MM_FIT_Raw
from xtinyhar_data.datasets.base import collate_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utd_manifest", type=str, default=None, help="Path to UTD-MHAD manifest.csv")
    ap.add_argument("--mmfit_manifest", type=str, default=None, help="Path to MM-Fit manifest.csv")
    ap.add_argument("--split", type=str, default=None, help="Optional: 'train'|'val'|'test'")
    args = ap.parse_args()

    if args.utd_manifest:
        utd = UTD_MHAD_Raw(args.utd_manifest, split=args.split)
        utd_loader = DataLoader(utd, batch_size=1, shuffle=False, collate_fn=collate_list)
        sample = next(iter(utd_loader))[0]
        print("[UTD-MHAD] seq_id:", sample["meta"]["seq_id"])
        print("  imu shape:", tuple(sample["imu"].shape))
        print("  skeleton:", "present" if sample["skeleton"] is not None else "None")
        print("  label:", sample["meta"]["label"])

    if args.mmfit_manifest:
        mmfit = MM_FIT_Raw(args.mmfit_manifest, split=args.split)
        mmfit_loader = DataLoader(mmfit, batch_size=1, shuffle=False, collate_fn=collate_list)
        sample = next(iter(mmfit_loader))[0]
        print("[MM-Fit] seq_id:", sample["meta"]["seq_id"])
        print("  imu shape:", tuple(sample["imu"].shape))
        print("  skeleton:", "present" if sample["skeleton"] is not None else "None")
        print("  label:", sample["meta"]["label"])

if __name__ == "__main__":
    main()
