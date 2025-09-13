from dataclasses import asdict
from typing import Optional, Dict, Any, List, Tuple
import numpy as np

from .config import PreprocessConfig
from .sliding_window import windows_indices
from .normalization import zscore_per_channel, recenter_and_scale_skeleton
from .alignment import resample_to_grid, make_time_grid
from .dynamic_patching import motion_variance, choose_patch_size, to_patches, to_patches_skeleton
from .augmentation import butter_lowpass_filter, add_gaussian_noise, moving_average_skeleton
from .utils import ensure_dir, save_npz, append_jsonl

class Preprocessor:
    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self._rng = np.random.default_rng(cfg.seed)

    def fit_transform(
        self,
        t_imu: np.ndarray,
        imu: np.ndarray,           # [T, C]
        labels: np.ndarray,        # [T] or [num_windows]
        t_skel: Optional[np.ndarray] = None,
        skel: Optional[np.ndarray] = None,  # [T, J, D]
        augment: bool = False,
        out_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Returns dict with 'imu_patches', optional 'skel_patches', 'y', and 'meta'.
        If out_dir provided, writes per-window .npz and meta.jsonl.
        """
        cfg = self.cfg
        if out_dir:
            ensure_dir(out_dir)
            open(f"{out_dir}/meta.jsonl", "w").close()

        # Align to common time grid
        t0, t1 = float(t_imu[0]), float(t_imu[-1])
        grid = make_time_grid(t0, t1, cfg.target_hz)
        imu_res = resample_to_grid(t_imu, imu, grid).astype(np.float32)

        skel_res = None
        if skel is not None and t_skel is not None:
            skel_res = resample_to_grid(t_skel, skel, grid).astype(np.float32)

        # Normalize + optional filters/augmentation
        imu_norm, imu_stats = zscore_per_channel(imu_res)
        if cfg.apply_lowpass:
            imu_norm = butter_lowpass_filter(imu_norm, fs=cfg.target_hz, cutoff=cfg.accel_lowpass_hz)
        if augment and cfg.add_gaussian_noise:
            imu_norm = add_gaussian_noise(imu_norm, rel_std=cfg.noise_std, rng=self._rng)

        if skel_res is not None:
            skel_norm, skel_stats = recenter_and_scale_skeleton(skel_res, ref_joint=0)
            skel_norm = moving_average_skeleton(skel_norm, k=cfg.skel_ma_window)
        else:
            skel_norm, skel_stats = None, None

        # Sliding windows
        W = int(round(cfg.window_seconds * cfg.target_hz))
        stride = max(1, int(round(W * (1.0 - cfg.overlap))))
        imu_patches_list: List[np.ndarray] = []
        skel_patches_list: List[np.ndarray] = []
        y_list: List[int] = []
        meta_list: List[Dict[str, Any]] = []

        labels_is_per_timestep = (len(labels) == imu_norm.shape[0])

        for wi, (a, b) in enumerate(windows_indices(imu_norm.shape[0], W, stride)):
            imu_w = imu_norm[a:b]  # [W, C]
            if imu_w.shape[0] < W:
                continue

            # Dynamic patching: variance -> choose P
            vw = motion_variance(imu_w)
            P = choose_patch_size(vw, cfg.p_min, cfg.p_max, cfg.var_low, cfg.var_high)

            imu_patch = to_patches(imu_w, P)  # [N, P*C]
            if imu_patch.shape[0] == 0:
                continue

            if skel_norm is not None:
                skel_w = skel_norm[a:b]  # [W, J, D]
                skel_patch = to_patches_skeleton(skel_w, P)  # [N, P*J*D]
            else:
                skel_patch = None

            # Window label
            if labels_is_per_timestep:
                y = int(np.bincount(labels[a:b]).argmax())
            else:
                if wi >= len(labels):
                    break
                y = int(labels[wi])

            imu_patches_list.append(imu_patch)
            if skel_patch is not None:
                skel_patches_list.append(skel_patch)
            y_list.append(y)

            meta = {
                "window_index": wi,
                "start_idx": a, "end_idx": b,
                "start_time": float(a / cfg.target_hz),
                "end_time": float(b / cfg.target_hz),
                "variance_Vw": vw,
                "chosen_P": P,
                "imu_stats_mean": imu_stats["mean"].tolist(),
                "imu_stats_std": imu_stats["std"].tolist()
            }
            if skel_stats is not None:
                meta.update(skel_stats)
            meta_list.append(meta)

            if out_dir:
                path_npz = f"{out_dir}/win_{wi:06d}.npz"
                if skel_patch is not None:
                    save_npz(path_npz, imu=imu_patch, skel=skel_patch, y=y)
                else:
                    save_npz(path_npz, imu=imu_patch, y=y)
                append_jsonl(f"{out_dir}/meta.jsonl", meta)

        out: Dict[str, Any] = {
            "imu_patches": imu_patches_list,
            "y": np.array(y_list, dtype=np.int64),
            "meta": meta_list
        }
        if skel_patches_list:
            out["skel_patches"] = skel_patches_list
        return out
