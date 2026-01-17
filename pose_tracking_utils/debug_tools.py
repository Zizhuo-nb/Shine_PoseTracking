# pose_tracking_utils/debug_tools.py
from __future__ import annotations
from typing import Tuple, Optional
import os
import csv
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch


def transform_xyz(pts_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    """(N,3) -> (N,3) world, using 4x4 T."""
    pts_h = np.hstack([pts_xyz, np.ones((pts_xyz.shape[0], 1), dtype=np.float64)])
    return (T @ pts_h.T).T[:, :3]


def crop_tracking_points_world(
    pts_lidar_xyz: np.ndarray,
    T_wl: np.ndarray,
    max_dist: float,
    min_z: float,
) -> np.ndarray:
    if pts_lidar_xyz.shape[0] == 0:
        return pts_lidar_xyz

    d = np.linalg.norm(pts_lidar_xyz, axis=1)
    pts = pts_lidar_xyz[d < float(max_dist)]
    if pts.shape[0] == 0:
        return pts

    pts_w = transform_xyz(pts.astype(np.float64, copy=False), T_wl.astype(np.float64, copy=False))
    pts_w = pts_w[pts_w[:, 2] > float(min_z)]
    return pts_w


@torch.no_grad()
def probe_sdf_stats(
    points_w_np: np.ndarray,
    voxel_field,
    decoder,
    max_probe: int = 20000,
    s_band: float = 0.02,
) -> Dict[str, float]:
    out: Dict[str, float] = {"n": 0}
    if points_w_np is None or points_w_np.shape[0] == 0:
        return out

    n = points_w_np.shape[0]
    if n > max_probe:
        idx = np.random.choice(n, size=max_probe, replace=False)
        pts = points_w_np[idx]
    else:
        pts = points_w_np

    # infer device/dtype
    dev = voxel_field.feature_indexs_list[0].device
    dt = getattr(voxel_field, "dtype", None) or next(decoder.parameters()).dtype

    x = torch.from_numpy(pts.astype(np.float32, copy=False)).to(device=dev, dtype=dt)

    valid = voxel_field.get_valid_mask(x)
    valid_ratio = float(valid.float().mean().item())

    feats = voxel_field.get_features(x)
    sdf = decoder(feats).reshape(-1)

    abs_sdf = sdf.abs()
    med = float(abs_sdf.median().item())
    p95 = float(torch.quantile(abs_sdf, 0.95).item())
    in_band = float((abs_sdf < float(s_band)).float().mean().item())
    sign_bal = float((sdf > 0).float().mean().item())

    out = {
        "n": float(pts.shape[0]),
        "valid_ratio": valid_ratio,
        "med_abs_sdf": med,
        "p95_abs_sdf": p95,
        "in_band_ratio": in_band,
        "sign_balance": sign_bal,
    }
    return out


def pose_error_np(T_trk: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float, float, float, float]:
    t_est = T_trk[:3, 3]
    t_gt = T_gt[:3, 3]
    dt = t_est - t_gt
    t_err = float(np.linalg.norm(dt))

    R_est = T_trk[:3, :3]
    R_gt = T_gt[:3, :3]
    R_err = R_gt.T @ R_est
    c = np.clip((np.trace(R_err) - 1.0) / 2.0, -1.0, 1.0)
    r_err = float(np.arccos(c) * 180.0 / np.pi)
    return t_err, r_err, float(dt[0]), float(dt[1]), float(dt[2])


@dataclass
class DebugCSVLogger:
    path: str
    overwrite: bool = True 
    float_fmt: str = "{:.3f}"
    ts_int: bool = True
    _need_header: bool = True

    def __post_init__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._need_header = (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0)

    def _format_row(self, row: Dict) -> Dict:
        out = {}
        for k, v in row.items():
            if k == "ts" and self.ts_int:
                out[k] = int(float(v))
                continue
            if isinstance(v, float):
                if np.isnan(v):
                    out[k] = ""
                else:
                    out[k] = self.float_fmt.format(v)
            else:
                out[k] = v
        return out

    def append_row(self, row: Dict):
        row = self._format_row(row)
        fieldnames = list(row.keys())

        if self._need_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
            self._need_header = False

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writerow(row)



def make_debug_row(
    fid: int,
    pts_lidar_xyz: np.ndarray,
    T_gt: np.ndarray,
    T_trk: Optional[np.ndarray],
    voxel_field,
    decoder,
    max_dist: float,
    min_z: float,
    max_probe: int = 20000,
    band_ratio: float = 0.20,  
) -> Dict:
    # GT
    pts_gt_used = crop_tracking_points_world(pts_lidar_xyz, T_gt, max_dist=max_dist, min_z=min_z)
    st_gt0 = probe_sdf_stats(pts_gt_used, voxel_field, decoder, max_probe=max_probe, s_band=1e9)
    p95_gt = float(st_gt0.get("p95_abs_sdf", np.nan))
    s_band = band_ratio * p95_gt if np.isfinite(p95_gt) else np.nan

    st_gt = probe_sdf_stats(pts_gt_used, voxel_field, decoder, max_probe=max_probe, s_band=s_band)

    # TRK
    if T_trk is None:
        st_trk = {}
    else:
        pts_trk_used = crop_tracking_points_world(pts_lidar_xyz, T_trk, max_dist=max_dist, min_z=min_z)
        st_trk = probe_sdf_stats(pts_trk_used, voxel_field, decoder, max_probe=max_probe, s_band=s_band)

    return {
        "ts": time.time(),
        "fid": int(fid),

        "valid_gt": st_gt.get("valid_ratio", np.nan),
        "valid_trk": st_trk.get("valid_ratio", np.nan),

        "p95_gt": st_gt.get("p95_abs_sdf", np.nan),
        "p95_trk": st_trk.get("p95_abs_sdf", np.nan),

        "inband_gt": st_gt.get("in_band_ratio", np.nan),
        "inband_trk": st_trk.get("in_band_ratio", np.nan),

        "band": s_band,  # 让你一眼知道“本行 inband 的阈值是多少”
    }


##########################gradient##################################


@torch.no_grad()
def sdf_and_grad_fd(
    points_xyz: np.ndarray,
    voxel_field,
    decoder,
    fd_eps: float = 0.02,
    chunk_size: int = 200000,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finite-difference (central difference) gradient of f(x), where
        f(x) = decoder( voxel_field.get_features(x) )

    Args:
        points_xyz: (N,3) world coords as numpy array.
        voxel_field: has get_features(torch.Tensor)->torch.Tensor
        decoder: torch nn.Module, forward(features)->(N,) or (N,1)
        fd_eps: epsilon in meters for central difference
        chunk_size: process in chunks to avoid OOM
        device/dtype: if None, inferred from voxel_field / decoder

    Returns:
        sdf:  (N,) float32
        grad: (N,3) float32  (df/dx, df/dy, df/dz)
    """
    if points_xyz is None:
        raise ValueError("points_xyz is None")
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {points_xyz.shape}")

    n = int(points_xyz.shape[0])
    if n == 0:
        return np.zeros((0,), np.float32), np.zeros((0, 3), np.float32)

    # infer device/dtype
    if device is None:
        if hasattr(voxel_field, "feature_indexs_list") and len(voxel_field.feature_indexs_list) > 0:
            device = voxel_field.feature_indexs_list[0].device
        else:
            device = next(decoder.parameters()).device
    if dtype is None:
        dtype = getattr(voxel_field, "dtype", None) or next(decoder.parameters()).dtype

    eps = float(fd_eps)
    if eps <= 0:
        raise ValueError("fd_eps must be > 0")

    decoder.eval()

    sdf_out = np.empty((n,), dtype=np.float32)
    grad_out = np.empty((n, 3), dtype=np.float32)

    ex = torch.tensor([eps, 0.0, 0.0], device=device, dtype=dtype)
    ey = torch.tensor([0.0, eps, 0.0], device=device, dtype=dtype)
    ez = torch.tensor([0.0, 0.0, eps], device=device, dtype=dtype)

    def f(x: torch.Tensor) -> torch.Tensor:
        feats = voxel_field.get_features(x)
        y = decoder(feats).reshape(-1)  # (M,)
        return y

    for st in range(0, n, int(chunk_size)):
        ed = min(n, st + int(chunk_size))
        x_np = points_xyz[st:ed].astype(np.float32, copy=False)
        x0 = torch.from_numpy(x_np).to(device=device, dtype=dtype)

        # f(x0) returned too (useful for diagnostics)
        y0 = f(x0)

        yxp = f(x0 + ex); yxm = f(x0 - ex)
        yyp = f(x0 + ey); yym = f(x0 - ey)
        yzp = f(x0 + ez); yzm = f(x0 - ez)

        gx = (yxp - yxm) / (2.0 * eps)
        gy = (yyp - yym) / (2.0 * eps)
        gz = (yzp - yzm) / (2.0 * eps)

        sdf_out[st:ed] = y0.detach().cpu().numpy().astype(np.float32)
        grad_out[st:ed, 0] = gx.detach().cpu().numpy().astype(np.float32)
        grad_out[st:ed, 1] = gy.detach().cpu().numpy().astype(np.float32)
        grad_out[st:ed, 2] = gz.detach().cpu().numpy().astype(np.float32)

    return sdf_out, grad_out
