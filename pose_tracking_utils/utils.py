from __future__ import annotations
from typing import Optional, Tuple
import torch
import numpy as np
import open3d as o3d
import config as cg
import math


def skew(v):
    S = torch.zeros(3,3, device= v.device, dtype=v.dtype)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 2] = -v[0]

    S[1, 0] = v[2]
    S[2, 0] = -v[1]
    S[2, 1] = v[0]
    return S


# def expmap(axis_angle:torch.Tensor):
#     angle = axis_angle.norm()
#     axis = axis_angle/angle
#     eye = torch.eye(3, device=axis_angle.device,dtype=torch.float64)
#     S = skew(axis)
#     R = eye + angle.sin()*S + (1-angle.cos())*(S@S)
#     return R
def expmap(axis_angle: torch.Tensor):
    theta = axis_angle.norm()
    I = torch.eye(3, device=axis_angle.device, dtype=torch.float64)
    W = skew(axis_angle)

    # 用 safe 的 A, B
    eps = 1e-8
    if theta < eps:
        # Taylor
        A = 1.0 - (theta**2) / 6.0
        B = 0.5 - (theta**2) / 24.0
    else:
        A = torch.sin(theta) / theta
        B = (1.0 - torch.cos(theta)) / (theta**2)

    return I + A * W + B * (W @ W)




def compute_gradient(sdf:torch.Tensor, points_world:torch.Tensor):
    assert sdf.shape[0] == points_world.shape[0]
    grads = torch.autograd.grad(
        outputs=sdf,
        inputs=points_world,
        grad_outputs=torch.ones_like(sdf),   # dy/dx，每个 SDF 的梯度
        create_graph=False,
        retain_graph=False,
        only_inputs=True
    )[0]
    return grads.detach()


########read point cloud and turn it into Tensor################
def read_point_cloud(path, device, dtype = torch.float64):
    pcd = o3d.io.read_point_cloud(path)
    pcd_numpy = np.asarray(pcd.points, dtype=np.float32)
    point_tensor = torch.from_numpy(pcd_numpy).to(device=device,dtype=dtype)
    return point_tensor
def parse_scan(file):
    pcd = o3d.io.read_point_cloud(file)
    pcd = pcd.voxel_down_sample(0.2)
    return pcd


###########initial guess to transfer first frame###########
def initial_guess(points_lidar: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3].to(dtype=points_lidar.dtype)
    t = T[:3, 3].to(dtype=points_lidar.dtype)
    points_world = points_lidar @ R.T + t
    return points_world



def forward_points_to_network(points, voxel_field, mlp):
    feature_vectors = voxel_field.get_features(points)
    pred = mlp(feature_vectors.float())
    return pred


def process_points(points_world:torch.Tensor, T:torch.tensor, radius, min_z):
    xyz = points_world[..., :3]
    # current_translation = T[:3,3].to(xyz.dtype)
    # points_distance = torch.norm(xyz - current_translation, p=2, dim=1, keepdim=False)
    # valid_mask = points_distance<radius
    # valid_mask = valid_mask & (xyz[:, 2] > min_z)
    valid_mask = xyz[:, 2] > min_z

    points_world = points_world[valid_mask]
    return points_world


def compute_pose_error(T_est: torch.Tensor, T_gt: torch.Tensor):
    t_est = T_est[:3, 3]
    t_gt  = T_gt[:3, 3]
    dt = t_est - t_gt  # dx,dy,dz
    t_err = torch.linalg.norm(dt).item()


    R_est = T_est[:3, :3]
    R_gt  = T_gt[:3, :3]
    R_err = R_gt.T @ R_est

   
    c = ((torch.trace(R_err) - 1.0) / 2.0).clamp(-1.0, 1.0)
    r_err = (torch.acos(c).item() * 180.0 / math.pi)

    return t_err, r_err, dt[0].item(), dt[1].item(), dt[2].item()
#################################################for visulize#################################
def sdf_colors_from_points(
    points_xyz: np.ndarray,
    voxel_field,
    decoder,
    s_max: float = 0.50,
    chunk_size: int = 200000,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    return_sdf: bool = False,
):
    """
    Compute SDF for point cloud and convert to RGB colors.

    Args:
        points_xyz: (N,3) numpy array (float32/float64)
        voxel_field: object with get_features(points: Tensor)->Tensor and optional get_valid_mask(points)->Tensor[bool]
        decoder: torch nn.Module, forward(feature_vectors)->(N,) or (N,1)
        s_max: color saturation range in meters. sdf in [-s_max, s_max] maps to blue-white-red.
        chunk_size: process points in chunks to avoid OOM
        device/dtype: if None, inferred from voxel_field (preferred) or decoder params.
        return_sdf: if True, also return sdf (N,) float32

    Returns:
        colors: (N,3) float64 in [0,1]
        (optional) sdf: (N,) float32
    """
    if points_xyz is None:
        raise ValueError("points_xyz is None")
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be (N,3), got {points_xyz.shape}")

    n = points_xyz.shape[0]
    if n == 0:
        colors = np.zeros((0, 3), dtype=np.float64)
        if return_sdf:
            return colors, np.zeros((0,), dtype=np.float32)
        return colors

    # infer device/dtype
    if device is None or dtype is None:
        if hasattr(voxel_field, "feature_indexs_list") and len(voxel_field.feature_indexs_list) > 0:
            dev0 = voxel_field.feature_indexs_list[0].device
        else:
            dev0 = next(decoder.parameters()).device
        dt0 = getattr(voxel_field, "dtype", None) or next(decoder.parameters()).dtype
        device = dev0 if device is None else device
        dtype = dt0 if dtype is None else dtype

    sdf_out = np.empty((n,), dtype=np.float32)

    decoder.eval()

    with torch.no_grad():
        for st in range(0, n, int(chunk_size)):
            ed = min(n, st + int(chunk_size))
            x_np = points_xyz[st:ed].astype(np.float32, copy=False)
            x = torch.from_numpy(x_np).to(device=device, dtype=dtype)

            feats = voxel_field.get_features(x)
            sdf = decoder(feats)

            # allow (M,) or (M,1)
            sdf = sdf.reshape(-1)
            sdf_out[st:ed] = sdf.detach().cpu().numpy().astype(np.float32)

    # signed sdf -> diverging colors: blue(neg) - white(0) - red(pos)
    s = np.clip(sdf_out / max(float(s_max), 1e-6), -1.0, 1.0)
    r = np.clip(s, 0.0, 1.0)
    b = np.clip(-s, 0.0, 1.0)
    g = 1.0 - 0.5 * (r + b)
    colors = np.stack([r, g, b], axis=-1).astype(np.float64)

    if return_sdf:
        return colors, sdf_out
    return colors



def run_full_debug_csv(tracker, gt_poses_np, begin_frame, end_frame, csv_path):
    import os
    import numpy as np
    import open3d as o3d
    from pose_tracking_utils.debug_tools import DebugCSVLogger, make_debug_row

    if os.path.exists(csv_path):
        os.remove(csv_path)

    dbg = DebugCSVLogger(csv_path, float_fmt="{:.3f}", ts_int=True)
    tracker.running_idx = begin_frame

    for fid in range(begin_frame, end_frame + 1):
        # raw points
        scan = o3d.io.read_point_cloud(tracker.file_list[fid])
        pts = np.asarray(scan.points, dtype=np.float64)

        T_gt = np.asarray(gt_poses_np[fid], dtype=np.float64)

        # tracking pose (sequential)
        if int(tracker.running_idx) != fid:
            raise RuntimeError(f"tracker.running_idx={tracker.running_idx} != fid={fid} (must be sequential)")
        T_trk_torch, _ = tracker.register_next()
        T_trk = T_trk_torch.detach().cpu().numpy().astype(np.float64)

        row = make_debug_row(
            fid=fid,
            pts_lidar_xyz=pts,
            T_gt=T_gt,
            T_trk=T_trk,
            voxel_field=tracker.feature,
            decoder=tracker.geo_decoder,
            max_dist=float(tracker.max_dist),
            min_z=float(tracker.min_z),
            max_probe=20000,
            band_ratio=0.30,   # 若你 make_debug_row 用的是 s_band，就改成 s_band=...
        )
        dbg.append_row(row)

    print(f"[DEBUG] saved: {csv_path}")


##########visual gradient array######
# pose_tracking_utils/vis_grad.py


from typing import Optional, Tuple
import numpy as np
import torch
import open3d as o3d


def build_sdf_grad_lineset(
    points_w: np.ndarray,
    voxel_field,
    decoder,
    max_arrows: int = 2000,
    step: int = 50,
    arrow_len: float = 0.5,
    eps_g: float = 1e-6,
) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()

    if points_w is None:
        _set_empty_lineset(ls)
        return ls

    if points_w.ndim != 2 or points_w.shape[1] != 3 or points_w.shape[0] == 0:
        _set_empty_lineset(ls)
        return ls

    pts = np.asarray(points_w, dtype=np.float32)

    # --- pick indices for arrows ---
    idx = np.arange(0, pts.shape[0], max(1, int(step)), dtype=np.int64)
    if idx.size == 0:
        _set_empty_lineset(ls)
        return ls
    if idx.size > int(max_arrows):
        idx = np.random.choice(idx, size=int(max_arrows), replace=False)

    pts_s = pts[idx]  # (M,3)

    # --- infer device/dtype from voxel_field/decoder ---
    if hasattr(voxel_field, "feature_indexs_list") and len(voxel_field.feature_indexs_list) > 0:
        dev = voxel_field.feature_indexs_list[0].device
    else:
        dev = next(decoder.parameters()).device
    dt = getattr(voxel_field, "dtype", None) or next(decoder.parameters()).dtype

    # --- autograd grad ---
    x = torch.from_numpy(pts_s).to(device=dev, dtype=dt)
    x.requires_grad_(True)

    with torch.enable_grad():
        feats = voxel_field.get_features(x)
        sdf = decoder(feats).reshape(-1)  # (M,)
        g = torch.autograd.grad(
            outputs=sdf.sum(),
            inputs=x,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]  # (M,3)

    g = g.detach()
    gn = torch.linalg.norm(g, dim=-1)  # (M,)

    valid = gn > float(eps_g)
    if int(valid.sum().item()) == 0:
        _set_empty_lineset(ls)
        return ls

    x0 = x.detach()[valid].to("cpu").numpy().astype(np.float64)        # (K,3)
    d  = (g[valid] / (gn[valid].unsqueeze(-1) + 1e-12)).to("cpu").numpy().astype(np.float64)
    x1 = x0 + float(arrow_len) * d                                     # (K,3)

    # --- build lineset ---
    K = x0.shape[0]
    pts_lines = np.vstack([x0, x1]).astype(np.float64)                 # (2K,3)
    lines = np.stack([np.arange(K), np.arange(K) + K], axis=1).astype(np.int32)

    cols = np.tile(np.array([[1.0, 1.0, 0.0]], dtype=np.float64), (K, 1))

    ls.points = o3d.utility.Vector3dVector(pts_lines)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(cols)
    return ls


def _set_empty_lineset(ls: o3d.geometry.LineSet):
    ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
    ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
