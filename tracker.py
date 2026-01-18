import os
import math
import open3d as o3d
import copy
import glob
import torch
from pathlib import Path
import yaml
import argparse
import torch.nn as nn
from utils.dataLoader import read_calib_file, read_poses_file
import numpy as np

from utils.config import Config
from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder

from pose_tracking_utils import icp_solver
from pose_tracking_utils.utils import compute_pose_error,run_full_debug_csv
from pose_tracking_utils.live_visu import run_live_viz_step0


def track_sequence(cfg_path: str,
                   ckpt_path: str,
                   out_pose_path: str = None,
                   out_err_path: str = None):
    ##################################################
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    setting = raw_cfg.get("setting", {})
    data_dir = setting.get("data_path", None)
    pose_path = setting.get("pose_path", None)
    output_root = setting.get("output_root", "results")
    begin_frame = int(setting.get("begin_frame", 0))
    end_frame = int(setting.get("end_frame", -1))
    if not data_dir:
        raise ValueError("YAML doesn't have setting.data_path")
    if not pose_path:
        raise ValueError("YAML doesn't have setting.pose_path")
    
    map_ply = setting.get("map_ply", None)
    if not map_ply:
        raise ValueError("YAML doesn't have setting.map_ply")

    # vis_n = int(setting.get("vis_n", 5))
    # vis_k = int(setting.get("vis_k", 50))
    

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if out_pose_path is None:
        out_pose_path = str(out_dir / "tracked_poses.txt")
    if out_err_path is None:
        out_err_path = str(out_dir / "pose_errors.txt")
    ######################################################
    cg = Config()
    cg.load(cfg_path)
    device = cg.device
    ##################################################
    voxel_field = NeuralHashVoxel(
        feature_dim=cg.feature_dim,
        feature_std=cg.feature_std,
        leaf_voxel_size=cg.leaf_voxel_size,
        voxel_level_num=cg.voxel_level_num,
        up_scale_factor=cg.scale_up_factor,
        device=cg.device,
        dtype=cg.dtype,
        buffer_size=int(cg.hash_buffer_size),
    )

    mlp = decoder.mlp(
        input_dim=cg.feature_dim,
        layers_num=cg.mlp_level,
        hidden_dim=cg.mlp_hidden_dim,
        output_dim=1,
        with_basis=True,
        device=cg.device,
    )

    ckpt = torch.load(ckpt_path, map_location=device)
    vf_state = ckpt["voxel_field"]

    with torch.no_grad():
        for level in range(len(voxel_field.features_list)):
            key = f"features_list.{level}"
            if key in vf_state:
                shape = vf_state[key].shape
                param_dtype = vf_state[key].dtype
                voxel_field.features_list[level] = nn.Parameter(
                    torch.empty(shape, device=device, dtype=param_dtype)
                )
    voxel_field.load_state_dict(vf_state)
    mlp.load_state_dict(ckpt["mlp"])

    with torch.no_grad():
        saved_index_lists = ckpt["feature_indexs_list"]
        for dst, src in zip(voxel_field.feature_indexs_list, saved_index_lists):
            dst.copy_(src.to(device=device))

    voxel_field.eval()
    mlp.eval()
    for p in voxel_field.parameters():
        p.requires_grad_(False)
    for p in mlp.parameters():
        p.requires_grad_(False)
    ############################################################

    calib = {"Tr": np.eye(4)}
    gt_poses = read_poses_file(pose_path, calib)   # list of 4x4 np.array
    gt_poses_torch = [torch.as_tensor(T, device=device, dtype=torch.float64) for T in gt_poses]
    if begin_frame < 0 or begin_frame >= len(gt_poses):
        raise IndexError(f"begin_frame={begin_frame} is more than gt_poses len={len(gt_poses)}")
    init_T = gt_poses_torch[begin_frame].clone()
    print(init_T)
    ###############
    gt_poses_np = [T for T in gt_poses]
    #################
#############################################################################################################
    scalar_factor = -5.0 / cg.truncated_sample_range_m
    tracker = icp_solver.Posetracker(
        test_folder=data_dir,
        voxel_field=voxel_field,
        geo_decoder=mlp,
        device=device,
        start_idx=begin_frame,
        max_dist=30,
        GM_k=0.3,
        num_iter=10,
        init_pose=init_T,
        radius=cg.radius,
        min_z=cg.min_z,
    )

    # if end_frame < 0:
    #     end_frame = len(tracker.file_list) - 1
    # end_frame = min(end_frame, len(tracker.file_list) - 1)
    # auto = bool(setting.get("auto_debug_csv", False))
    # if auto:
    #     csv_path = str(setting.get("debug_csv_path", str(out_dir / "sdf_debug_full.csv")))
    #     run_full_debug_csv(tracker, gt_poses_np, begin_frame, end_frame, csv_path)
    #     return
    ########################
    poses_out = []
    err_rows = []
    while tracker.running_idx <= end_frame:
        i = tracker.running_idx
        print(f"[INFO] Processing frame {i}: {os.path.basename(tracker.file_list[i])}")
        T_global,_ = tracker.register_next()
        poses_out.append(T_global.clone().cpu()) 

        T_gt = gt_poses_torch[i]
        t_err, r_err, dx, dy, dz = compute_pose_error(T_global, T_gt)
        print(f"[ERR] frame {i}: t_err={t_err:.4f} m, r_err={r_err:.3f} deg, "
              f"dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
        err_rows.append((i, t_err, r_err, dx, dy, dz))

    with open(out_pose_path, "w", encoding="utf-8") as f:
        for T in poses_out:
            Tn = T.numpy()
            vals = Tn[:3, :].reshape(-1).tolist()
            f.write(" ".join(f"{v:.5f}" for v in vals) + "\n")

    with open(out_err_path, "w", encoding="utf-8") as f:
        f.write("# frame t_err(m) r_err(deg) dx dy dz\n")
        for (i, t_err, r_err, dx, dy, dz) in err_rows:
            f.write(f"{i} {t_err:.6f} {r_err:.6f} {dx:.6f} {dy:.6f} {dz:.6f}\n")

    print(f"[INFO] Saved poses:  {out_pose_path}")
    print(f"[INFO] Saved errors: {out_err_path}")
############################
#     run_live_viz_step0(
#     tracker=tracker,
#     map_ply=map_ply,
#     begin_frame=begin_frame,
#     end_frame=end_frame,
#     gt_poses_np=gt_poses_np,
#     gt_poses_torch=gt_poses_torch,
#     out_pose_path=out_pose_path,
#     out_err_path=out_err_path,
#     vis_n=vis_n,
#     vis_k=1,
#     resume_from_file=False,
# )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pose tracking + per-frame GT error print/save")
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--out_pose", default=None, help="Output pose txt path (optional)")
    parser.add_argument("--out_err", default=None, help="Output error txt path (optional)")
    args = parser.parse_args()

    print("[INFO] tracker main start")
    track_sequence(args.cfg, args.ckpt, out_pose_path=args.out_pose, out_err_path=args.out_err)
