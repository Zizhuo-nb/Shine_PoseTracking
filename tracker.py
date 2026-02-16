import os
import torch
from pathlib import Path
import yaml
import argparse
import torch.nn as nn
import numpy as np

from pose_tracking_utils.draw import draw_txt
from utils.config import Config
from utils.dataLoader import read_poses_file
from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder

from pose_tracking_utils import icp_solver
from pose_tracking_utils.live_visu import run_live_viz

# ---------------- main pipeline ----------------
def track_sequence(cfg_path: str,
                   ckpt_path: str,
                   out_pose_path: str = None):
    # --------- read yaml (for switches/paths) ----------
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}
    setting = raw_cfg.get("setting", {})

    data_dir = setting.get("data_path", None)
    pose_path = setting.get("pose_path", None)
    output_root = setting.get("output_root", "results")
    begin_frame = int(setting.get("begin_frame", 0))
    end_frame = int(setting.get("end_frame", -1))
    enable_vis = bool(setting.get("enable_vis", False))
    vis_n = int(setting.get("vis_n", 5))

    # plotting option (optional)
    plot_equal = bool(setting.get("plot_equal_axis", False))

    if not data_dir:
        raise ValueError("YAML doesn't have setting.data_path")
    if not pose_path:
        raise ValueError("YAML doesn't have setting.pose_path")

    map_ply = setting.get("map_ply", None)
    if enable_vis and (not map_ply):
        raise ValueError("YAML doesn't have setting.map_ply but enable_vis=true")

    out_dir = Path(output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    if out_pose_path is None:
        yaml_name = Path(cfg_path).stem
        run_name = f"{yaml_name}_{begin_frame:06d}-{end_frame:06d}"
        pose_dir = out_dir / "pose_global"
        pose_dir.mkdir(parents=True, exist_ok=True)
        out_pose_path = str(pose_dir / f"{run_name}.txt")


    # --------- load model cfg ----------
    cg = Config()
    cg.load(cfg_path)
    device = cg.device

    # --------- build voxel field + mlp ----------
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

    # Make sure features_list params exist with correct shapes
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

    # --------- init pose from GT ----------
    calib = {"Tr": np.eye(4)}
    gt_poses = read_poses_file(pose_path, calib)  # list of 4x4 np.array
    gt_poses_torch = [torch.as_tensor(T, device=device, dtype=torch.float64) for T in gt_poses]

    if begin_frame < 0 or begin_frame >= len(gt_poses_torch):
        raise IndexError(f"begin_frame={begin_frame} is more than gt_poses len={len(gt_poses_torch)}")

    init_T = gt_poses_torch[begin_frame].clone()
    # print(init_T)

    # --------- create tracker ----------
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

    # normalize end_frame
    if end_frame < 0:
        end_frame = len(tracker.file_list) - 1
    end_frame = min(end_frame, len(tracker.file_list) - 1)

    # --------- mode switch ----------
    if enable_vis:
        run_live_viz(
            tracker=tracker,
            map_ply=map_ply,
            begin_frame=begin_frame,
            end_frame=end_frame,
            vis_n=vis_n,
        )
        return

    # --------- tracking only: save poses then plot ----------
    poses_out = []
    while tracker.running_idx <= end_frame:
        i = int(tracker.running_idx)
        print(f"[INFO] Processing frame {i}: {os.path.basename(tracker.file_list[i])}")
        T_global, _ = tracker.register_next_global()
        poses_out.append(T_global.detach().cpu())

    # Write as 12 floats per line (your plot script reads this)
    os.makedirs(os.path.dirname(out_pose_path) or ".", exist_ok=True)
    with open(out_pose_path, "w", encoding="utf-8") as f:
        for T in poses_out:
            Tn = T.numpy().astype(np.float64)
            vals = Tn[:3, :].reshape(-1).tolist()  # 12 values
            f.write(" ".join(f"{v:.8f}" for v in vals) + "\n")

    print(f"[INFO] Saved poses: {out_pose_path}")

    # Plot immediately (same process)
    draw_txt(out_pose_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose tracking: YAML enable_vis controls live viz vs plot")
    parser.add_argument("--cfg", required=True, help="Path to YAML config")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--out_pose", default=None, help="Output pose txt path (optional)")
    args = parser.parse_args()

    print("[INFO] tracker main start")
    track_sequence(args.cfg, args.ckpt, out_pose_path=args.out_pose)
