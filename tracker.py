import os
import torch
import torch.nn as nn
from pathlib import Path

from utils.dataLoader import read_calib_file, read_poses_file
import numpy as np
from utils.config import Config
from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder

from pose_tracking_utils import icp_solver




def track_sequence(cg: Config,
                   ckpt_path: str,
                   data_dir: str,
                   out_root_dir: str = None,
                   init_pose_override: torch.Tensor = None,
                   pose_file: str = None,     
                   err_file: str = None,
                   err_thresh: float = 0.1,
                   use_constant_velocity: bool = False,
                   tracker_max_dist: float = None):

    device = cg.device
    dtype = cg.dtype
    max_dist = tracker_max_dist if tracker_max_dist is not None else 22
    if out_root_dir is not None:
        if not os.path.exists(out_root_dir):
            os.makedirs(out_root_dir, exist_ok=True)
#####################################################################################
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
#########################################read pose for initial guess################################
    calib = {"Tr": np.eye(4)}
    pose_path = cg.pose_path
    gt_poses = read_poses_file(pose_path, calib)

    if init_pose_override is not None:
        init_T = init_pose_override.to(device=device, dtype=torch.float64)
    else:
        init_T = torch.tensor(gt_poses[cg.begin_frame], device=device, dtype=torch.float64)
#############################################################################################################
    
    tracker = icp_solver.Posetracker(
        test_folder=data_dir,
        voxel_field=voxel_field,
        geo_decoder=mlp,
        device=device,
        start_idx=cg.begin_frame,
        end_idx=cg.end_frame,
        max_dist=max_dist,
        GM_k=0.11,
        num_iter=5,
        init_pose=init_T,
        radius=cg.radius,
        min_z=cg.min_z,
        use_constant_velocity = use_constant_velocity
    )

    poses_out = []
    need_refine = False
    num_scans = len(tracker.file_list)

    pose_f = None
    err_f = None
    if pose_file is not None:
        pose_mode = "a" if os.path.exists(pose_file) else "w"
        err_mode  = "a" if os.path.exists(err_file) else "w"
        pose_f = open(pose_file, pose_mode)
        err_f  = open(err_file, err_mode)


    for i in range(num_scans):
        global_idx = cg.begin_frame + i
        T_global, scan = tracker.register_next()
        poses_out.append(T_global.clone().cpu())

        ###################### debug ###############
        gt_T = torch.tensor(gt_poses[global_idx], device=device, dtype=torch.float64)
        gt_flat = gt_T[:3, :4].reshape(-1)       
        est_flat = T_global[:3, :4].reshape(-1)    
        err_flat = est_flat - gt_flat           
        err_list = err_flat.detach().cpu().tolist()
        
        print(f"[CMP] frame {global_idx}")
        print("  err:  " + " ".join(f"{v: .3f}" for v in err_list))

        if i == num_scans - 1:
            ex, ey, ez = err_list[3], err_list[7], err_list[11]
            print(ex,ey,ez)
            if (abs(ex) > err_thresh or
                abs(ey) > err_thresh or
                abs(ez) > err_thresh):
                need_refine = True
        ###################### debug ###############
        if pose_f is not None:
            vals = T_global.detach().cpu().numpy()[:3, :].reshape(-1).tolist()
            line_pose = " ".join(f"{v:.6f}" for v in vals)
            pose_f.write(line_pose + "\n")

        if err_f is not None:
            line_err = f"{global_idx:6d}" + "".join(f"{v:10.3f}" for v in err_list)
            err_f.write(line_err + "\n")

    if pose_f is not None:
        pose_f.close()
    if err_f is not None:
        err_f.close()


    last_pose = poses_out[-1].to(device=device, dtype=torch.float64)
    return poses_out, last_pose , need_refine

