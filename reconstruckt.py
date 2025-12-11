import os
import sys
import torch
import torch.nn as nn
import numpy as np

from utils.config import Config
from utils.dataLoader import dataLoader, read_calib_file, read_poses_file
from utils import mesher

from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder


def load_model_from_ckpt(cg: Config, ckpt_path: str, device: torch.device):
   
    voxel_field = NeuralHashVoxel(
        feature_dim=cg.feature_dim,
        feature_std=cg.feature_std,
        leaf_voxel_size=cg.leaf_voxel_size,
        voxel_level_num=cg.voxel_level_num,
        up_scale_factor=cg.scale_up_factor,
        device=device,
        dtype=cg.dtype,
        buffer_size=int(cg.hash_buffer_size),
    )

    mlp = decoder.mlp(
        input_dim=cg.feature_dim,
        layers_num=cg.mlp_level,
        hidden_dim=cg.mlp_hidden_dim,
        output_dim=1,
        with_basis=True,
        device=device,
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

    return voxel_field, mlp


def compute_global_bbox(cg: Config, dataloader: dataLoader, device: torch.device):
    global_min = torch.tensor([+1e9, +1e9, +1e9], device=device, dtype=cg.dtype)
    global_max = torch.tensor([-1e9, -1e9, -1e9], device=device, dtype=cg.dtype)

    for i in range(cg.begin_frame, cg.end_frame):
        print(f"[BBOX] Accumulating frame {i}")

        points = dataloader.frame_transfered(i).to(device=device, dtype=cg.dtype)
        translation = dataloader.translation(i).to(device=device, dtype=cg.dtype)

        distances = torch.norm(points - translation, p=2, dim=1, keepdim=False)
        valid_mask = distances < cg.radius
        points = points[valid_mask]
        points = points[points[:, 2] > cg.min_z]

        if points.shape[0] == 0:
            continue

        cur_min = points.min(dim=0)[0]
        cur_max = points.max(dim=0)[0]

        global_min = torch.minimum(global_min, cur_min)
        global_max = torch.maximum(global_max, cur_max)

    min_x, min_y, min_z = global_min.tolist()
    max_x, max_y, max_z = global_max.tolist()

    print("[BBOX] Global bbox:")
    print(f"  x: [{min_x:.3f}, {max_x:.3f}]")
    print(f"  y: [{min_y:.3f}, {max_y:.3f}]")
    print(f"  z: [{min_z:.3f}, {max_z:.3f}]")

    return min_x, max_x, min_y, max_y, min_z, max_z


def reconstruct_scene(cfg_path: str, ckpt_path: str, out_mesh_name: str = "mesh_full_scene.ply"):

    cg = Config()
    cg.load(cfg_path)
    device = cg.device

    print("[INFO] Loading model...")
    voxel_field, mlp = load_model_from_ckpt(cg, ckpt_path, device)

    print("[INFO] Building dataloader for bbox computation...")
    dataloader = dataLoader(
        points_floder=cg.data_path,
        pose_path=cg.pose_path,
        calib_path=cg.calib_path,
        device=device,
    )

    min_x, max_x, min_y, max_y, min_z, max_z = compute_global_bbox(cg, dataloader, device)

    os.makedirs(cg.output_root, exist_ok=True)
    mesh_path = os.path.join(cg.output_root, out_mesh_name)

    print("[INFO] Reconstructing mesh with mesher.create_mesh ...")
    mesh = mesher.create_mesh(
        voxel_field,
        mlp,
        mesh_path,
        max_x,
        min_x,
        max_y,
        min_y,
        max_z,
        min_z,
        cg.mesh_resolution,
        scale=1.0,
    )

    print(f"[INFO] Mesh reconstruction done. Saved to: {mesh_path}")
    return mesh_path


if __name__ == "__main__":
    """
    python reconstruct_all.py config/zappa/zappa.yaml experiments/model_000099.pth
    """
    if len(sys.argv) < 3:
        print("Usage: python reconstruct_all.py <config_path> <ckpt_path>")
        sys.exit(1)

    cfg_path = sys.argv[1]
    ckpt_path = sys.argv[2]

    reconstruct_scene(cfg_path, ckpt_path)