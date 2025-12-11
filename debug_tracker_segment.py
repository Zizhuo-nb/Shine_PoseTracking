# debug_train_and_track_segment.py

import os
from pathlib import Path
import torch

from utils.config import Config
from utils.dataLoader import dataLoader
from utils.dataSampler import dataSampler

from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder

from mapping import static_mapping
from tracker import track_sequence


if __name__ == "__main__":
    cfg_path = "config/zappa/zappa.yaml"
    data_dir = "/automount_home_students/zzhang/research/ThirdSemester/project_part2/data/cppdata/Data/PLY/"
    out_root_dir = "./debug_4300_4400" 

    begin = 4325
    end   = 4350    
    cg = Config()
    cg.load(cfg_path)
    cg.yaml_name   = Path(cfg_path).stem
    cg.begin_frame = begin
    cg.end_frame   = end

    os.makedirs(out_root_dir, exist_ok=True)
    pose_file = os.path.join(out_root_dir, "poses.txt")
    err_file  = os.path.join(out_root_dir, "errors.txt")


    if os.path.exists(pose_file):
        os.remove(pose_file)
    if os.path.exists(err_file):
        os.remove(err_file)

    dataloader = dataLoader(points_floder=cg.data_path,
                            pose_path=cg.pose_path,
                            calib_path=cg.calib_path,
                            device=cg.device)

    sampler = dataSampler(truncated_area=cg.truncated_sample_range_m,
                          truncated_num=cg.truncated_sample_num,
                          occupied_area=cg.occupied_sample_range_m,
                          occupied_num=cg.occupied_sample_num,
                          free_space_num=cg.free_sample_num,
                          device=cg.device,
                          dtype=cg.dtype)


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

    print(f"[INFO] Train submap {begin}-{end-1} ...")
    static_mapping(
        cg,          
        voxel_field,
        mlp,
        dataloader,
        sampler,
        cg.begin_frame,
        cg.end_frame
    )

    del voxel_field, mlp
    torch.cuda.empty_cache()

    run_name = f"{cg.yaml_name}_{begin:06d}-{end-1:06d}"
    ckpt_path = os.path.join("model_save", f"{run_name}.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")


    poses_out, last_pose, need_refine = track_sequence(
        cg=cg,
        ckpt_path=ckpt_path,
        data_dir=data_dir,   
        out_root_dir=out_root_dir,
        init_pose_override=None,
        pose_file=pose_file,     
        err_file=err_file,
        err_thresh=0.3,      
    )

    print(f"[INFO] Done segment {begin}-{end-1}")
    print(f"[INFO] last_pose:\n{last_pose}")
    print(f"[INFO] need_refine = {need_refine}")
