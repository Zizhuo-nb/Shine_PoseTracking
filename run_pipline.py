# run_pipeline.py
import os
from pathlib import Path
import torch

from utils.config import Config
from utils.dataLoader import dataLoader
from utils.dataSampler import dataSampler

from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder
import numpy as np
from mapping import static_mapping
from tracker import track_sequence


def run_pipeline(cfg_path: str,
                 data_dir: str,
                 out_root_root: str,
                 model_root: str = "model_save",
                 coarse_len: int = 500,
                 sub_len: int = 100,
                 delet: bool = True,
                 refine_len: int = 50,
                 err_thresh: float = 0.1):

    cg_global = Config()
    cg_global.load(cfg_path)
    cg_global.yaml_name = Path(cfg_path).stem
    global_begin = cg_global.begin_frame
    global_end = cg_global.end_frame
########################for debuging, if you wanna start from middle############################
    # init_np = np.array([
    #     [ -0.313540, 0.948580, 0.043457, -15.023902],
    #     [-0.942258, -0.316468, 0.109534, -112.979258],
    #     [ 0.117655, -0.006604, 0.993032, 6.690242],
    #     [ 0.0,       0.0,       0.0,        1.0      ],
    # ], dtype=np.float64)
    # first_init_pose = torch.tensor(init_np, device=cg_global.device, dtype=torch.float64)
################################################################################################
    
    run_name = f"{cg_global.yaml_name}_{global_begin:06d}-{global_end-1:06d}"
    out_root_dir = os.path.join(out_root_root, run_name)
    os.makedirs(out_root_dir, exist_ok=True)

    pose_file = os.path.join(out_root_dir, "poses.txt")
    err_file  = os.path.join(out_root_dir, "errors.txt")

    if os.path.exists(pose_file):
        os.remove(pose_file)
    if os.path.exists(err_file):
        os.remove(err_file)


    dataloader = dataLoader(points_floder=cg_global.data_path,
                            pose_path=cg_global.pose_path,
                            calib_path=cg_global.calib_path,
                            device=cg_global.device)

    sampler = dataSampler(truncated_area=cg_global.truncated_sample_range_m,
                          truncated_num=cg_global.truncated_sample_num,
                          occupied_area=cg_global.occupied_sample_range_m,
                          occupied_num=cg_global.occupied_sample_num,
                          free_space_num=cg_global.free_sample_num,
                          device=cg_global.device,
                          dtype=cg_global.dtype)

    last_pose: torch.Tensor = None 
    start = global_begin
    end_global = global_end

    while start < end_global:
        end = min(start + sub_len, end_global)
        print(f"\n========== Submap {start} - {end-1} ==========")

        if last_pose is None :
            entry_pose = None
            # entry_pose = first_init_pose
        else:
            entry_pose = last_pose

        cg_sub = Config()
        cg_sub.load(cfg_path)
        cg_sub.yaml_name = cg_global.yaml_name
        cg_sub.begin_frame = start
        cg_sub.end_frame = end

        voxel_field = NeuralHashVoxel(
            feature_dim=cg_sub.feature_dim,
            feature_std=cg_sub.feature_std,
            leaf_voxel_size=cg_sub.leaf_voxel_size,
            voxel_level_num=cg_sub.voxel_level_num,
            up_scale_factor=cg_sub.scale_up_factor,
            device=cg_sub.device,
            dtype=cg_sub.dtype,
            buffer_size=int(cg_sub.hash_buffer_size),
        )

        mlp = decoder.mlp(
            input_dim=cg_sub.feature_dim,
            layers_num=cg_sub.mlp_level,
            hidden_dim=cg_sub.mlp_hidden_dim,
            output_dim=1,
            with_basis=True,
            device=cg_sub.device,
        )

        static_mapping(cg_sub, voxel_field, mlp, dataloader, sampler,
                       cg_sub.begin_frame, cg_sub.end_frame)

        del voxel_field, mlp
        torch.cuda.empty_cache()

        run_name_sub = f"{cg_sub.yaml_name}_{cg_sub.begin_frame:06d}-{cg_sub.end_frame-1:06d}"
        ckpt_path = os.path.join(model_root, f"{run_name_sub}.pth")

        _, _, need_refine = track_sequence(
            cg=cg_sub,
            ckpt_path=ckpt_path,
            data_dir=data_dir,
            out_root_dir=out_root_dir,
            init_pose_override=None,
            pose_file=None,     
            err_file=None, 
            err_thresh=err_thresh,
            use_constant_velocity=False
        )

        if (not need_refine) or ((end - start) <= refine_len):
            print(f"[INFO] Submap {start}-{end-1} OK, write final results")
            _, last_pose, _ = track_sequence(
                cg=cg_sub,
                ckpt_path=ckpt_path,
                data_dir=data_dir,
                out_root_dir=out_root_dir,
                init_pose_override=entry_pose,
                pose_file=pose_file,
                err_file=err_file,
                err_thresh=err_thresh,
                use_constant_velocity=False,
            )

            if delet and os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print(f"[INFO] Removed ckpt: {ckpt_path}")

            start = end
            continue

       #####################################################################################
        print(f"[INFO] Submap {start}-{end-1} has large error, re-train with 2×{refine_len} frames")

        if delet and os.path.exists(ckpt_path):
            os.remove(ckpt_path)
            print(f"[INFO] Removed coarse ckpt: {ckpt_path}")

        refine_pose = entry_pose 
        mid = start + refine_len

        for sub_start, sub_end in [(start, mid), (mid, end)]:
            print(f"[INFO]   refine sub-submap {sub_start}-{sub_end-1}")

            cg_sub_small = Config()
            cg_sub_small.load(cfg_path)
            cg_sub_small.yaml_name = cg_global.yaml_name
            cg_sub_small.begin_frame = sub_start
            cg_sub_small.end_frame = sub_end


            voxel_field = NeuralHashVoxel(
                feature_dim=cg_sub_small.feature_dim,
                feature_std=cg_sub_small.feature_std,
                leaf_voxel_size=cg_sub_small.leaf_voxel_size,
                voxel_level_num=cg_sub_small.voxel_level_num,
                up_scale_factor=cg_sub_small.scale_up_factor,
                device=cg_sub_small.device,
                dtype=cg_sub_small.dtype,
                buffer_size=int(cg_sub_small.hash_buffer_size),
            )

            mlp = decoder.mlp(
                input_dim=cg_sub_small.feature_dim,
                layers_num=cg_sub_small.mlp_level,
                hidden_dim=cg_sub_small.mlp_hidden_dim,
                output_dim=1,
                with_basis=True,
                device=cg_sub_small.device,
            )

            static_mapping(cg_sub_small, voxel_field, mlp, dataloader, sampler,
                           cg_sub_small.begin_frame, cg_sub_small.end_frame)

            del voxel_field, mlp
            torch.cuda.empty_cache()

            run_name_small = f"{cg_sub_small.yaml_name}_{cg_sub_small.begin_frame:06d}-{cg_sub_small.end_frame-1:06d}"
            ckpt_path_small = os.path.join(model_root, f"{run_name_small}.pth")

            _, _, need_refine_small = track_sequence(
                cg=cg_sub_small,
                ckpt_path=ckpt_path_small,
                data_dir=data_dir,
                out_root_dir=out_root_dir,
                init_pose_override=refine_pose,
                pose_file=None,
                err_file=None,
                err_thresh=err_thresh,
                use_constant_velocity=False,
            )

            if not need_refine_small:
                print("[INFO]  OK!!!  Start to write into file")
                _, refine_pose, _ = track_sequence(
                    cg=cg_sub_small,
                    ckpt_path=ckpt_path_small,
                    data_dir=data_dir,
                    out_root_dir=out_root_dir,
                    init_pose_override=refine_pose,
                    pose_file=pose_file,
                    err_file=err_file,
                    err_thresh=err_thresh,
                    use_constant_velocity=False,
                )
            else:
                print(f"[INFO]   sub-submap {sub_start}-{sub_end-1} still large error, enable constant velocity")
                _, _, need_refine_small = track_sequence(
                    cg=cg_sub_small,
                    ckpt_path=ckpt_path_small,
                    data_dir=data_dir,
                    out_root_dir=out_root_dir,
                    init_pose_override=refine_pose,
                    pose_file=None,
                    err_file=None,
                    err_thresh=err_thresh,
                    use_constant_velocity=True,
                )

                if not need_refine_small:
                    print("[INFO] OK!!!! Start to write into file")
                    _, refine_pose, _ = track_sequence(
                        cg=cg_sub_small,
                        ckpt_path=ckpt_path_small,
                        data_dir=data_dir,
                        out_root_dir=out_root_dir,
                        init_pose_override=refine_pose,
                        pose_file=pose_file,
                        err_file=err_file,
                        err_thresh=err_thresh,
                        use_constant_velocity=True,
                    )
                else:
                    print(f"[INFO]   sub-submap {sub_start}-{sub_end-1} still large error, "
                        f"enable CV + small max_dist")
                    # _, _, _ = track_sequence(
                    #     cg=cg_sub_small,
                    #     ckpt_path=ckpt_path_small,
                    #     data_dir=data_dir,
                    #     out_root_dir=out_root_dir,
                    #     init_pose_override=refine_pose,
                    #     pose_file=None,
                    #     err_file=None,
                    #     err_thresh=err_thresh,
                    #     use_constant_velocity=True,
                    #     tracker_max_dist=15,
                    # )
                    print("[INFO] Have no idea (at least now no)  OK!!!! Start to write into file")
                    _, refine_pose, _ = track_sequence(
                        cg=cg_sub_small,
                        ckpt_path=ckpt_path_small,
                        data_dir=data_dir,
                        out_root_dir=out_root_dir,
                        init_pose_override=refine_pose,
                        pose_file=pose_file,
                        err_file=err_file,
                        err_thresh=err_thresh,
                        use_constant_velocity=True,
                        tracker_max_dist=15,
                    )


            if delet and os.path.exists(ckpt_path_small):
                os.remove(ckpt_path_small)
                print(f"[INFO] Removed refine ckpt: {ckpt_path_small}")

        last_pose = refine_pose
        start = end


if __name__ == "__main__":
    cfg_path = "config/zappa/zappa.yaml"
    data_dir = "/automount_home_students/zzhang/research/ThirdSemester/project_part2/data/cppdata/Data/PLY/"
    out_root_root = "./results_sub_map"

    run_pipeline(
        cfg_path=cfg_path,
        data_dir=data_dir,
        out_root_root=out_root_root,
        model_root="model_save",
        sub_len=100,
        delet=True,
        refine_len=50,
        err_thresh=0.2,
    )
