import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

import open3d as o3d
from tqdm import tqdm
from collections import deque

from utils.config import Config
from utils.dataLoader import dataLoader
from utils.dataSampler import dataSampler
from utils.visualizer import MapVisualizer
from utils import tools
from utils import mesher

from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder

def static_mapping(cg, voxel_field, mlp, dataloder, sampler, startframe, endframe):

    map_samples = torch.tensor([], device=cg.device, dtype=cg.dtype)
    map_labels = torch.tensor([], device=cg.device, dtype=cg.dtype)

    window_size = cg.sliding_window_size
    points_num_deque = deque()

    # main loop
    vis = None
    ENABLE_MESH_SAVE = False
    ###############3mesher#############3
    gmin_x = gmin_y = gmin_z = float("inf")
    gmax_x = gmax_y = gmax_z = float("-inf")
    mesh = None
    ###################################
    for i in range(startframe, endframe):

        time_step = i - startframe
        print(f"Processing frame {i}")

        points = dataloder.frame_transfered(i).to(cg.dtype)
        current_translation = dataloder.translation(i).to(cg.dtype)

        points_distances = torch.norm(points - current_translation, p=2, dim=1, keepdim=False)
        valid_mask = points_distances<cg.radius

        points = points[valid_mask]
        points = points[points[:,2] > cg.min_z]
        ##################mesher####################
        if points.numel() > 0:
            gmax_x = max(gmax_x, points[:,0].max().item())
            gmin_x = min(gmin_x, points[:,0].min().item())
            gmax_y = max(gmax_y, points[:,1].max().item())
            gmin_y = min(gmin_y, points[:,1].min().item())
            gmax_z = max(gmax_z, points[:,2].max().item())
            gmin_z = min(gmin_z, points[:,2].min().item())

        ############################################

        max_x = torch.max(points[:,0]).item()
        min_x = torch.min(points[:,0]).item()
        max_y = torch.max(points[:,1]).item()
        min_y = torch.min(points[:,1]).item()
        max_z = torch.max(points[:,2]).item()
        min_z = torch.min(points[:,2]).item()

        trans_tensor = current_translation.repeat(points.shape[0],1)
        surface_sample, surface_pd, free_sample, free_pd = sampler.ray_sample(points, trans_tensor)

        voxel_field.update(surface_sample)

        this_sample = torch.cat((surface_sample, free_sample), dim=0)
        this_label = torch.cat((surface_pd, free_pd),dim=0)
        points_num_deque.append(this_sample.shape[0])

        field_param = list(voxel_field.parameters())
        mlp_param = list(mlp.parameters())

        field_param_opt_dict = {'params': field_param, 'lr': 0.001}
        mlp_param_opt_dict = {'params': mlp_param, 'lr': 0.001}
        # fix the mlp after first window's traning
        if time_step <= window_size:
            pre_samples = map_samples
            pre_labels = map_labels
            param_list = [field_param_opt_dict, mlp_param_opt_dict]
        else:
            throw_num = points_num_deque.popleft()
            pre_samples = map_samples[throw_num:,:]
            pre_labels = map_labels[throw_num:]
            tools.freeze_model(mlp)
            param_list = [field_param_opt_dict]

        map_samples = torch.cat((pre_samples, this_sample), dim=0)
        map_labels = torch.cat((pre_labels, this_label.squeeze(0)), dim=0)

        opt = optim.Adam(param_list, betas=(0.9,0.99), eps = 1e-15)

        if time_step == 0:
            map_iterations = cg.iterion_num*10
        else:
            map_iterations = cg.iterion_num

        loss_bce = nn.BCELoss()
        scalar_factor = -5.0/cg.truncated_sample_range_m   
        with tqdm(total=map_iterations) as pbar:
            pbar.set_description('map Processing:')
            for iteration in range(map_iterations):
                total_point_num = map_samples.shape[0]

                indices = torch.randint(0, total_point_num, (cg.batch_size,), device='cuda')
                
                data_sample = map_samples[indices].float()
                label_sample = map_labels[indices].float()

                feature_vectors = voxel_field.get_features(data_sample)
                pred = mlp(feature_vectors.float())

                pred_occ = torch.sigmoid(pred*scalar_factor)
                label_occ = torch.sigmoid(label_sample*scalar_factor)

                loss = loss_bce(pred_occ, label_occ)

                loss.backward()
                opt.step()
                opt.zero_grad()
                pbar.update(1)

        # just visulization
        if ENABLE_MESH_SAVE and time_step % window_size == 0:
            mesh_path = cg.output_root + 'mesh_' + str(i) 
            mesh = mesher.create_mesh(voxel_field, mlp, mesh_path, max_x, min_x, max_y, min_y, max_z, min_z, cg.mesh_resolution, scale=1)
            mesh.compute_vertex_normals()

        scan = o3d.geometry.PointCloud()
        scan.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        if vis is not None:
            vis.update(scan=scan, mesh=mesh)
    #######################save model##########################
    run_name = f"{cg.yaml_name}_{startframe:06d}-{endframe-1:06d}"
    model_root = getattr(cg, "model_root", None)
    if not model_root:
        model_root = os.path.join(cg.output_root, "models")
    os.makedirs(model_root, exist_ok=True)
    save_path = os.path.join(model_root, f"{run_name}.pth")
    checkpoint = {
        "voxel_field": voxel_field.state_dict(),
        "mlp": mlp.state_dict(),
        "feature_indexs_list": [idx.clone().cpu()
                                for idx in voxel_field.feature_indexs_list],
        "config": cg.__dict__,
    }
    torch.save(checkpoint, save_path)
    print(f"[INFO] Model saved to: {save_path}")
    ###########################################################
    #######################mesher#############################
    mesh_root = getattr(cg, "mesh_root", None) 
    if not mesh_root:
        mesh_root = os.path.join(cg.output_root, "meshes")

    os.makedirs(mesh_root, exist_ok=True)

    run_name = f"{cg.yaml_name}_{startframe:06d}-{endframe-1:06d}"
    full_mesh_prefix = os.path.join(mesh_root, f"{run_name}_full")


    mesh = mesher.create_mesh(
        voxel_field, mlp, full_mesh_prefix,
        gmax_x, gmin_x, gmax_y, gmin_y, gmax_z, gmin_z,
        cg.mesh_resolution, 
        scale=1
    )
    mesh.compute_vertex_normals()
    print(f"[INFO] Full mesh saved: {full_mesh_prefix}.ply")

    ###########################################################

if __name__ == "__main__":

    cg = Config()
    if len(sys.argv) > 1:
        cg.load(sys.argv[1])
        cfg_path = sys.argv[1]
        cg.yaml_name = Path(cfg_path).stem
    else:
        sys.exit("No config file.")

    dataloder = dataLoader(points_floder = cg.data_path,
                           pose_path = cg.pose_path, 
                           calib_path = cg.calib_path,
                           device = cg.device)
    
    if (cg.end_frame - cg.begin_frame) < 0:
        sys.exit("end frame should be larger than begin frame")
    
    sampler = dataSampler(truncated_area = cg.truncated_sample_range_m,
                          truncated_num = cg.truncated_sample_num,
                          occupied_area = cg.occupied_sample_range_m,
                          occupied_num = cg.occupied_sample_num,
                          free_space_num = cg.free_sample_num,
                          device = cg.device,
                          dtype = cg.dtype)
            
    voxel_field = NeuralHashVoxel(feature_dim = cg.feature_dim, 
                            feature_std = cg.feature_std, 
                            leaf_voxel_size = cg.leaf_voxel_size, 
                            voxel_level_num = cg.voxel_level_num,
                            up_scale_factor = cg.scale_up_factor, 
                            device = cg.device,
                            dtype = cg.dtype,
                            buffer_size = int(cg.hash_buffer_size))
    
    mlp = decoder.mlp(input_dim = cg.feature_dim,
                      layers_num = cg.mlp_level,
                      hidden_dim = cg.mlp_hidden_dim,
                      output_dim = 1,
                      with_basis = True,
                      device = cg.device)

    static_mapping(cg, voxel_field, mlp, dataloder, sampler, cg.begin_frame, cg.end_frame)




