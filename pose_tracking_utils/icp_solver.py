import torch
from .utils import expmap,parse_scan,compute_gradient,forward_points_to_network,process_points
from typing import Optional
from os.path import join
import numpy as np
from model.neural_voxel_hash import NeuralHashVoxel
from model import decoder
import sys
from utils.config import Config
import glob
import os

class Posetracker:
    def __init__(self, 
                 test_folder, 
                 voxel_field, 
                 geo_decoder,
                 device, 
                 start_idx,
                 end_idx,     
                 max_dist=50,
                 GM_k=None, 
                 num_iter=20,
                 init_pose=None,
                 radius=None,
                 min_z=None,
                 use_constant_velocity: bool = False):

        self.radius = radius
        self.min_z = min_z
        self.device = device
        self.num_iter = num_iter
        self.max_dist = max_dist
    
        self.feature = voxel_field.to(device)
        self.geo_decoder = geo_decoder.to(device) 

        self.test_folder = test_folder
        self.GM_k = GM_k
        self.use_constant_velocity = use_constant_velocity

        all_files = sorted(glob.glob(os.path.join(test_folder, "*.ply")))
        if len(all_files) == 0:
            raise RuntimeError(f"No .ply files found in {test_folder}")

        if end_idx is None or end_idx > len(all_files):
            end_idx = len(all_files)
        if start_idx < 0 or start_idx >= end_idx:
            raise ValueError(
                f"Invalid frame range: start_idx={start_idx}, end_idx={end_idx}, "
                f"total={len(all_files)}"
            )

        self.base_idx = start_idx   
        self.file_list = all_files[start_idx:end_idx]

        if len(self.file_list) == 0:
            raise RuntimeError(
                f"No .ply files in range [{start_idx}, {end_idx}) for {test_folder}"
            )

        self.running_idx = 0

        if init_pose is None:
            self.pose = torch.eye(4, device=device, dtype=torch.float64)
        else:
            self.pose = init_pose.to(device=device, dtype=torch.float64)

        self.constant_velocity = torch.eye(4, device=device, dtype=torch.float64)
        self.est_poses = []


    def forward(self, points:torch.Tensor):
        device = points.device
        points = points[..., :3].clone().detach().to(device)
        points.requires_grad_(True)
        sdf = forward_points_to_network(points, self.feature, self.geo_decoder)
        sdf = sdf.reshape(-1,1)
        grads = compute_gradient(sdf, points)
        return points.detach(), sdf.detach() , grads.detach()

    def register_next(self):
        if self.running_idx >= len(self.file_list):
            raise IndexError("No more scans to register.")
        ply_path = self.file_list[self.running_idx]
        scan = parse_scan(ply_path)
        points = np.asarray(scan.points)
        param = next(self.geo_decoder.parameters())
        points = torch.tensor(points, dtype=param.dtype, device=param.device)
        points = points[points.norm(dim=-1)<self.max_dist]
        points = torch.cat([points, torch.ones_like(points[:,:1])], dim=-1)

        if self.use_constant_velocity and len(self.est_poses)>0:
            init_guess = self.constant_velocity @ self.pose
        else:
            init_guess = self.pose

        current_pose = self.registration_scan(points, 
                                              num_iter=self.num_iter,
                                              initial_guess= init_guess)
        if self.use_constant_velocity:
            self.constant_velocity = current_pose @ torch.linalg.inv(
                self.pose) if len(self.est_poses)>0 else torch.eye(4).to(current_pose)

        self.pose = current_pose
        self.est_poses.append(current_pose.detach().cpu().numpy())
        self.running_idx +=1
        return current_pose, scan


    def registration_scan(self, points, num_iter, initial_guess = None):
        if initial_guess is None:
            print("**********************initial_guess is None##################")
            T = torch.eye(4, device=points.device, dtype=torch.float64)
        else:
            T = initial_guess 
        points = points.detach().T
        for _ in range(num_iter):
            points_world = (T.to(dtype=points.dtype) @ points).T
            points_world = process_points(points_world, T, radius = self.radius, min_z= self.min_z)
            DT = self.registration_step(points_world, GM_k=self.GM_k).detach()
            T = DT @ T
            change = torch.acos((torch.trace(DT[:3, :3]) - 1) / 2) + DT[:3, -1].norm()
            if change < 1e-4:
                break
        return T


    def registration_step(self, points: torch.Tensor, GM_k=None):
        if points.shape[0] < 10:
                return torch.eye(4, device=points.device, dtype=torch.float64)

        points, distances, gradients = self.forward(points)
        grad_norm = gradients.norm(dim=-1, keepdim=True)




        eps = 1e-5
        valid = grad_norm.squeeze(-1) > eps
        if valid.sum() < 10:
            print("valid points are too little!!!!!")
            return torch.eye(4, device=points.device, dtype=torch.float64)
        points = points[valid] 
        distances = distances[valid]
        gradients = gradients[valid]
        grad_norm = grad_norm[valid] 
        gradients = gradients/grad_norm
        distances = distances/grad_norm
        T = df_icp(points[..., :3], gradients, distances, GM_k=GM_k)
        return T
#################################################################################################

cg = Config()
if len(sys.argv) > 1:
    cg.load(sys.argv[1])
else:
    sys.exit("No config file.")

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

def df_icp(points, gradients, distances, GM_k = None):
    if GM_k is None:
        w = 1
    else:
        w = GM_k/(GM_k+distances**2)**2
    
    cross = torch.cross(points, gradients)
    J = torch.cat([gradients,cross],-1)
    N = J.T @(w*J)
    g = -(J*w).T@distances
    t = torch.linalg.inv(N.to(dtype=torch.float64)) @ (g.to(dtype=torch.float64)).squeeze()
    T = torch.eye(4, device=points.device, dtype=torch.float64)
    T[:3,:3] = expmap(t[3:])
    T[:3,-1] = t[:3]
    return T