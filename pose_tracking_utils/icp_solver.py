import torch
from .utils import expmap,parse_scan,compute_gradient,forward_points_to_network,process_points
import numpy as np
import glob
import os
from natsort import natsorted

class Posetracker:
    def __init__(self, 
                 test_folder, 
                 voxel_field, 
                 geo_decoder,
                 device, 
                 start_idx,
                 max_dist=50,
                 GM_k=None, 
                 num_iter=20,
                 init_pose = None,
                 radius = None,
                 min_z = None,
                 ):
        
        self.last_used_points_world = None   # (M,3) torch, world
        self.last_used_frame_id = None
        self.radius = radius
        self.min_z = min_z
        self.device = device
        self.num_iter = num_iter
        self.max_dist = max_dist
    
        self.feature = voxel_field.to(device)
        self.geo_decoder = geo_decoder.to(device) 
        self.running_idx = start_idx

        self.initpose = init_pose
        self.test_folder = test_folder
        self.file_list = natsorted(glob.glob(os.path.join(test_folder, "*.ply")))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .ply files found in {test_folder}")

        self.GM_k = GM_k
        self.pose = torch.eye(4, device=device, dtype=torch.float64)
        
        if init_pose is None:
            self.pose = torch.eye(4, device=device, dtype=torch.float64)
        else:
            self.pose = init_pose.to(device=device, dtype=torch.float64)

        self.constant_velocity = torch.eye(4, device=device, dtype=torch.float64)
        self.est_poses = []

        pass

    def forward(self, points:torch.Tensor):
        device = points.device
        points = points[..., :3].clone().detach().to(device)
        points.requires_grad_(True)
        sdf = forward_points_to_network(points, self.feature, self.geo_decoder)
        sdf = sdf.reshape(-1,1)
        grads = compute_gradient(sdf, points)


        if self.running_idx == 0:   # 只看第一帧
            print("[DEBUG forward] sdf_mean=", sdf.abs().mean().item(),
                "sdf_max=", sdf.abs().max().item(),
                "grad_mean=", grads.norm(dim=-1).mean().item())
            print(self.initpose)

        return points.detach(), sdf.detach() , grads.detach()
    
        # pred = forward_points_to_network(points, self.feature, self.geo_decoder)  # (N,) logit
        # occ  = torch.sigmoid(pred * self.scalar_factor).reshape(-1, 1)          # (N,1) prob
        # res  = (occ - 0.5)*10                                                        # (N,1) 0-level set at 0.5

        # grads = compute_gradient(res, points)                                   # d(res)/dx

        # if self.running_idx == 0:
        #     print("[DEBUG forward] valid_res_mean=", res.abs().mean().item(),
        #         "res_max=", res.abs().max().item(),
        #         "grad_mean=", grads.norm(dim=-1).mean().item())

        # return points.detach(), res.detach(), grads.detach()



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
        current_pose = self.registration_scan(points, 
                                              num_iter=self.num_iter,
                                              initial_guess=self.constant_velocity@self.pose)
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
        for it in range(num_iter):
            points_world = (T.to(dtype=points.dtype) @ points).T
            points_world = process_points(points_world, T, radius = self.radius, min_z= self.min_z)
            DT = self.registration_step(points_world, GM_k=self.GM_k).detach()
            T = DT @ T
            change = torch.acos((torch.trace(DT[:3, :3]) - 1) / 2) + DT[:3, -1].norm()
            if change < 1e-4:
                # print("#########break###########")
                break
        return T

    def registration_step(self, points: torch.Tensor, GM_k=None):
        points, distances, gradients = self.forward(points)
        hash_valid = self.feature.get_valid_mask(points[...,:3])
        grad_norm = gradients.norm(dim=-1, keepdim=True)
        # eps = 1e-4
        # grad_valid = grad_norm.squeeze(-1) > eps
        # valid = hash_valid & grad_valid
        valid = hash_valid
        if valid.sum() < 10:
            print("valid points are too little!!!!!")
            # return torch.eye(4, device=points.device, dtype=torch.float64)
        points = points[valid] # (M,4)
        distances = distances[valid]  # (M,)
        gradients = gradients[valid] # (M,3)
        grad_norm = grad_norm[valid]  # (M,1)
        # gradients = gradients/(grad_norm+ 1e-12)
        # distances = distances/(grad_norm +1e-12)
        self.last_used_points_world = points.detach()
        self.last_used_frame_id = int(self.running_idx)
        T = df_icp(points[..., :3], gradients, distances, GM_k=GM_k)
        return T

#################################################################################################

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
