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


def expmap(axis_angle: torch.Tensor):
    theta = axis_angle.norm()
    I = torch.eye(3, device=axis_angle.device, dtype=torch.float64)
    W = skew(axis_angle)

    eps = 1e-8
    if theta < eps:
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
        grad_outputs=torch.ones_like(sdf),  
        create_graph=False,
        retain_graph=False,
        only_inputs=True
    )[0]
    return grads.detach()


########read point cloud and turn it into Tensor################
def parse_scan(file):
    pcd = o3d.io.read_point_cloud(file)
    pcd = pcd.voxel_down_sample(0.2)
    return pcd


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
