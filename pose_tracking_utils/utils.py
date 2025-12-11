import torch
import numpy as np
import open3d as o3d
import config as cg
import os
from pathlib import Path

def skew(v):
    S = torch.zeros(3,3, device= v.device, dtype=v.dtype)
    S[0, 1] = -v[2]
    S[0, 2] = v[1]
    S[1, 2] = -v[0]

    S[1, 0] = v[2]
    S[2, 0] = -v[1]
    S[2, 1] = v[0]
    return S


def expmap(axis_angle:torch.Tensor):
    angle = axis_angle.norm()
    axis = axis_angle/angle
    eye = torch.eye(3, device=axis_angle.device,dtype=torch.float64)
    S = skew(axis)
    R = eye + angle.sin()*S + (1-angle.cos())*(S@S)
    return R


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
def read_point_cloud(path, device, dtype = torch.float64):
    pcd = o3d.io.read_point_cloud(path)
    pcd_numpy = np.asarray(pcd.points, dtype=np.float32)
    point_tensor = torch.from_numpy(pcd_numpy).to(device=device,dtype=dtype)
    return point_tensor
def parse_scan(file):
    pcd = o3d.io.read_point_cloud(file)
    return pcd


###########initial guess to transfer first frame###########
def initial_guess(points_lidar: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    R = T[:3, :3].to(dtype=points_lidar.dtype)
    t = T[:3, 3].to(dtype=points_lidar.dtype)
    points_world = points_lidar @ R.T + t
    return points_world



def forward_points_to_network(points, voxel_field, mlp):
    feature_vectors = voxel_field.get_features(points)
    pred = mlp(feature_vectors.float())
    return pred


def process_points(points_world:torch.Tensor, T:torch.tensor, radius, min_z):
    xyz = points_world[..., :3]
    current_translation = T[:3,3].to(xyz.dtype)
    points_distance = torch.norm(xyz - current_translation, p=2, dim=1, keepdim=False)
    valid_mask = points_distance<radius
    valid_mask = valid_mask & (xyz[:, 2] > min_z)

    points_world = points_world[valid_mask]
    return points_world


def create_output_folder(root_dir: str,
                         begin_frame: int,
                         end_frame: int,
                         yaml_name) -> str:
 
    run_name = f"{yaml_name}_{begin_frame:06d}-{end_frame-1:06d}"
    out_dir = os.path.join(root_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

