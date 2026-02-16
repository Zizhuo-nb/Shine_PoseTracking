from typing import List
import os
import getpass
from torch.optim.optimizer import Optimizer
import torch
import torch.nn as nn

import numpy as np
import open3d as o3d
import matplotlib.cm as cm


# set up weight and bias
def setup_wandb():
    username = getpass.getuser()
    print(username)
    wandb_key_path =  username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):")
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system("export WANDB_API_KEY=$(cat \"" + wandb_key_path + "\")")

def stepwise_learning_rate_decay(optimizer: Optimizer, learning_rate: float, iteration_number: int,
                                 steps: List, reduce: float = 0.1) -> float:
    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce base learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] *= reduce

    return learning_rate

def num_model_weights(model: nn.Module) -> int:
    num_weights = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    return num_weights

def print_model_summary(model: nn.Module):
    for child in model.children():
        print(child)

def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

def save_checkpoint(feature_octree, decoder_model, optimizer, run_path, checkpoint_name, epoch):
    torch.save({
            'epoch': epoch,
            'feature_octree': feature_octree.state_dict(),
            'decoder': decoder_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(run_path, f"{checkpoint_name}.pth"))


def voxel_down_sample_torch(points: torch.tensor, voxel_size: float):
    """
        voxel based downsampling. Returns the indices of the points which are closest to the voxel centers. 
    Args:
        points (torch.Tensor): [N,3] point coordinates
        voxel_size (float): grid resolution

    Returns:
        indices (torch.Tensor): [M] indices of the original point cloud, downsampled point cloud would be `points[indices]`  

    Reference: Louis Wiesmann
    """
    _quantization = 1000

    offset = torch.floor(points.min(dim=0)[0]/voxel_size).long()
    grid = torch.floor(points / voxel_size)
    center = (grid + 0.5) * voxel_size
    dist = ((points - center) ** 2).sum(dim=1)**0.5
    dist = dist / dist.max() * (_quantization - 1) # for speed up

    grid = grid.long() - offset
    v_size = grid.max().ceil()
    grid_idx = grid[:, 0] + grid[:, 1] * v_size + grid[:, 2] * v_size * v_size

    unique, inverse = torch.unique(grid_idx, return_inverse=True)
    idx_d = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
       
    offset = 10**len(str(idx_d.max().item()))

    idx_d = idx_d + dist.long() * offset
    idx = torch.empty(unique.shape, dtype=inverse.dtype,
                      device=inverse.device).scatter_reduce_(dim=0, index=inverse, src=idx_d, reduce="amin", include_self=False)
    idx = idx % offset

    return idx


def sdf_slice(field, mlp, t, max_x, min_x, max_y, min_y, max_z, min_z, max_range_sdf, horizontal=False):

    if horizontal:
        select_y = (max_y + min_y)/2.0
        resolution = 0.05

        x = torch.arange(round((max_x - min_x)/resolution) + 1, dtype=torch.long)
        z = torch.arange(round((max_z - min_z)/resolution) + 1, dtype=torch.long)

        sample_x, sample_z = torch.meshgrid(x,z)

        sample_x_f = (sample_x*resolution + min_x).flatten()
        sample_z_f = (sample_z*resolution + min_z).flatten()

        sample_y_f = torch.ones_like(sample_z_f) * select_y

        coord_xyz = torch.stack((sample_x_f, sample_y_f, sample_z_f)).float().cuda()
    else:
        select_z = max_z - 2.5
        resolution = 0.1
        x = torch.arange(round((max_x - min_x)/resolution) + 1, dtype=torch.long)
        y = torch.arange(round((max_y - min_y)/resolution) + 1, dtype=torch.long)

        sample_x, sample_y = torch.meshgrid(x, y)

        coord_xy = torch.stack((sample_x.flatten(), sample_y.flatten())).float()

        coord_xy *= resolution
        coord_xy[0,:] += min_x
        coord_xy[1,:] += min_y
        sample_z = torch.ones(1, coord_xy.shape[1]) * select_z
        coord_xyz = torch.cat([coord_xy,sample_z],0).cuda()

    coord_xyz = coord_xyz.T

    feature_vectors = field.get_features(coord_xyz.contiguous())
    test_t = torch.ones(coord_xyz.shape[0],1,device='cuda')*t
    _, sdf, _ = mlp(feature_vectors.float(),test_t.long())

    min_sdf = -max_range_sdf
    max_sdf = max_range_sdf

    sdf_cpu = sdf.detach().cpu().numpy()
    sdf_pred_show = np.clip((sdf_cpu - min_sdf) / (max_sdf - min_sdf), 0., 1.)

    color_map = cm.get_cmap('jet')
    colors = color_map(sdf_pred_show)[:, :3].astype(np.float64)

    sdf_map_pc = o3d.geometry.PointCloud()
    sdf_map_pc.points = o3d.utility.Vector3dVector(coord_xyz.detach().cpu().numpy())
    sdf_map_pc.colors = o3d.utility.Vector3dVector(colors)

    return sdf_map_pc

