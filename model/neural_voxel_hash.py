import torch
import torch.nn as nn

import math
import time
import sys
from tqdm import tqdm
import open3d as o3d

import numpy as np

class NeuralHashVoxel(nn.Module):

    def __init__(self, 
                 feature_dim, 
                 feature_std, 
                 leaf_voxel_size, 
                 voxel_level_num,
                 up_scale_factor,
                 device, 
                 dtype = torch.float32,
                 buffer_size: int = int(1e7)) -> None:
        
        super().__init__()
        # feature setting
        self.feature_dim = feature_dim
        self.feature_std = feature_std

        # map structure setting
        self.leaf_voxel_size = leaf_voxel_size
        self.voxel_level_num = voxel_level_num
        self.up_scale_factor = up_scale_factor

        self.dtype = dtype
        self.device = device

        self.buffer_size = buffer_size

        # hash function
        self.primes = torch.tensor(
            [73856093, 19349669, 83492791], dtype=torch.int64, device=self.device)
        # for point to corners
        self.steps = torch.tensor([[0., 0., 0.], [0., 0., 1.], 
                                   [0., 1., 0.], [0., 1., 1.], 
                                   [1., 0., 0.], [1., 0., 1.], 
                                   [1., 1., 0.], [1., 1., 1.]], dtype=self.dtype, device=self.device)
        
        self.features_list = nn.ParameterList([])
        self.feature_indexs_list = []
        # self.corner_list = []
        for l in range(self.voxel_level_num):
            features = nn.Parameter(torch.tensor([],device=self.device))
            feature_indexs = torch.full([buffer_size], -1, dtype=torch.int64, 
                                             device=self.device) # -1 for not valid (occupied)
            self.features_list.append(features)
            self.feature_indexs_list.append(feature_indexs)

            # corner = torch.tensor([], device=self.device, dtype=torch.float32)
            # self.corner_list.append(corner)

        self.to(self.device)

    def update(self, points: torch.Tensor):
        for i in range(self.voxel_level_num):
            current_resolution = self.leaf_voxel_size*(self.up_scale_factor**i)

            corners = self.to_corners(points, current_resolution)
            # remove reptitive coordinates
            offset = corners.min(dim=0,keepdim=True)[0]
            shift_corners = corners - offset
            v_size = shift_corners.max() + 1
            corner_idx = shift_corners[:, 0] + shift_corners[:, 1] * v_size + shift_corners[:, 2] * v_size * v_size
            unique, index, counts = torch.unique(corner_idx, sorted=False ,return_inverse=True, return_counts=True)

            unique_corners = torch.zeros_like(corners[:len(counts), :], device=corners.device)
            index.unsqueeze_(-1)
            unique_corners.scatter_add_(-2, index.expand(corners.shape), corners)
            unique_corners /= counts.unsqueeze(-1)

            # hash function
            keys = (unique_corners.to(self.primes) * self.primes).sum(-1) % self.buffer_size

            update_mask = (self.feature_indexs_list[i][keys] == -1)

            new_feature_count = unique_corners[update_mask].shape[0]

            self.feature_indexs_list[i][keys[update_mask]] = torch.arange(new_feature_count, dtype=self.feature_indexs_list[i].dtype, 
                                                                        device=self.feature_indexs_list[i].device) + self.features_list[i].shape[0]
            
            new_fts = self.feature_std*torch.randn(new_feature_count, self.feature_dim, device=self.device, dtype=self.dtype)
            self.features_list[i] = nn.Parameter(torch.cat((self.features_list[i], new_fts),0))



    def get_features(self, query_points): 
        sum_features = torch.zeros(query_points.shape[0], self.feature_dim, device=self.device, dtype=self.dtype)
        # valid_mask = torch.ones(query_points.shape[0], device=self.device, dtype=bool)
        for i in range(self.voxel_level_num):
           current_resolution = self.leaf_voxel_size*(self.up_scale_factor**i)

           query_corners = self.to_corners(query_points, current_resolution).to(self.primes)
 
           query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size

           hash_index_nx8 = self.feature_indexs_list[i][query_keys].reshape(-1,8)

           featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

           features_index = hash_index_nx8[featured_query_mask].reshape(-1,1).squeeze(1)

           coeffs = self.interpolat(query_points[featured_query_mask], current_resolution)
           
           sum_features[featured_query_mask] += (self.features_list[i][features_index]*coeffs).reshape(-1,8,self.feature_dim).sum(1)

        return sum_features#, valid_mask

    def get_valid_mask(self, query_points):
        n = self.voxel_level_num-1
        current_resolution = self.leaf_voxel_size*(self.up_scale_factor**n)
        
        query_corners = self.to_corners(query_points, current_resolution).to(self.primes)
 
        query_keys = (query_corners * self.primes).sum(-1) % self.buffer_size

        hash_index_nx8 = self.feature_indexs_list[n][query_keys].reshape(-1,8)

        featured_query_mask = (hash_index_nx8.min(dim=1)[0]) > -1

        return featured_query_mask
    
    def interpolat(self, x, resolution):
        coords = x / resolution
        d_coords = coords - torch.floor(coords)
        tx = d_coords[:,0]
        _1_tx = 1-tx
        ty = d_coords[:,1]
        _1_ty = 1-ty
        tz = d_coords[:,2]
        _1_tz = 1-tz
        p0 = _1_tx*_1_ty*_1_tz
        p1 = _1_tx*_1_ty*tz
        p2 = _1_tx*ty*_1_tz
        p3 = _1_tx*ty*tz
        p4 = tx*_1_ty*_1_tz
        p5 = tx*_1_ty*tz
        p6 = tx*ty*_1_tz
        p7 = tx*ty*tz
        p = torch.stack((p0,p1,p2,p3,p4,p5,p6,p7),0).T.reshape(-1,1)
        return p

    def to_corners(self, points: torch.Tensor, resolution):
        origin_corner = torch.floor(points / resolution)
        corners = (origin_corner.repeat(1,8) + self.steps.reshape(1,-1)).reshape(-1,3)
        return corners


