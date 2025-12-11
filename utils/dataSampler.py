import math
import random
from timeit import repeat

import numpy as np

import open3d as o3d

import torch

class dataSampler():

    def __init__(self,       
            truncated_area,
            truncated_num,    
            occupied_area,
            occupied_num,
            free_space_num,
            device,
            dtype
    ):
       self.truncated_area = truncated_area
       self.truncated_num = truncated_num
       self.occupied_area = occupied_area
       self.occupied_num = occupied_num
       self.free_space_num = free_space_num
       self.device = device
       self.dtype = dtype

    def ray_sample(self, 
                   points,
                   translation):
        points = points.to(self.device)
        shift_points = points - translation
        distances = torch.norm(shift_points, p=2, dim=1, keepdim=True)
        valid_idxs = (distances>1e-8).reshape(-1)
        distances = distances[valid_idxs]
        shift_points = shift_points[valid_idxs]
        translation = translation[valid_idxs]

        truncated_samples = torch.rand(shift_points.shape[0]*self.truncated_num, 1, device=self.device)*self.truncated_area
        occupied_samples = -torch.rand(shift_points.shape[0]*self.occupied_num, 1, device=self.device)*self.occupied_area
        
        repeated_truncated_distances = distances.repeat(1, self.truncated_num).reshape(-1,1)
        repeated_occupied_distances = distances.repeat(1, self.occupied_num).reshape(-1,1)
        repeated_free_distances = distances.repeat(1, self.free_space_num).reshape(-1,1)

        truncated_ratios = 1.0 - truncated_samples/repeated_truncated_distances
        occupied_ratios = 1.0 - occupied_samples/repeated_occupied_distances
        repeated_free_ratios = 1 - self.truncated_area/(distances.repeat(1,self.free_space_num).reshape(-1,1))
        free_space_ratios = self.area_uniform_sample(shift_points.shape[0]*self.free_space_num)*repeated_free_ratios

        truncated_ratios = truncated_ratios.reshape(shift_points.shape[0],-1)
        occupied_ratios = occupied_ratios.reshape(shift_points.shape[0],-1)

        truncated_psdf = truncated_samples.reshape(shift_points.shape[0],-1)
        occupied_psdf = occupied_samples.reshape(shift_points.shape[0],-1)
        surface_psdf = torch.cat((truncated_psdf, occupied_psdf),1).reshape(-1,1)

        surface_ratios = torch.cat((truncated_ratios, occupied_ratios),1).reshape(-1,1)
        surface_r_points = shift_points.repeat(1, self.truncated_num+self.occupied_num).reshape(-1,3)
        surface_r_translations = translation.repeat(1, self.truncated_num+self.occupied_num).reshape(-1,3)
        surface_sample_points = surface_r_points*surface_ratios + surface_r_translations

        free_r_points = shift_points.repeat(1, self.free_space_num).reshape(-1,3)
        free_r_translations = translation.repeat(1, self.free_space_num).reshape(-1,3)
        free_sample_points = free_r_points*free_space_ratios + free_r_translations
        free_psdf = repeated_free_distances*(1-free_space_ratios)

        return surface_sample_points, surface_psdf.squeeze(1), free_sample_points, free_psdf.squeeze(1)
    
    def area_uniform_sample(self, n):
        a = torch.sqrt(torch.rand(n, 1, device=self.device))
        b = torch.rand(n, 1, device=self.device)
        p_x = 0*(1-a) + 1*(a * (1-b)) + 1*(a*b)

        return p_x