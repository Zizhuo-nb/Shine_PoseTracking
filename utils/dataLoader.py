import sys
import os

import torch
import open3d as o3d
import numpy as np
from numpy.linalg import inv
from natsort import natsorted


def read_calib_file(filename):
        """ 
            read calibration file (with the kitti format)
            returns -> dict calibration matrices as 4*4 numpy arrays
        """
        calib = {}
        calib_file = open(filename)
        key_num = 0

        for line in calib_file:
            # print(line)
            key, content = line.strip().split(":")
            values = [float(v) for v in content.strip().split()]
            pose = np.zeros((4,4))
            
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            calib[key] = pose

        calib_file.close()
        return calib

def read_poses_file(filename, calibration):
        """ 
            read pose file (with the kitti format)
        """
        pose_file = open(filename)

        poses = []

        Tr = calibration["Tr"]
        Tr_inv = inv(Tr)

        for line in pose_file:
            values = [float(v) for v in line.strip().split()]

            pose = np.zeros((4, 4))
            pose[0, 0:4] = values[0:4]
            pose[1, 0:4] = values[4:8]
            pose[2, 0:4] = values[8:12]
            pose[3, 3] = 1.0

            poses.append(np.matmul(Tr_inv, np.matmul(pose, Tr))) # lidar pose in world frame

        pose_file.close()
        return poses


class dataLoader():

    def __init__(self,
        points_floder,
        pose_path,
        calib_path,
        device
    ):
        self.points_floder = points_floder
        self.points_path_list = os.listdir(points_floder)
        
        # self.points_path_list.sort(key=lambda x:int((x.split('.')[0])))
        self.points_path_list = natsorted(self.points_path_list)

        if calib_path != "":
            calib = read_calib_file(calib_path)
        else:
             calib = {"Tr": np.eye(4)}
        self.poses = read_poses_file(pose_path, calib)

        self.device = device
 
    def frame_raw(self, frame_id) -> torch.tensor:
        path = self.points_floder + self.points_path_list[frame_id]
        np_points,pcd,times = self.read_point_cloud(path,frame_id)
        torch_points = torch.from_numpy(np_points).to(self.device)
        if times is None:
            times = None
        else:
            times = torch.from_numpy(times).to(self.device)
        return torch_points,pcd, times
    
    def frame_transfered(self, frame_id) -> torch.tensor:
        points,_,_ = self.frame_raw(frame_id)

        pose = torch.tensor(self.poses[frame_id], device=self.device)
        allones = torch.ones(points.shape[0],1, device=self.device)
        points_homo = torch.cat((points,allones),1).double()
        points_trans = (torch.mm(pose,points_homo.T).T)[:,0:-1]

        return points_trans
    
    def translation(self, frame_id) -> torch.tensor:
        pose = torch.tensor(self.poses[frame_id], device=self.device)
        current_translation = pose[0:-1,-1]
        return current_translation
    

    def read_point_cloud(self, filename: str, frame_id: int = None) -> np.ndarray:
        # read point cloud from either (*.ply, *.pcd) or (kitti *.bin) format
        if ".bin" in filename:
            # we also read the intensity channel here
            points = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))
            points = points[:,0:3]
        elif ".ply" in filename or ".pcd" in filename:
            pc_load = o3d.io.read_point_cloud(filename)

            pcd = pc_load.voxel_down_sample(0.2)
            points = np.asarray(pc_load.points)
            pcd = np.asarray(pcd.points)
            times = load_ply_time_only_fast(filename)
        else:
            sys.exit("The format of the imported point cloud is wrong (support only *pcd, *ply and *bin)")

        return points,pcd,times


#=============================================

def load_ply_time_only_fast(filename: str) -> np.ndarray:
    """
    For YOUR confirmed PLY layout:
      binary_little_endian
      vertex: float x, float y, float z, double time
    Returns:
      times: (N,) float64
    """
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("time", "<f8")])
    with open(filename, "rb") as f:
        # skip header
        while True:
            line = f.readline()
            if not line:
                raise RuntimeError("PLY header ended unexpectedly")
            if line.strip() == b"end_header":
                break
        data = np.fromfile(f, dtype=dt)
    return data["time"].astype(np.float64)

