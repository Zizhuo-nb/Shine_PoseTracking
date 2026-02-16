import yaml
import os
import torch

class Config:
    def __init__(self):

        # Default values
        self.incremental: bool = False
        self.data_path: str = ""  # input point cloud folder
        self.pose_path: str = ""  # input pose file
        self.calib_path: str = ""  # input calib file (to sensor frame)
        self.output_root: str = ""  # output root folder
        self.enable_deskew: bool = False
        self.begin_frame: int = 0  # begin from this frame
        self.end_frame: int = 0  # end at this frame
        self.device: str = "cuda"  # use "cuda" or "cpu"
        self.gpu_id: str = "0"  # used GPU id
        self.dtype = torch.float32 # default torch tensor data type

        #preprocessing
        self.min_z: float = -5.0  # filter for z coordinates (unit: m)
        self.radius: float = 40.0

        # neural voxel hash
        self.leaf_voxel_size: float = 0.3 # we use the voxel hashing structure to maintain the neural points, the voxel size is set as this value
        self.voxel_level_num : int = 2
        self.scale_up_factor : float = 1.5
        self.hash_buffer_size: int = int(1e7)

        self.feature_dim: int = 8  # length of the feature for each grid feature
        self.feature_std: float = 0.0  # grid feature initialization standard deviation

        # decoder
        self.mlp_level: int = 2
        self.mlp_hidden_dim: int = 64

        # sampler
        self.truncated_sample_range_m: float = 0.5
        self.truncated_sample_num: int = 3
        self.occupied_sample_range_m: float = 0.45
        self.occupied_sample_num: int = 2
        self.free_sample_num: int = 2

        # loss
        self.ekional_lamda: float = 0.02
        self.free_space_lamda: float = 0.2

        # mapping
        self.sliding_window_size: int = 30
        self.iterion_num: int = 20
        self.batch_size: int = 16384
        self.learning_rate: float = 0.01

        # meshing
        self.mesh_resolution: float = 0.2



    def load(self, config_file):
        config_args = yaml.safe_load(open(os.path.abspath(config_file)))

        # setting
        self.incremental = config_args["setting"].get("incremental",False)
        self.enable_deskew = config_args["setting"].get("enable_deskew",False)
        self.data_path = config_args["setting"]["data_path"] 
        self.pose_path = config_args["setting"]["pose_path"]
        self.calib_path = config_args["setting"]["calib_path"]
        self.output_root = config_args["setting"]["output_root"]  
        self.begin_frame = config_args["setting"]["begin_frame"]
        self.end_frame = config_args["setting"]["end_frame"]
        self.device = config_args["setting"]["device"]
        self.gpu_id = config_args["setting"]["gpu_id"]
        self.hash_buffer_size = config_args["setting"]["hash_buffer_size"]

        # preprocessing
        self.min_z = config_args["preprocessing"]["min_z"]
        self.radius = config_args["preprocessing"]["radius"]
  
        # neuralvoxel 
        self.leaf_voxel_size = config_args["neuralvoxel"]["leaf_voxel_size"]
        self.voxel_level_num = config_args["neuralvoxel"]["voxel_level_num"]
        self.scale_up_factor = config_args["neuralvoxel"]["scale_up_factor"]
        self.feature_dim = config_args["neuralvoxel"]["feature_dim"]
        self.feature_std = config_args["neuralvoxel"]["feature_std"]

        # decoder
        self.mlp_level = config_args["decoder"]["mlp_level"]
        self.mlp_hidden_dim = config_args["decoder"]["mlp_hidden_dim"]

        # sampler
        self.truncated_sample_range_m = config_args["sampler"]["truncated_sample_range_m"]
        self.truncated_sample_num = config_args["sampler"]["truncated_sample_num"]
        self.occupied_sample_range_m = config_args["sampler"]["occupied_sample_range_m"]
        self.occupied_sample_num = config_args["sampler"]["occupied_sample_num"]
        self.free_sample_num = config_args["sampler"]["free_sample_num"]       

        # mapping
        self.sliding_window_size = config_args["mapping"]["sliding_window_size"]  
        self.iterion_num = config_args["mapping"]["iterion_num"] 
        self.batch_size = config_args["mapping"]["batch_size"] 
        self.learning_rate = config_args["mapping"]["learning_rate"]

        # meshing
        self.mesh_resolution = config_args["meshing"]["mesh_resolution"]