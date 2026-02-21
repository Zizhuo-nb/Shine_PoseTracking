
## Environment Setup (Conda)

### 1) Install Conda
Install **Miniconda** (or Mambaforge/Anaconda) on your system.
Anaconda3 : https://www.anaconda.com/download

### 2) Create the environment
From the repository root (.\Shine_PoseTracking\):

```bash
conda env create -f environment.yml
```

### 3) Activate the environment

Use the environment name defined in environment.yml (example: shine_posetracking):
```bash
conda activate shine_posetracking
```
## Run

1) Edit config: `config/zappa/zappa.yaml`

- `setting.data_path: "[FILL]"`
- `setting.pose_path: "[FILL]"`
- `setting.output_root: "../results"` (by default)
- `setting.map_ply: "[FILL]"` (from global mode)
- `setting.begin_frame: [FILL]`
- `setting.end_frame: [FILL]`

2)For Localization mode (with initial guess under map frame), run mapping.py, then tracker.py:

```bash
python mapping.py <path_to_yaml>
python tracker.py --cfg <path_to_yaml> --ckpt <path_to_ckpt>
```
after mapping, you will get .pth file and .ply file (.ply file is for "map_ply:" in yaml, set the path by your self)
For tracker.py, if enable 'enable_vis' in yaml, it'll show the reconstruction and pointCloud(using estimated pose) for each frame

3)For Incremental mode, run mapping.py:

```bash
python mapping.py <path_to_yaml>
```
"enable_deskew" in yaml is optional

## Results
### 1) Ground Truth Trajectory
<img width="640" height="480" alt="Figure_gt" src="https://github.com/user-attachments/assets/923707cf-44c6-448c-a101-8325044248aa" />


### 2) Tracked Trajectory
<img width="640" height="480" alt="Figure_tracked" src="https://github.com/user-attachments/assets/78c0cb4f-d731-4009-87b1-1cb5f68322f8" />

## Data
zhang, . zizhuo ., wu, . shuyi ., & Trekel, N. (2026). Implicit Neural Mapping Test Dataset for Pose Tracking (Master Project) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.18700647

## References

[1] X. Zhong, Y. Pan, J. Behley, and C. Stachniss, *SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations*, ICRA 2023.  

[2] L. Wiesmann, T. Guadagnino, I. Vizzo, N. Zimmerman, Y. Pan, H. Kuang, J. Behley, and C. Stachniss, *LocNDF: Neural Distance Field Mapping for Robot Localization*, IEEE Robotics and Automation Letters (RA-L), 2023.  

