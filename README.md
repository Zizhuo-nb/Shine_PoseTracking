# SHINE_PoseTracking

## Environment Setup (Conda)

### 1) Install Conda
Install **Miniconda** (or Mambaforge/Anaconda) on your system.

### 2) Create the environment
From the repository root:

```bash
conda env create -f environment.yml


## Run
change some path in config floder,mapping first, then posetracking
example:

```
python mapping.py ./config/kitti/kitti.yaml
python tracker.py ./config/litti/kitti.yaml
```
