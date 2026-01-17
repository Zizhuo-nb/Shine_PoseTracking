## Environment Setup (Conda)

### 1) Install Conda
Install **Miniconda** (or Mambaforge/Anaconda) on your system.
Anaconda3 : https://www.anaconda.com/download

### 2) Create the environment
From the repository root:

```bash
conda env create -f environment.yml
```

### 3) Activate the environment

Use the environment name defined in environment.yml (example: shine_plus):
```bash
conda activate shine_plus
```
## Run

1) Edit config: `config/zappa/zappa.yaml`

- `setting.data_path: "[FILL]"`
- `setting.pose_path: "[FILL]"`
- `setting.output_root: "./results"`
- `setting.begin_frame: [FILL]`
- `setting.end_frame: [FILL]`

2) Run mapping, then pose tracking:

```bash
python mapping.py ./config/kitti/kitti.yaml
python tracker.py ./config/kitti/kitti.yaml
```
