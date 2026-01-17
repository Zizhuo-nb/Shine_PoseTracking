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
## Results
### 1) Ground Truth Trajectory

![Result](https://private-user-images.githubusercontent.com/208842620/537233868-2716211e-f8b9-4380-9929-6128874ad778.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg2OTE3NTgsIm5iZiI6MTc2ODY5MTQ1OCwicGF0aCI6Ii8yMDg4NDI2MjAvNTM3MjMzODY4LTI3MTYyMTFlLWY4YjktNDM4MC05OTI5LTYxMjg4NzRhZDc3OC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExN1QyMzEwNThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mZDNhMGRmMjdkMjgwMTk3OGM1NTEwZWY1Y2FiODlmZjliOWVmNmI2M2ZmN2YxMTlkODZmZTY4ZjI0NGMwZWZmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9._iEO8WSu9CVlGD9IB_3SRxjeunNRM5VFGyjWgsC5ye4)

### 2) Tracked Trajectory

![Result](https://private-user-images.githubusercontent.com/208842620/537233758-ce3070cd-f9c4-4432-a46a-bf0b12c8e24d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg2OTE5NDQsIm5iZiI6MTc2ODY5MTY0NCwicGF0aCI6Ii8yMDg4NDI2MjAvNTM3MjMzNzU4LWNlMzA3MGNkLWY5YzQtNDQzMi1hNDZhLWJmMGIxMmM4ZTI0ZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExN1QyMzE0MDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02MTI4MjMzMGZkOGIyNTEyNmUzMjY0NTFlMzA3OWRlMjkwNGQ4NDU2ZDkzMmJlY2E4Mzc2NmIyNGM4YjhjNzhkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.uVOJ-RV3gerPrZBEwIeRn7DC54hrS0hgcFVe-S7sg9U)
