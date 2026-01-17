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

![Result]([https://private-user-images.githubusercontent.com/208842620/537233868-2716211e-f8b9-4380-9929-6128874ad778.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg2OTE3NTgsIm5iZiI6MTc2ODY5MTQ1OCwicGF0aCI6Ii8yMDg4NDI2MjAvNTM3MjMzODY4LTI3MTYyMTFlLWY4YjktNDM4MC05OTI5LTYxMjg4NzRhZDc3OC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExN1QyMzEwNThaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1mZDNhMGRmMjdkMjgwMTk3OGM1NTEwZWY1Y2FiODlmZjliOWVmNmI2M2ZmN2YxMTlkODZmZTY4ZjI0NGMwZWZmJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9._iEO8WSu9CVlGD9IB_3SRxjeunNRM5VFGyjWgsC5ye4](https://private-user-images.githubusercontent.com/208842620/537235345-a5d6f5df-6db0-46a0-841e-27db2779eeea.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg2OTM0NDMsIm5iZiI6MTc2ODY5MzE0MywicGF0aCI6Ii8yMDg4NDI2MjAvNTM3MjM1MzQ1LWE1ZDZmNWRmLTZkYjAtNDZhMC04NDFlLTI3ZGIyNzc5ZWVlYS5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExN1QyMzM5MDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05MmZhYzI2MTFmNzJiZjk3NDQ2MTNjOTVjZmUyZjBhYTlkMjUxNmJmOTQ4YjEzMWMwM2JkOGMwMmZjYTA2M2M1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.G3c_jHwQ3g43u68H92KX8Xo5Qm3ibgCh-S6xjGy0lR0))

### 2) Tracked Trajectory

![Result]([https://private-user-images.githubusercontent.com/208842620/537233758-ce3070cd-f9c4-4432-a46a-bf0b12c8e24d.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg2OTE5NDQsIm5iZiI6MTc2ODY5MTY0NCwicGF0aCI6Ii8yMDg4NDI2MjAvNTM3MjMzNzU4LWNlMzA3MGNkLWY5YzQtNDQzMi1hNDZhLWJmMGIxMmM4ZTI0ZC5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExN1QyMzE0MDRaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT02MTI4MjMzMGZkOGIyNTEyNmUzMjY0NTFlMzA3OWRlMjkwNGQ4NDU2ZDkzMmJlY2E4Mzc2NmIyNGM4YjhjNzhkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.uVOJ-RV3gerPrZBEwIeRn7DC54hrS0hgcFVe-S7sg9U](https://private-user-images.githubusercontent.com/208842620/537235399-1f2d0c42-6bb2-4c98-a56b-56761eb97fdc.png?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njg2OTM0ODMsIm5iZiI6MTc2ODY5MzE4MywicGF0aCI6Ii8yMDg4NDI2MjAvNTM3MjM1Mzk5LTFmMmQwYzQyLTZiYjItNGM5OC1hNTZiLTU2NzYxZWI5N2ZkYy5wbmc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTE3JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDExN1QyMzM5NDNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT05MzE2YjliYjg5MTVlN2M4ZGQxNDZkMWZhM2NiN2Q0ODU3MzQ1NDMwMDViMzc0NGQ4Nzc2MDljMDYyMDllZDFkJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.RZ9Hxrvm9IHRAM7Mo5vUYoEo9rphDegU-U8wMIH64Zc))


## References

[1] X. Zhong, Y. Pan, J. Behley, and C. Stachniss, *SHINE-Mapping: Large-Scale 3D Mapping Using Sparse Hierarchical Implicit Neural Representations*, ICRA 2023.  

[2] L. Wiesmann, T. Guadagnino, I. Vizzo, N. Zimmerman, Y. Pan, H. Kuang, J. Behley, and C. Stachniss, *LocNDF: Neural Distance Field Mapping for Robot Localization*, IEEE Robotics and Automation Letters (RA-L), 2023.  

