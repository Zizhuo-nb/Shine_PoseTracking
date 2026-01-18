import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_kitti_poses_txt(path: str) -> np.ndarray:
    """
    KITTI pose txt: each line has 12 floats (3x4) in row-major:
    r11 r12 r13 tx  r21 r22 r23 ty  r31 r32 r33 tz
    Returns: (N, 3) translations [x, y, z]
    """
    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data[None, :]

    if data.shape[1] < 12:
        raise ValueError(f"Expected >=12 columns, got {data.shape[1]}")

    # If there are extra columns (e.g., index/timestamp), take the last 12 as pose.
    pose12 = data[:, -12:]
    tx = pose12[:, 3]
    ty = pose12[:, 7]
    tz = pose12[:, 11]
    return np.stack([tx, ty, tz], axis=1)

def draw_txt(path):
  
    t = load_kitti_poses_txt(path)
    x, y = t[:, 0], t[:, 1]

    plt.figure()
    plt.plot(x, y)
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("XY Trajectory")

    plt.grid(True)
    plt.show()


