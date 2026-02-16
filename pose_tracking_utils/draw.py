import sys
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

# def draw_txt(path: str):
#     t = load_kitti_poses_txt(path)
#     x, y = t[:, 0], t[:, 1]

#     plt.figure()
#     plt.plot(x, y)
#     plt.xlabel("x (m)")
#     plt.ylabel("y (m)")
#     plt.title("XY Trajectory")
#     plt.grid(True)
#     plt.show()
def draw_txt(path: str, equal_axis: bool = False): ####3d version
    t = load_kitti_poses_txt(path)
    x, y, z = t[:, 0], t[:, 1], t[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("3D Trajectory")
    ax.grid(True)

    # Optional: make axes have equal scale (helps avoid distortion).
    if equal_axis:
        # Set equal aspect in 3D by matching ranges.
        max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
        mid_x = (x.max() + x.min()) * 0.5
        mid_y = (y.max() + y.min()) * 0.5
        mid_z = (z.max() + z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python draw_traj.py <poses.txt>")
        sys.exit(1)
    draw_txt(sys.argv[1])



# import sys
# import os
# import numpy as np
# import matplotlib.pyplot as plt


# def load_kitti_poses_txt(path: str) -> np.ndarray:
#     """
#     KITTI pose txt: each line has 12 floats (3x4) in row-major:
#     r11 r12 r13 tx  r21 r22 r23 ty  r31 r32 r33 tz
#     Returns: (N, 3) translations [x, y, z]
#     """
#     data = np.loadtxt(path)
#     if data.ndim == 1:
#         data = data[None, :]

#     if data.shape[1] < 12:
#         raise ValueError(f"[{path}] Expected >=12 columns, got {data.shape[1]}")

#     # If there are extra columns (e.g., index/timestamp), take the last 12 as pose.
#     pose12 = data[:, -12:]
#     tx = pose12[:, 3]
#     ty = pose12[:, 7]
#     tz = pose12[:, 11]
#     return np.stack([tx, ty, tz], axis=1)


# def set_axes_equal_3d(ax, xyz_all: np.ndarray):
#     """
#     Make 3D axes have equal scale based on ALL trajectories points.
#     xyz_all: (M, 3)
#     """
#     x, y, z = xyz_all[:, 0], xyz_all[:, 1], xyz_all[:, 2]
#     x_min, x_max = np.min(x), np.max(x)
#     y_min, y_max = np.min(y), np.max(y)
#     z_min, z_max = np.min(z), np.max(z)

#     max_range = max(x_max - x_min, y_max - y_min, z_max - z_min) / 2.0
#     mid_x = (x_max + x_min) * 0.5
#     mid_y = (y_max + y_min) * 0.5
#     mid_z = (z_max + z_min) * 0.5

#     ax.set_xlim(mid_x - max_range, mid_x + max_range)
#     ax.set_ylim(mid_y - max_range, mid_y + max_range)
#     ax.set_zlim(mid_z - max_range, mid_z + max_range)


# def draw_txt_multi(
#     paths,
#     equal_axis: bool = False,
#     use_linestyle: bool = False,
#     dim: int = 3,  # 2 or 3
# ):
#     # Load all trajectories
#     trajs = []
#     labels = []
#     for p in paths:
#         t = load_kitti_poses_txt(p)
#         trajs.append(t)
#         labels.append(os.path.basename(p))

#     if dim == 2:
#         plt.figure()
#         linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

#         for i, (t, name) in enumerate(zip(trajs, labels)):
#             x, y = t[:, 0], t[:, 1]
#             ls = linestyles[i % len(linestyles)] if use_linestyle else "-"
#             plt.plot(x, y, linestyle=ls, label=name)

#         plt.xlabel("x (m)")
#         plt.ylabel("y (m)")
#         plt.title("XY Trajectories")
#         plt.grid(True)
#         plt.legend()
#         plt.axis("equal" if equal_axis else "auto")
#         plt.show()
#         return

#     # 3D
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")

#     linestyles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
#     xyz_all = []

#     for i, (t, name) in enumerate(zip(trajs, labels)):
#         x, y, z = t[:, 0], t[:, 1], t[:, 2]
#         ls = linestyles[i % len(linestyles)] if use_linestyle else "-"
#         ax.plot(x, y, z, linestyle=ls, label=name)
#         xyz_all.append(t)

#     ax.set_xlabel("x (m)")
#     ax.set_ylabel("y (m)")
#     ax.set_zlabel("z (m)")
#     ax.set_title("3D Trajectories")
#     ax.grid(True)
#     ax.legend()

#     if equal_axis:
#         xyz_all = np.vstack(xyz_all)
#         set_axes_equal_3d(ax, xyz_all)

#     plt.show()


# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python draw_traj.py <poses1.txt> [poses2.txt ...] [--2d] [--equal] [--ls]")
#         sys.exit(1)

#     # Simple arg parsing (no extra deps)
#     args = sys.argv[1:]
#     dim = 3
#     equal_axis = False
#     use_linestyle = False

#     if "--2d" in args:
#         dim = 2
#         args.remove("--2d")
#     if "--equal" in args:
#         equal_axis = True
#         args.remove("--equal")
#     if "--ls" in args:
#         use_linestyle = True
#         args.remove("--ls")

#     paths = args
#     if len(paths) == 0:
#         print("No pose files provided.")
#         sys.exit(1)

#     draw_txt_multi(paths, equal_axis=equal_axis, use_linestyle=use_linestyle, dim=dim)




# '''
# 3D 多轨迹（默认不同颜色）
# python draw_traj.py a.txt b.txt c.txt

# 3D 多轨迹 + 轴等比例
# python draw_traj.py a.txt b.txt --equal

# 3D 多轨迹 + 颜色 + 线型一起区分
# python draw_traj.py a.txt b.txt c.txt --ls

# 2D XY 版本
# python draw_traj.py a.txt b.txt --2d --equal
# '''