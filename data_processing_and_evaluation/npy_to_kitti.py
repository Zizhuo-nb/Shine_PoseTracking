import argparse
import numpy as np

def q_to_R(qx, qy, qz, qw):
    # quaternion (x,y,z,w) -> rotation matrix
    n = qx*qx + qy*qy + qz*qz + qw*qw
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    xx, yy, zz = qx*qx*s, qy*qy*s, qz*qz*s
    xy, xz, yz = qx*qy*s, qx*qz*s, qy*qz*s
    wx, wy, wz = qw*qx*s, qw*qy*s, qw*qz*s

    R = np.array([
        [1.0 - (yy + zz),       xy - wz,       xz + wy],
        [      xy + wz, 1.0 - (xx + zz),       yz - wx],
        [      xz - wy,       yz + wx, 1.0 - (xx + yy)],
    ], dtype=np.float64)
    return R

def to_kitti_3x4(poses, quat_layout):
    poses = np.asarray(poses)

    # Case A: (N,4,4)
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        return poses[:, :3, :4].astype(np.float64)

    # Case B: (N,3,4)
    if poses.ndim == 3 and poses.shape[1:] == (3, 4):
        return poses.astype(np.float64)

    # Case C: (N,12) already flattened 3x4
    if poses.ndim == 2 and poses.shape[1] == 12:
        return poses.reshape(-1, 3, 4).astype(np.float64)

    # Case D: (N,7) quaternion + translation
    if poses.ndim == 2 and poses.shape[1] == 7:
        out = np.zeros((poses.shape[0], 3, 4), dtype=np.float64)
        for i in range(poses.shape[0]):
            v = poses[i]
            if quat_layout == "t_q":         # [tx,ty,tz,qx,qy,qz,qw]
                tx, ty, tz, qx, qy, qz, qw = map(float, v)
            elif quat_layout == "q_t":       # [qx,qy,qz,qw,tx,ty,tz]
                qx, qy, qz, qw, tx, ty, tz = map(float, v)
            else:
                raise ValueError("quat_layout must be 't_q' or 'q_t'")
            R = q_to_R(qx, qy, qz, qw)
            out[i, :3, :3] = R
            out[i, :3,  3] = [tx, ty, tz]
        return out

    raise ValueError(f"Unsupported pose npy shape: {poses.shape}, ndim={poses.ndim}")

def save_kitti_txt(T_3x4, out_txt):
    # KITTI: each line 12 floats (row-major of 3x4)
    with open(out_txt, "w", encoding="utf-8") as f:
        for i in range(T_3x4.shape[0]):
            row = T_3x4[i].reshape(-1)
            f.write(" ".join(f"{x:.12f}" for x in row) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npy", required=True, help="input pose .npy path")
    ap.add_argument("--out", required=True, help="output kitti pose txt path, e.g. poses/00.txt")
    ap.add_argument("--quat_layout", default="t_q", choices=["t_q", "q_t"],
                    help="only used when input is (N,7)")
    args = ap.parse_args()

    poses = np.load(args.npy)
    T_3x4 = to_kitti_3x4(poses, quat_layout=args.quat_layout)
    save_kitti_txt(T_3x4, args.out)

    # quick checks
    print("[INFO] loaded npy shape:", poses.shape)
    print("[INFO] exported KITTI poses shape:", T_3x4.shape, "->", args.out)

if __name__ == "__main__":
    main()
