# pose_tracking_utils/deskew.py
import torch

def _so3_log_torch(R: torch.Tensor) -> torch.Tensor:
    # R: (3,3)
    tr = torch.trace(R)
    c = torch.clamp((tr - 1.0) * 0.5, -1.0, 1.0)
    th = torch.acos(c)
    if float(th) < 1e-12:
        return torch.zeros(3, device=R.device, dtype=R.dtype)
    W = (R - R.t()) / (2.0 * torch.sin(th))
    w = torch.stack([W[2,1], W[0,2], W[1,0]])
    return th * w

def _so3_exp_batch(w: torch.Tensor) -> torch.Tensor:
    """
    w: (N,3) rotation vectors
    return R: (N,3,3)
    """
    device, dtype = w.device, w.dtype
    th = torch.linalg.norm(w, dim=-1)  # (N,)
    R = torch.eye(3, device=device, dtype=dtype).expand(w.shape[0], 3, 3).clone()

    small = th < 1e-12
    if small.all():
        return R

    k = w / th.clamp_min(1e-12).unsqueeze(-1)  # (N,3)
    kx, ky, kz = k[:,0], k[:,1], k[:,2]

    K = torch.zeros((w.shape[0], 3, 3), device=device, dtype=dtype)
    K[:,0,1] = -kz; K[:,0,2] =  ky
    K[:,1,0] =  kz; K[:,1,2] = -kx
    K[:,2,0] = -ky; K[:,2,1] =  kx

    sinth = torch.sin(th).view(-1,1,1)
    costh = torch.cos(th).view(-1,1,1)

    R = torch.eye(3, device=device, dtype=dtype).view(1,3,3) + sinth * K + (1.0 - costh) * (K @ K)
    # for tiny angles, keep identity (numerical)
    R[small] = torch.eye(3, device=device, dtype=dtype)
    return R

@torch.no_grad()
def kiss_deskew(xyz: torch.Tensor,
                times_abs: torch.Tensor,
                T_t2: torch.Tensor,
                T_t1: torch.Tensor,
                dt: float = 0.1,
                kf_state: dict = None) -> torch.Tensor:

    if (T_t2 is None) or (T_t1 is None):
        return xyz
    if dt <= 1e-9:
        return xyz

    R2 = T_t2[:3,:3]
    t2 = T_t2[:3,3]
    R1 = T_t1[:3,:3]
    t1 = T_t1[:3,3]

    v = (R2.t() @ (t1 - t2)) / dt                    # (3,)
    w = _so3_log_torch(R2.t() @ R1) / dt              # (3,)
    
    if kf_state is not None:
        z = torch.cat([v, w], dim=0).to(device=xyz.device, dtype=torch.float64)
        x_f = kf_velomega_step_torch(kf_state, z)      # (6,)
        v = x_f[:3].to(dtype=xyz.dtype)
        w = x_f[3:].to(dtype=xyz.dtype)
    t0 = times_abs.min()
    t1 = times_abs.max()
    alpha = 0
    t_ref = (1.0 - alpha) * t0 + alpha * t1

    s = times_abs - t_ref
                    # (N,)
    s_omega = s.unsqueeze(-1) * w.unsqueeze(0)             # (N,3)

    R = _so3_exp_batch(s_omega)                            # (N,3,3)
    xyz_rot = torch.bmm(R, xyz.unsqueeze(-1)).squeeze(-1)
    xyz_out = xyz_rot + s.unsqueeze(-1) * v.unsqueeze(0)
    return xyz_out
#==============================================================
@torch.no_grad()
def kf_velomega_step_torch(state: dict,
                           z: torch.Tensor,
                           q_v: float = 0.15, q_w: float = 0.1,
                           r_v: float = 0.25, r_w: float = 0.15):

    z = z.reshape(6).to(dtype=torch.float64)
    dev = z.device

    if ("x" not in state) or ("P" not in state):
        state["x"] = torch.zeros(6, device=dev, dtype=torch.float64)
        state["P"] = torch.eye(6, device=dev, dtype=torch.float64) * 10.0

    x = state["x"]
    P = state["P"]

    Q = torch.diag(torch.tensor([q_v*q_v]*3 + [q_w*q_w]*3, device=dev, dtype=torch.float64))
    R = torch.diag(torch.tensor([r_v*r_v]*3 + [r_w*r_w]*3, device=dev, dtype=torch.float64))


    # Predict (F=I)
    xm = x #prior
    Pm = P + Q # predicted covariance
    

    # Update (H=I)
    y = z - xm
    S = Pm + R
    K = Pm @ torch.linalg.inv(S)
    x = xm + K @ y
    P = (torch.eye(6, device=dev, dtype=torch.float64) - K) @ Pm

    state["x"], state["P"] = x, P
    return x
