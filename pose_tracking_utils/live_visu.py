import os
import copy
import time
import numpy as np
import open3d as o3d

from pose_tracking_utils.utils import sdf_colors_from_points, parse_scan, compute_pose_error,build_sdf_grad_lineset
from pose_tracking_utils.debug_tools import DebugCSVLogger, make_debug_row


def _load_map(map_ply: str):
    mesh = o3d.io.read_triangle_mesh(map_ply)
    if mesh is not None and mesh.has_triangles() and len(mesh.triangles) > 0:
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.65, 0.65, 0.65])
        return mesh

    pcd = o3d.io.read_point_cloud(map_ply)
    if pcd is None or pcd.is_empty():
        raise RuntimeError(f"bad map_ply: {map_ply}")
    pcd.paint_uniform_color([0.65, 0.65, 0.65])
    return pcd


def _append_trk_pose_txt(path: str, fid: int, T: np.ndarray):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    m = T[:3, :].reshape(-1).tolist()
    with open(path, "a", encoding="utf-8") as f:
        f.write(str(int(fid)) + " " + " ".join(f"{x:.8f}" for x in m) + "\n")
from typing import Optional
        
def _pcd_from_solver_points(tracker) -> Optional[o3d.geometry.PointCloud]:

    pts = getattr(tracker, "last_used_points_world", None)
    if pts is None:
        return None
    if hasattr(pts, "detach"):
        pts = pts.detach()
    if pts.numel() == 0:
        return None

    pts_np = pts[:, :3].to("cpu").numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    return pcd


def run_live_viz_step0(
    tracker,
    map_ply: str,
    begin_frame: int,
    end_frame: int,
    gt_poses_np: list,
    gt_poses_torch: list,
    out_pose_path: str,
    out_err_path: str,
    vis_n: int = 1,
    vis_k: int = 1,
    s_max: float = 0.2,
    chunk_size: int = 200000,

    resume_from_file: bool = False,
):
    debug_csv_path = out_err_path.replace(".txt", "_sdf_debug.csv")
    dbg = DebugCSVLogger(debug_csv_path,overwrite=True)

    begin_frame = int(begin_frame)
    end_frame = int(end_frame)
    if end_frame < 0:
        end_frame = len(tracker.file_list) - 1
    end_frame = min(end_frame, len(tracker.file_list) - 1)
    begin_frame = max(0, begin_frame)
    if begin_frame > end_frame:
        begin_frame = end_frame

    map_geom = _load_map(map_ply)
    

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Tracking Viz", width=1280, height=720)
    vis.add_geometry(map_geom)
    grad_ls = o3d.geometry.LineSet()
    vis.add_geometry(grad_ls)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 2.0
    opt.show_coordinate_frame = True
    if hasattr(opt, "light_on"):
        opt.light_on = True
    if hasattr(opt, "mesh_show_back_face"):
        opt.mesh_show_back_face = True

    state = {
        "pose_mode": "gt",          # "gt" | "trk"
        "color_mode": "solid",      # "solid" | "sdf"
        "disp": begin_frame,
        "overlay": [],
        "scan_cache": {},
        "trk_pose_cache": {},      
        "key_last": {},
        "trk_used_pts_cache": {},   # fid -> (M,3) np.float64, world points actually used by solver
        "show_grad": False,
        "grad_step": 50,         
        "grad_max": 5000,        
        "grad_len": 0.5,  
    }

    # ---------- file init ----------
    os.makedirs(os.path.dirname(out_pose_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_err_path) or ".", exist_ok=True)

    if not resume_from_file:
     
        with open(out_pose_path, "w", encoding="utf-8") as f:
            f.write("# fid r11 r12 r13 tx r21 r22 r23 ty r31 r32 r33 tz\n")
        with open(out_err_path, "w", encoding="utf-8") as f:
            f.write("# frame t_err(m) r_err(deg) dx dy dz\n")
        tracker.running_idx = begin_frame

    else:

        pass

    def _debounce(key: str, dt: float = 0.12) -> bool:
        now = time.time()
        last = state["key_last"].get(key, 0.0)
        if now - last < dt:
            return False
        state["key_last"][key] = now
        return True

    def _print_state(tag=""):
        print(
            f"[VIS]{tag} pose={state['pose_mode']} color={state['color_mode']} grad={int(state.get('show_grad', False))} "
            f"disp={state['disp']} trk_next={int(tracker.running_idx)} trk_cached={len(state['trk_pose_cache'])} "
            f"trk_pts_cached={len(state.get('trk_used_pts_cache', {}))}"
        )

    def _get_scan(fid: int) -> o3d.geometry.PointCloud:
        sc = state["scan_cache"].get(fid, None)
        if sc is None:
            sc = parse_scan(tracker.file_list[fid])  # 原始点云，不下采样
            state["scan_cache"][fid] = sc
        return sc

    def _ensure_trk_upto(target_fid: int):
        target_fid = int(target_fid)
        while int(tracker.running_idx) <= target_fid:
            fid = int(tracker.running_idx)
            T_global, _ = tracker.register_next()  # 内部 running_idx += 1
            # ---- cache solver-used points for replay (world coords) ----
            pts_used = getattr(tracker, "last_used_points_world", None)
            if pts_used is not None and hasattr(pts_used, "numel") and pts_used.numel() > 0:
                state["trk_used_pts_cache"][fid] = pts_used.detach().to("cpu").numpy().astype(np.float64)
            else:
                state["trk_used_pts_cache"][fid] = None  # 标记这一帧没拿到

            Tn = T_global.detach().cpu().numpy().astype(np.float64)

            state["trk_pose_cache"][fid] = Tn
            _append_trk_pose_txt(out_pose_path, fid, Tn)

            # err vs GT
            T_gt = gt_poses_torch[fid]
            t_err, r_err, dx, dy, dz = compute_pose_error(T_global, T_gt)
            with open(out_err_path, "a", encoding="utf-8") as f:
                f.write(f"{fid} {t_err:.6f} {r_err:.6f} {dx:.6f} {dy:.6f} {dz:.6f}\n")
            scan = _get_scan(fid)  # 你已有 scan_cache getter
            pts_lidar = np.asarray(scan.points, dtype=np.float64)

            T_gt = gt_poses_np[fid]
     
            T_trk = Tn

            row = make_debug_row(
                fid=fid,
                pts_lidar_xyz=pts_lidar,
                T_gt=T_gt,
                T_trk=T_trk,
                voxel_field=tracker.feature,
                decoder=tracker.geo_decoder,
                max_dist=float(tracker.max_dist),
                min_z=float(tracker.min_z),
                max_probe=20000,
                band_ratio=0.30,  
            )

            dbg.append_row(row)

    def _get_T(fid: int) -> np.ndarray:
        if state["pose_mode"] == "gt":
            return np.asarray(gt_poses_np[fid], dtype=np.float64)

        if fid not in state["trk_pose_cache"]:
            _ensure_trk_upto(fid)
        return state["trk_pose_cache"][fid]

    def _apply_color(pcd: o3d.geometry.PointCloud):
        if state["color_mode"] == "solid":
            pcd.paint_uniform_color([0.0, 1.0, 1.0])
            return
        pts = np.asarray(pcd.points)
        colors = sdf_colors_from_points(
            points_xyz=pts,
            voxel_field=tracker.feature,
            decoder=tracker.geo_decoder,
            s_max=float(s_max),
            chunk_size=int(chunk_size),
            return_sdf=False,
        )
        pcd.colors = o3d.utility.Vector3dVector(colors)

    def _clear_overlay():
        for e in state["overlay"]:
            vis.remove_geometry(e, reset_bounding_box=False)
        state["overlay"] = []

    def _render_frame(fid: int):
        fid = int(np.clip(fid, begin_frame, end_frame))
        state["disp"] = fid

        # ---------- build pcd ----------
        if state["pose_mode"] == "trk":
            _ensure_trk_upto(fid)

            pts_np = state["trk_used_pts_cache"].get(fid, None)
            if pts_np is not None:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(pts_np) 
            else:
                scan_raw = _get_scan(fid)
                pcd = copy.deepcopy(scan_raw)
                pcd.transform(_get_T(fid))
        else:
            scan_raw = _get_scan(fid)
            pcd = copy.deepcopy(scan_raw)
            pcd.transform(_get_T(fid))


        _apply_color(pcd)

        # ---------- draw ----------
        _clear_overlay()
        vis.add_geometry(pcd, reset_bounding_box=False)
        state["overlay"] = [pcd]

        _print_state(" [RENDER]")

        # ----- gradient arrows (immediate) -----
        if state.get("show_grad", False):
            pts_now = np.asarray(pcd.points, dtype=np.float32)
            new_ls = build_sdf_grad_lineset(
                points_w=pts_now,
                voxel_field=tracker.feature,
                decoder=tracker.geo_decoder,
                step=int(state["grad_step"]),
                max_arrows=int(state["grad_max"]),
                arrow_len=float(state["grad_len"]),
            )
            grad_ls.points = new_ls.points
            grad_ls.lines  = new_ls.lines
            grad_ls.colors = new_ls.colors
        else:
            # clear
            grad_ls.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            grad_ls.lines  = o3d.utility.Vector2iVector(np.zeros((0, 2), dtype=np.int32))
            grad_ls.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))

        vis.update_geometry(grad_ls)
        vis.poll_events()
        vis.update_renderer()



    # ---- keys
    def on_toggle_grad(_v):
        if not _debounce("G", 0.15):
            return False
        state["show_grad"] = not state.get("show_grad", False)
        _render_frame(state["disp"]) 
        print(f"[VIS] grad_arrows -> {state['show_grad']}")
        return False

    

    def on_toggle_pose(_v):
        if not _debounce("T", 0.15):
            return False
        state["pose_mode"] = "trk" if state["pose_mode"] == "gt" else "gt"
        _render_frame(state["disp"])
        return False

    def on_toggle_color(_v):
        if not _debounce("V", 0.15):
            return False
        state["color_mode"] = "sdf" if state["color_mode"] == "solid" else "solid"
        _render_frame(state["disp"])
        return False

    def on_back(_v):
        if not _debounce("A", 0.06):
            return False
        if state["disp"] <= begin_frame:
            return False
        _render_frame(state["disp"] - 1)
        return False

    def on_forward(_v):
        if not _debounce("D", 0.06):
            return False
        if state["disp"] >= end_frame:
            return False
        _render_frame(state["disp"] + 1)
        return False

    def on_space(_v):
        if not _debounce("SPACE", 0.10):
            return False
        if state["disp"] >= end_frame:
            return False
        _render_frame(min(end_frame, state["disp"] + int(vis_n)))
        return False
    vis.register_key_callback(ord("G"), on_toggle_grad)
    vis.register_key_callback(ord("T"), on_toggle_pose)
    vis.register_key_callback(ord("V"), on_toggle_color)
    vis.register_key_callback(ord("A"), on_back)
    vis.register_key_callback(ord("D"), on_forward)
    vis.register_key_callback(32, on_space)

    _print_state(" [INIT]")
    _render_frame(state["disp"])

    try:
        vis.run()
    finally:
        vis.destroy_window()
