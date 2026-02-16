import os
import time
import numpy as np
import open3d as o3d


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
    if not pcd.has_colors():
        pcd.paint_uniform_color([0.65, 0.65, 0.65])
    return pcd


def run_live_viz(
    tracker,
    map_ply: str,
    begin_frame: int,
    end_frame: int,
    vis_n: int = 1,
):
    """
    Only:
      1) show reconstructed map ply (map_ply)
      2) show solver-used processed points: tracker.last_used_points_world
    Keys:
      A : prev frame
      D : next frame
      SPACE : forward by vis_n
      R : reset view
      Q / ESC : quit
    """

    begin_frame = int(begin_frame)
    end_frame = int(end_frame)
    if end_frame < 0:
        end_frame = len(tracker.file_list) - 1
    end_frame = min(end_frame, len(tracker.file_list) - 1)
    begin_frame = max(0, begin_frame)
    if begin_frame > end_frame:
        begin_frame = end_frame

    tracker.running_idx = begin_frame

    map_geom = _load_map(map_ply)

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window("Map + Solver Used Points (A/D/SPACE/R)", width=1280, height=720)
    vis.add_geometry(map_geom)

    used_pcd = o3d.geometry.PointCloud()
    vis.add_geometry(used_pcd)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.0, 0.0, 0.0])
    opt.point_size = 3.0
    opt.show_coordinate_frame = True
    if hasattr(opt, "light_on"):
        opt.light_on = True
    if hasattr(opt, "mesh_show_back_face"):
        opt.mesh_show_back_face = True

    state = {
        "disp": begin_frame,
        "key_last": {},
        "used_pts_cache": {},   # fid -> (M,3) np.float64 or None
        "view_inited": False,
    }

    def _debounce(key: str, dt: float = 0.10) -> bool:
        now = time.time()
        last = state["key_last"].get(key, 0.0)
        if now - last < dt:
            return False
        state["key_last"][key] = now
        return True

    def _ensure_upto(target_fid: int):
        target_fid = int(target_fid)
        while int(tracker.running_idx) <= target_fid:
            fid = int(tracker.running_idx)
            tracker.register_next_global() 

            pts_used = getattr(tracker, "last_used_points_world", None)
            if pts_used is not None and hasattr(pts_used, "numel") and pts_used.numel() > 0:
                pts_np = pts_used.detach().to("cpu").numpy().astype(np.float64)
                state["used_pts_cache"][fid] = pts_np
            else:
                state["used_pts_cache"][fid] = None

    def _render(fid: int, reset_view: bool = False):
        fid = int(np.clip(fid, begin_frame, end_frame))
        state["disp"] = fid

        _ensure_upto(fid)
        pts = state["used_pts_cache"].get(fid, None)

        if pts is None or len(pts) == 0:
            used_pcd.points = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
            used_pcd.colors = o3d.utility.Vector3dVector(np.zeros((0, 3), dtype=np.float64))
        else:
            used_pcd.points = o3d.utility.Vector3dVector(pts)
            used_pcd.paint_uniform_color([0.0, 1.0, 1.0])

        vis.update_geometry(used_pcd)

        if reset_view or (not state["view_inited"]):
            vis.reset_view_point(True)
            state["view_inited"] = True

        vis.poll_events()
        vis.update_renderer()

        n_used = 0 if pts is None else len(pts)
        print(f"[VIS] fid={fid} used_pts={n_used} trk_next={int(tracker.running_idx)}")

    # ---- keys ----
    def on_back(_v):
        if not _debounce("A", 0.06):
            return False
        if state["disp"] <= begin_frame:
            return False
        _render(state["disp"] - 1)
        return False

    def on_forward(_v):
        if not _debounce("D", 0.06):
            return False
        if state["disp"] >= end_frame:
            return False
        _render(state["disp"] + 1)
        return False

    def on_space(_v):
        if not _debounce("SPACE", 0.10):
            return False
        if state["disp"] >= end_frame:
            return False
        _render(min(end_frame, state["disp"] + int(vis_n)))
        return False

    def on_reset(_v):
        if not _debounce("R", 0.15):
            return False
        _render(state["disp"], reset_view=True)
        return False

    def on_quit(_v):
        vis.close()
        return False

    vis.register_key_callback(ord("A"), on_back)
    vis.register_key_callback(ord("D"), on_forward)
    vis.register_key_callback(32, on_space)      # SPACE
    vis.register_key_callback(ord("R"), on_reset)
    vis.register_key_callback(ord("Q"), on_quit)
    vis.register_key_callback(256, on_quit)      # ESC

    # init
    _render(state["disp"], reset_view=True)

    try:
        vis.run()
    finally:
        vis.destroy_window()
