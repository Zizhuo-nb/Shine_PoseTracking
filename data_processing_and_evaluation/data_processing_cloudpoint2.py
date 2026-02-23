#!/usr/bin/env python3
from pathlib import Path
from rosbags.highlevel import AnyReader
import yaml
from plyfile import PlyData, PlyElement
import os
import argparse
import sys
from typing import Iterable, List, Optional, Tuple

import numpy as np

try:
    from rosbags.highlevel import AnyReader
except ImportError as e:
    raise ImportError(
        'No rosbags：\n'
        '    python -m pip install -U rosbags'
    ) from e

import open3d as o3d


_DATATYPES = {
    1: np.dtype(np.int8),    # INT8
    2: np.dtype(np.uint8),   # UINT8
    3: np.dtype(np.int16),   # INT16
    4: np.dtype(np.uint16),  # UINT16
    5: np.dtype(np.int32),   # INT32
    6: np.dtype(np.uint32),  # UINT32
    7: np.dtype(np.float32), # FLOAT32
    8: np.dtype(np.float64), # FLOAT64
}

DUMMY_FIELD_PREFIX = "unnamed_field"

def write_ply_xyz_time(filepath: str,
                       points: np.ndarray,
                       timestamps: np.ndarray) -> None:
    """
    time
    """
    if points.shape[0] != timestamps.shape[0]:
        raise ValueError(
            f"points 数量 {points.shape[0]} 和 timestamps 数量 "
            f"{timestamps.shape[0]} 不一致"
        )

    #dtype：x, y, z, time
    vertex = np.empty(
        points.shape[0],
        dtype=[
            ("x", "f8"),
            ("y", "f8"),
            ("z", "f8"),
            ("time", "f8"),
        ],
    )
    vertex["x"] = points[:, 0]
    vertex["y"] = points[:, 1]
    vertex["z"] = points[:, 2]
    vertex["time"] = timestamps

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el], text=False).write(filepath)

def dtype_from_fields(fields, point_step):
    """
    Convert a Iterable of sensor_msgs.msg.PointField messages to a np.dtype.
    :param fields: The point cloud fields.
                   (Type: iterable of sensor_msgs.msg.PointField)
    :param point_step: Point step size in bytes. Calculated from the given fields by default.
                       (Type: optional of integer)
    :returns: NumPy datatype
    """
    # Create a lists containing the names, offsets and datatypes of all fields
    field_names = []
    field_offsets = []
    field_datatypes = []
    for i, field in enumerate(fields):
        # Datatype as numpy datatype
        datatype = _DATATYPES[field.datatype]
        # Name field
        if field.name == "":
            name = f"{DUMMY_FIELD_PREFIX}_{i}"
        else:
            name = field.name
        # Handle fields with count > 1 by creating subfields with a suffix consiting
        # of "_" followed by the subfield counter [0 -> (count - 1)]
        assert field.count > 0, "Can't process fields with count = 0."
        for a in range(field.count):
            # Add suffix if we have multiple subfields
            if field.count > 1:
                subfield_name = f"{name}_{a}"
            else:
                subfield_name = name
            assert subfield_name not in field_names, "Duplicate field names are not allowed!"
            field_names.append(subfield_name)
            # Create new offset that includes subfields
            field_offsets.append(field.offset + a * datatype.itemsize)
            field_datatypes.append(datatype.str)

    # Create dtype
    dtype_dict = {"names": field_names, "formats": field_datatypes, "offsets": field_offsets}
    if point_step is not None:
        dtype_dict["itemsize"] = point_step
    return np.dtype(dtype_dict)

def read_points(
    cloud,
    field_names: Optional[List[str]] = None,
    uvs: Optional[Iterable] = None,
    reshape_organized_cloud: bool = False,
) -> np.ndarray:
    """
    Read points from a sensor_msgs.PointCloud2 message.
    :param cloud: The point cloud to read from sensor_msgs.PointCloud2.
    :param field_names: The names of fields to read. If None, read all fields.
                        (Type: Iterable, Default: None)
    :param uvs: If specified, then only return the points at the given
        coordinates. (Type: Iterable, Default: None)
    :param reshape_organized_cloud: Returns the array as an 2D organized point cloud if set.
    :return: Structured NumPy array containing all points.
    """
    # Cast bytes to numpy array
    points = np.ndarray(
        shape=(cloud.width * cloud.height,),
        dtype=dtype_from_fields(cloud.fields, point_step=cloud.point_step),
        buffer=cloud.data,
    )

    # Keep only the requested fields
    if field_names is not None:
        assert all(
            field_name in points.dtype.names for field_name in field_names
        ), "Requests field is not in the fields of the PointCloud!"
        # Mask fields
        points = points[list(field_names)]

    # Swap array if byte order does not match
    if bool(sys.byteorder != "little") != bool(cloud.is_bigendian):
        points = points.byteswap(inplace=True)

    # Select points indexed by the uvs field
    if uvs is not None:
        # Don't convert to numpy array if it is already one
        if not isinstance(uvs, np.ndarray):
            uvs = np.fromiter(uvs, int)
        # Index requested points
        points = points[uvs]

    # Cast into 2d array if cloud is 'organized'
    if reshape_organized_cloud and cloud.height > 1:
        points = points.reshape(cloud.width, cloud.height)

    return points


def read_point_cloud(msg):
    """
    Extract points and timestamps from a PointCloud2-like message.

    :return: (points, timestamps)
        points: array of x, y, z points, shape: (N, 3)
        timestamps: array of per-point timestamps, shape: (N,)
    """
    field_names = ["x", "y", "z"]
    t_field = None
    for field in msg.fields:
        if field.name in ["t", "timestamp", "time"]:
            t_field = field.name
            field_names.append(t_field)
            break

    points_structured = read_points(msg, field_names=field_names)

    xs = np.asarray(points_structured["x"], dtype=np.float64)
    ys = np.asarray(points_structured["y"], dtype=np.float64)
    zs = np.asarray(points_structured["z"], dtype=np.float64)
    points = np.column_stack([xs, ys, zs])

    if t_field:
        timestamps = np.asarray(points_structured[t_field], dtype=np.float64)
    else:
        timestamps = np.array([], dtype=np.float64)

    #make the timestamps correspondence with the points
    if points.size > 0:
        valid_mask = ~np.any(np.isnan(points), axis=1)
        points = points[valid_mask]
        if t_field and timestamps.size > 0:
            timestamps = timestamps[valid_mask]

    return points, timestamps



def pointcloud2_to_xyz(msg) -> np.ndarray:
    points, _ = read_point_cloud(msg)  # (N,3), (_, per-point timestamps)
    return points


def convert_rosbag2(
    bag_dir: str,
    output_root: str,
    lidar_topic: str,
    storage_id: str = "sqlite3",          
    start_offset_sec: float = 0.0,        
    end_offset_sec: Optional[float] = None,  
):
    """
    generate velodyne/*.ply, scans_merged.pcd, times。
    """

    os.makedirs(output_root, exist_ok=True)
    velodyne_dir = os.path.join(output_root, "velodyne")
    os.makedirs(velodyne_dir, exist_ok=True)

    times_path = os.path.join(output_root, "times")
    merged_points_list = []

    bag_path = Path(bag_dir)

    with AnyReader([bag_path]) as reader, open(times_path, "w") as times_f:
        connections = [c for c in reader.connections if c.topic == lidar_topic]
        if not connections:
            available = "\n  ".join(sorted({c.topic for c in reader.connections}))
            raise RuntimeError(
                f"no topic: {lidar_topic}\n"
                f"the topic in the rosbags is: \n  {available}"
            )

        print(f"[INFO] start reading bag: {bag_dir}")
        print(f"[INFO] use topic: {lidar_topic}")
        print(f"[INFO] start: {start_offset_sec} s")
        print(f"[INFO] end: {end_offset_sec} s")

        frame_idx = 0
        t0_ns = None  # start

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            ts_ns = int(timestamp)
            if t0_ns is None:
                t0_ns = ts_ns 

            rel_t = (ts_ns - t0_ns) / 1e9  # relative time

            if rel_t < start_offset_sec:
                continue


            if end_offset_sec is not None and rel_t > end_offset_sec:
                print("[INFO] overtime")
                break

            # 
            msg = reader.deserialize(rawdata, connection.msgtype)

            #  points + per-point timestamps
            points, timestamps = read_point_cloud(msg)  # (N,3), (N,)
            if points.size == 0:
                continue

            if timestamps.size == 0:
                print(
                    "[WARN] no per-point timestamps "
                    "(t/time/timestamp), can't use for deskew,only for xyz。"
                )

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                o3d.io.write_point_cloud(ply_path, pcd, write_ascii=False)
            else:
                #  x,y,z,time
                times_f.write(f"{ts_ns}\n")
                ply_name = f"{ts_ns}.ply"
                ply_path = os.path.join(velodyne_dir, ply_name)

                write_ply_xyz_time(ply_path, points, timestamps)

            merged_points_list.append(points)
            frame_idx += 1
            if frame_idx % 50 == 0:
                print(f"[INFO] already done {frame_idx} frame")


    print(f"[INFO] all frames: {frame_idx}")

    if merged_points_list:
        merged_points = np.vstack(merged_points_list)
        merged_pcd = o3d.geometry.PointCloud()
        merged_pcd.points = o3d.utility.Vector3dVector(merged_points)
        merged_path = os.path.join(output_root, "scans_merged.pcd")
        o3d.io.write_point_cloud(merged_path, merged_pcd, write_ascii=False)
        print(f"[INFO] merged: {merged_path}")
    else:
        print("[WARN] no scans_merged.pcd")


def main():
    """
    reading setting file
    generate data
    """
    # path
    script_dir = Path(__file__).resolve().parent
    config_path = script_dir / "parameters.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"can't find the setting file .yaml: {config_path}")

    # read YAML
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # setting
    bag_dir = cfg["bag"]                 
    lidar_topic = cfg["topic"]           
    out_dir = cfg["out"]                 
    storage_id = cfg.get("storage_id", "sqlite3")  
    start_offset_sec = cfg.get("start_offset_sec", 0.0)
    end_offset_sec = cfg.get("end_offset_sec", None)

    print("setting:")
    print(f"  bag        = {bag_dir}")
    print(f"  topic      = {lidar_topic}")
    print(f"  out        = {out_dir}")
    print(f"  storage_id = {storage_id}")
    print(f"  start_offset_s  = {start_offset_sec}")
    print(f"  end_offset_s    = {end_offset_sec}")

    
    convert_rosbag2(
        bag_dir=bag_dir,
        output_root=out_dir,
        lidar_topic=lidar_topic,
        storage_id=storage_id,
        start_offset_sec=float(start_offset_sec),
        end_offset_sec=float(end_offset_sec) if end_offset_sec is not None else None,
    )


if __name__ == "__main__":
    main()