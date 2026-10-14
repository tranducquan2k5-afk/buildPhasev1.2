"""
Point Cloud Multi-Format I/O Converter
======================================
Supports comprehensive 3D LiDAR point cloud formats across CVAT, LabelCloud,
CloudCompare, ROS, and Autonomous Driving frameworks:
- .bin (nuScenes 5-channel, KITTI 4-channel raw binary)
- .pcd (Point Cloud Data - ASCII & Binary, CVAT 3D standard)
- .ply (Stanford Polygon format - CVAT, MeshLab, CloudCompare)
- .las / .laz (ASPRS standard format for mobile and aerial LiDAR)
- .xyz / .txt / .csv (Delimited ASCII coordinate tables)
- .npy (NumPy array format)
"""

import os
from pathlib import Path
import numpy as np
import open3d as o3d

try:
    import laspy
    HAS_LASPY = True
except ImportError:
    HAS_LASPY = False


def estimate_ring_index(points_xyz, num_rings=32):
    """
    Estimates ring index (0 to num_rings-1) from elevation angle
    if ring index is missing in the input point cloud.
    """
    r = np.linalg.norm(points_xyz[:, :3], axis=1) + 1e-6
    elevation = np.arcsin(np.clip(points_xyz[:, 2] / r, -1.0, 1.0)) # radians
    # Standard Velodyne VLP-32C vertical FOV: -25 deg to +15 deg (~0.7 rad range)
    elev_deg = np.rad2deg(elevation)
    ring = np.clip(np.round((elev_deg + 25.0) / 40.0 * (num_rings - 1)), 0, num_rings - 1)
    return ring.astype(np.float32)


def load_point_cloud(file_path_or_bytes, file_ext=None):
    """
    Load point cloud from various formats into standard (N, 5) float32 numpy array:
    Columns: [x, y, z, intensity, ring_index]
    """
    if isinstance(file_path_or_bytes, (str, Path)):
        file_path = Path(file_path_or_bytes)
        ext = file_path.suffix.lower() if file_ext is None else file_ext.lower()
        if not ext.startswith('.'):
            ext = '.' + ext
        path_str = str(file_path)
    else:
        # File-like or bytes
        ext = file_ext.lower() if file_ext else '.bin'
        if not ext.startswith('.'):
            ext = '.' + ext
        path_str = None

    # 1. RAW BINARY (.bin) - nuScenes (N, 5) or KITTI (N, 4)
    if ext == '.bin':
        if path_str:
            data = np.fromfile(path_str, dtype=np.float32)
        else:
            if hasattr(file_path_or_bytes, 'read'):
                raw = file_path_or_bytes.read()
            else:
                raw = file_path_or_bytes
            data = np.frombuffer(raw, dtype=np.float32)

        if len(data) % 5 == 0 and (len(data) % 4 != 0 or len(data) >= 5000):
            pts = data.reshape(-1, 5)
        elif len(data) % 4 == 0:
            pts_4 = data.reshape(-1, 4)
            ring = estimate_ring_index(pts_4[:, :3])
            pts = np.column_stack([pts_4, ring]).astype(np.float32)
        elif len(data) % 3 == 0:
            pts_3 = data.reshape(-1, 3)
            intensity = np.ones((len(pts_3), 1), dtype=np.float32) * 50.0
            ring = estimate_ring_index(pts_3).reshape(-1, 1)
            pts = np.hstack([pts_3, intensity, ring]).astype(np.float32)
        else:
            raise ValueError(f"Cannot determine shape for .bin buffer with {len(data)} floats.")
        return pts.astype(np.float32)

    # 2. NUMPY (.npy)
    elif ext == '.npy':
        if path_str:
            arr = np.load(path_str)
        else:
            if hasattr(file_path_or_bytes, 'seek'):
                file_path_or_bytes.seek(0)
            arr = np.load(file_path_or_bytes)
        if arr.ndim == 2:
            if arr.shape[1] >= 5:
                return arr[:, :5].astype(np.float32)
            elif arr.shape[1] == 4:
                ring = estimate_ring_index(arr[:, :3])
                return np.column_stack([arr, ring]).astype(np.float32)
            elif arr.shape[1] == 3:
                intensity = np.ones((len(arr), 1), dtype=np.float32) * 50.0
                ring = estimate_ring_index(arr).reshape(-1, 1)
                return np.hstack([arr, intensity, ring]).astype(np.float32)
        raise ValueError(f"Invalid .npy point cloud shape: {arr.shape}")

    # 3. LAS / LAZ (.las, .laz)
    elif ext in ('.las', '.laz'):
        if not HAS_LASPY:
            raise ImportError("laspy is required to read .las/.laz files. Run `pip install laspy[lazrs]`.")
        if path_str:
            las = laspy.read(path_str)
        else:
            import io
            if isinstance(file_path_or_bytes, bytes):
                buf = io.BytesIO(file_path_or_bytes)
            else:
                buf = file_path_or_bytes
            las = laspy.read(buf)

        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        try:
            intensity = np.asarray(las.intensity, dtype=np.float32)
            # Normalize if intensity is in [0, 65535]
            if intensity.max() > 255.0:
                intensity = (intensity / 65535.0 * 255.0).astype(np.float32)
        except Exception:
            intensity = np.ones_like(x) * 50.0

        ring = estimate_ring_index(np.column_stack([x, y, z]))
        return np.column_stack([x, y, z, intensity, ring]).astype(np.float32)

    # 4. PCD (.pcd) or PLY (.ply)
    elif ext in ('.pcd', '.ply'):
        # Open3D handles PCD (ASCII and Binary) and PLY smoothly
        temp_created = False
        if not path_str:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                if hasattr(file_path_or_bytes, 'read'):
                    tmp.write(file_path_or_bytes.read())
                else:
                    tmp.write(file_path_or_bytes)
                path_str = tmp.name
                temp_created = True

        try:
            pcd = o3d.io.read_point_cloud(path_str)
            xyz = np.asarray(pcd.points, dtype=np.float32)
            if len(xyz) == 0:
                raise ValueError(f"Loaded empty point cloud from {ext} file.")

            # If color is present, use luminance as intensity
            if pcd.has_colors():
                colors = np.asarray(pcd.colors, dtype=np.float32)
                intensity = (colors[:, 0] * 0.299 + colors[:, 1] * 0.587 + colors[:, 2] * 0.114) * 255.0
            else:
                # Try parsing raw intensity from PCD header if binary or ascii
                intensity = np.ones((len(xyz),), dtype=np.float32) * 50.0

            ring = estimate_ring_index(xyz)
            pts = np.column_stack([xyz, intensity, ring]).astype(np.float32)
            return pts
        finally:
            if temp_created and os.path.exists(path_str):
                os.remove(path_str)

    # 5. ASCII TABLES (.xyz, .txt, .csv, .pts)
    elif ext in ('.xyz', '.txt', '.csv', '.pts'):
        if path_str:
            # Check separator (comma or whitespace)
            with open(path_str, 'r') as f:
                first_line = f.readline()
                delimiter = ',' if ',' in first_line else None
            arr = np.loadtxt(path_str, delimiter=delimiter, dtype=np.float32)
        else:
            if hasattr(file_path_or_bytes, 'read'):
                content = file_path_or_bytes.read().decode('utf-8', errors='ignore')
            else:
                content = file_path_or_bytes.decode('utf-8', errors='ignore')
            from io import StringIO
            delimiter = ',' if ',' in content.split('\n', 1)[0] else None
            arr = np.loadtxt(StringIO(content), delimiter=delimiter, dtype=np.float32)

        if arr.ndim == 2:
            if arr.shape[1] >= 5:
                return arr[:, :5].astype(np.float32)
            elif arr.shape[1] == 4:
                ring = estimate_ring_index(arr[:, :3])
                return np.column_stack([arr, ring]).astype(np.float32)
            elif arr.shape[1] == 3:
                intensity = np.ones((len(arr), 1), dtype=np.float32) * 50.0
                ring = estimate_ring_index(arr).reshape(-1, 1)
                return np.hstack([arr, intensity, ring]).astype(np.float32)
        raise ValueError(f"Invalid text point cloud shape: {arr.shape}")

    else:
        raise ValueError(f"Unsupported point cloud format: '{ext}'. Supported: .bin, .pcd, .ply, .las, .laz, .xyz, .csv, .txt, .npy")


def save_point_cloud(points, output_path, format=None):
    """
    Save (N, 4) or (N, 5) point cloud to destination format:
    - .bin: nuScenes float32 binary
    - .pcd: Point Cloud Data (Open3D format)
    - .ply: Stanford Polygon (Open3D format)
    - .las / .laz: ASPRS format
    - .xyz / .csv / .txt: ASCII coordinate tables
    - .npy: NumPy binary format
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower() if format is None else format.lower()
    if not ext.startswith('.'):
        ext = '.' + ext

    pts = np.asarray(points, dtype=np.float32)

    # 1. .bin
    if ext == '.bin':
        pts.tofile(str(output_path))
        return str(output_path)

    # 2. .npy
    elif ext == '.npy':
        np.save(str(output_path), pts)
        return str(output_path)

    # 3. .pcd or .ply
    elif ext in ('.pcd', '.ply'):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
        if pts.shape[1] >= 4:
            # Save intensity as grayscale color for Open3D/CloudCompare/MeshLab
            norm_i = np.clip(pts[:, 3] / 255.0, 0.0, 1.0)
            colors = np.column_stack([norm_i, norm_i, norm_i]).astype(np.float64)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        if ext == '.pcd':
            o3d.io.write_point_cloud(str(output_path), pcd, write_ascii=False)
        else:
            o3d.io.write_point_cloud(str(output_path), pcd)
        return str(output_path)

    # 4. .las or .laz
    elif ext in ('.las', '.laz'):
        if not HAS_LASPY:
            raise ImportError("laspy is required. Run `pip install laspy[lazrs]`.")
        header = laspy.LasHeader(point_format=2, version="1.2")
        las = laspy.LasData(header)
        las.x = pts[:, 0]
        las.y = pts[:, 1]
        las.z = pts[:, 2]
        if pts.shape[1] >= 4:
            las.intensity = np.clip(pts[:, 3] * 256.0, 0, 65535).astype(np.uint16)
        las.write(str(output_path))
        return str(output_path)

    # 5. .xyz or .csv or .txt
    elif ext in ('.xyz', '.csv', '.txt'):
        delimiter = ',' if ext == '.csv' else ' '
        fmt = '%.4f'
        np.savetxt(str(output_path), pts, delimiter=delimiter, fmt=fmt)
        return str(output_path)

    else:
        raise ValueError(f"Unsupported export format: {ext}")
