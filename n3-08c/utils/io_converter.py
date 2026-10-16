"""
Point Cloud Multi-Format I/O Converter
======================================
Hỗ trợ chuyển đổi toàn diện các định dạng đám mây điểm 3D LiDAR:
- .bin (nuScenes 5-channel [x, y, z, intensity, ring] / KITTI 4-channel)
- .pcd (Point Cloud Data - ASCII & Binary, chuẩn CVAT 3D / LabelCloud)
- .ply (Stanford Polygon format - CVAT, MeshLab, CloudCompare)
- .las / .laz (ASPRS standard format cho Mobile / Airborne LiDAR)
- .xyz / .txt / .csv (Delimited ASCII coordinate tables)
- .npy (NumPy array format)

Hỗ trợ Open3D / laspy khi có sẵn, kèm fallback thuần NumPy/Python để đảm bảo
tương thích 100% trên mọi nền tảng Windows/Linux.
"""

import io
import os
import re
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np

# Thử nạp Open3D và laspy
try:
    import open3d as o3d
    HAS_OPEN3D = True
except Exception:
    HAS_OPEN3D = False

try:
    import laspy
    HAS_LASPY = True
except Exception:
    HAS_LASPY = False


def estimate_ring_index(points_xyz: np.ndarray, num_rings: int = 32) -> np.ndarray:
    """
    Ước lượng chỉ số tia (ring index 0..num_rings-1) từ góc ngước (elevation)
    cho cảm biến Velodyne HDL-32E (FOV dọc: -30.67° đến +10.67°).
    """
    xyz = points_xyz[:, :3]
    r = np.linalg.norm(xyz, axis=1) + 1e-6
    elevation_rad = np.arcsin(np.clip(xyz[:, 2] / r, -1.0, 1.0))
    elev_deg = np.rad2deg(elevation_rad)
    # Khoảng -30.67 đến +10.67 (~41.34 deg)
    ring = np.clip(np.round((elev_deg + 30.67) / 41.34 * (num_rings - 1)), 0, num_rings - 1)
    return ring.astype(np.float32)


# ==================== FALLBACK PCD / PLY READ & WRITE ====================

def _write_pcd_ascii(pts: np.ndarray, out_path: Path):
    """Ghi định dạng .pcd ASCII chuẩn PCD v0.7 (CVAT / CloudCompare / PCL)."""
    n_pts = len(pts)
    has_ring = pts.shape[1] >= 5
    header = [
        "# .PCD v0.7 - Point Cloud Data",
        "VERSION 0.7",
        "FIELDS x y z intensity ring" if has_ring else "FIELDS x y z intensity",
        "SIZE 4 4 4 4 4" if has_ring else "SIZE 4 4 4 4",
        "TYPE F F F F F" if has_ring else "TYPE F F F F",
        "COUNT 1 1 1 1 1" if has_ring else "COUNT 1 1 1 1",
        f"WIDTH {n_pts}",
        "HEIGHT 1",
        "VIEWPOINT 0 0 0 1 0 0 0",
        f"POINTS {n_pts}",
        "DATA ascii"
    ]
    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        if has_ring:
            np.savetxt(f, pts[:, :5], fmt="%.4f %.4f %.4f %.2f %d")
        else:
            np.savetxt(f, pts[:, :4], fmt="%.4f %.4f %.4f %.2f")


def _read_pcd_simple(file_path_or_bytes) -> np.ndarray:
    """Đọc tệp .pcd ASCII hoặc nhị phân cơ bản khi không có Open3D."""
    if isinstance(file_path_or_bytes, (str, Path)):
        with open(file_path_or_bytes, "rb") as f:
            raw = f.read()
    elif hasattr(file_path_or_bytes, "read"):
        raw = file_path_or_bytes.read()
    else:
        raw = file_path_or_bytes

    header_lines = []
    offset = 0
    while True:
        line_end = raw.find(b"\n", offset)
        if line_end == -1:
            break
        line = raw[offset:line_end].decode("ascii", errors="ignore").strip()
        offset = line_end + 1
        if not line or line.startswith("#"):
            continue
        header_lines.append(line)
        if line.startswith("DATA"):
            break

    data_type = "ascii"
    fields = ["x", "y", "z"]
    for hl in header_lines:
        if hl.startswith("DATA"):
            data_type = hl.split()[1].lower()
        elif hl.startswith("FIELDS"):
            fields = hl.split()[1:]

    data_bytes = raw[offset:]
    if data_type == "ascii":
        arr = np.loadtxt(io.StringIO(data_bytes.decode("ascii", errors="ignore")), dtype=np.float32)
        if arr.ndim == 1 and len(arr) > 0:
            arr = arr.reshape(1, -1)
    elif data_type == "binary":
        n_fields = len(fields)
        arr = np.frombuffer(data_bytes, dtype=np.float32).reshape(-1, n_fields)
    else:
        raise ValueError(f"Unsupported PCD data type: {data_type}")

    return arr


def _write_ply_ascii(pts: np.ndarray, out_path: Path):
    """Ghi định dạng .ply ASCII chuẩn Stanford PLY (MeshLab, CloudCompare, CVAT)."""
    n_pts = len(pts)
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {n_pts}",
        "property float x",
        "property float y",
        "property float z",
        "property float intensity",
        "end_header"
    ]
    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(header) + "\n")
        np.savetxt(f, pts[:, :4], fmt="%.4f %.4f %.4f %.2f")


# ==================== MAIN FUNCTIONS ====================

def load_point_cloud(file_path_or_bytes, file_ext: Optional[str] = None) -> np.ndarray:
    """
    Nạp dữ liệu mây điểm từ 8 định dạng sang mảng chuẩn (N, 5) float32:
    Các cột: [x, y, z, intensity, ring_index]
    """
    if isinstance(file_path_or_bytes, (str, Path)):
        file_path = Path(file_path_or_bytes)
        ext = file_path.suffix.lower() if file_ext is None else file_ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        path_str = str(file_path)
    else:
        ext = file_ext.lower() if file_ext else ".bin"
        if not ext.startswith("."):
            ext = "." + ext
        path_str = None

    # 1. RAW BINARY (.bin) - nuScenes (N, 5) hoặc KITTI (N, 4)
    if ext == ".bin":
        if path_str:
            data = np.fromfile(path_str, dtype=np.float32)
        else:
            raw = file_path_or_bytes.read() if hasattr(file_path_or_bytes, "read") else file_path_or_bytes
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
            raise ValueError(f"Không nhận diện được kích thước float32 cho .bin: {len(data)} phần tử")
        return pts.astype(np.float32)

    # 2. NUMPY (.npy)
    elif ext == ".npy":
        if path_str:
            arr = np.load(path_str)
        else:
            if hasattr(file_path_or_bytes, "seek"):
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
        raise ValueError(f"Mảng .npy không hợp lệ: {arr.shape}")

    # 3. LAS / LAZ (.las, .laz)
    elif ext in (".las", ".laz"):
        if not HAS_LASPY:
            raise ImportError("Cần thư viện laspy để đọc .las/.laz: `pip install laspy[lazrs]`")
        if path_str:
            las = laspy.read(path_str)
        else:
            buf = io.BytesIO(file_path_or_bytes) if isinstance(file_path_or_bytes, bytes) else file_path_or_bytes
            las = laspy.read(buf)

        x = np.asarray(las.x, dtype=np.float32)
        y = np.asarray(las.y, dtype=np.float32)
        z = np.asarray(las.z, dtype=np.float32)
        try:
            intensity = np.asarray(las.intensity, dtype=np.float32)
            if intensity.max() > 255.0:
                intensity = (intensity / 65535.0 * 255.0).astype(np.float32)
        except Exception:
            intensity = np.ones_like(x) * 50.0

        ring = estimate_ring_index(np.column_stack([x, y, z]))
        return np.column_stack([x, y, z, intensity, ring]).astype(np.float32)

    # 4. PCD (.pcd) hoặc PLY (.ply)
    elif ext in (".pcd", ".ply"):
        if HAS_OPEN3D and path_str:
            pcd = o3d.io.read_point_cloud(path_str)
            xyz = np.asarray(pcd.points, dtype=np.float32)
            if len(xyz) > 0:
                if pcd.has_colors():
                    colors = np.asarray(pcd.colors, dtype=np.float32)
                    intensity = (colors[:, 0] * 0.299 + colors[:, 1] * 0.587 + colors[:, 2] * 0.114) * 255.0
                else:
                    intensity = np.ones((len(xyz),), dtype=np.float32) * 50.0
                ring = estimate_ring_index(xyz)
                return np.column_stack([xyz, intensity, ring]).astype(np.float32)

        # Fallback đọc thuần
        if ext == ".pcd":
            arr = _read_pcd_simple(file_path_or_bytes if not path_str else path_str)
            xyz = arr[:, :3]
            intensity = arr[:, 3] if arr.shape[1] >= 4 else np.ones((len(xyz),), dtype=np.float32) * 50.0
            ring = arr[:, 4] if arr.shape[1] >= 5 else estimate_ring_index(xyz)
            return np.column_stack([xyz, intensity, ring]).astype(np.float32)
        else:
            # Fallback ply ascii
            content = (Path(path_str).read_bytes() if path_str else (
                file_path_or_bytes.read() if hasattr(file_path_or_bytes, "read") else file_path_or_bytes
            )).decode("ascii", errors="ignore")
            lines = content.splitlines()
            header_end = 0
            for i, l in enumerate(lines):
                if l.strip() == "end_header":
                    header_end = i + 1
                    break
            data_str = "\n".join(lines[header_end:])
            arr = np.loadtxt(io.StringIO(data_str), dtype=np.float32)
            xyz = arr[:, :3]
            intensity = arr[:, 3] if arr.shape[1] >= 4 else np.ones((len(xyz),), dtype=np.float32) * 50.0
            ring = estimate_ring_index(xyz)
            return np.column_stack([xyz, intensity, ring]).astype(np.float32)

    # 5. ASCII TABLES (.xyz, .txt, .csv, .pts)
    elif ext in (".xyz", ".txt", ".csv", ".pts"):
        if path_str:
            with open(path_str, "r", encoding="utf-8", errors="ignore") as f:
                first_line = f.readline()
                delimiter = "," if "," in first_line else None
            arr = np.loadtxt(path_str, delimiter=delimiter, dtype=np.float32)
        else:
            raw = file_path_or_bytes.read() if hasattr(file_path_or_bytes, "read") else file_path_or_bytes
            text = raw.decode("utf-8", errors="ignore")
            delimiter = "," if "," in text.split("\n", 1)[0] else None
            arr = np.loadtxt(io.StringIO(text), delimiter=delimiter, dtype=np.float32)

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
        raise ValueError(f"Định dạng bảng ASCII không hợp lệ: {arr.shape}")

    else:
        raise ValueError(f"Định dạng không được hỗ trợ: '{ext}'. Hỗ trợ: .bin, .pcd, .ply, .las, .laz, .xyz, .csv, .npy")


def save_point_cloud(points: np.ndarray, output_path: Union[str, Path], format: Optional[str] = None) -> str:
    """
    Lưu đám mây điểm (N, 4) hoặc (N, 5) sang 8 định dạng:
    - .bin: Nhị phân nuScenes float32
    - .pcd: Point Cloud Data (ASCII / Binary) cho CVAT 3D / LabelCloud
    - .ply: Stanford Polygon (ASCII / Binary)
    - .las / .laz: Chuẩn đo đạc Mobile/Aerial LiDAR
    - .xyz / .csv / .txt: Bảng tọa độ ASCII
    - .npy: Mảng NumPy nhị phân
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower() if format is None else format.lower()
    if not ext.startswith("."):
        ext = "." + ext

    pts = np.asarray(points, dtype=np.float32)

    # 1. .bin
    if ext == ".bin":
        pts.tofile(str(out_path))
        return str(out_path)

    # 2. .npy
    elif ext == ".npy":
        np.save(str(out_path), pts)
        return str(out_path)

    # 3. .pcd
    elif ext == ".pcd":
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
            if pts.shape[1] >= 4:
                norm_i = np.clip(pts[:, 3] / 255.0, 0.0, 1.0)
                colors = np.column_stack([norm_i, norm_i, norm_i]).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=True)
        else:
            _write_pcd_ascii(pts, out_path)
        return str(out_path)

    # 4. .ply
    elif ext == ".ply":
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts[:, :3].astype(np.float64))
            if pts.shape[1] >= 4:
                norm_i = np.clip(pts[:, 3] / 255.0, 0.0, 1.0)
                colors = np.column_stack([norm_i, norm_i, norm_i]).astype(np.float64)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            o3d.io.write_point_cloud(str(out_path), pcd)
        else:
            _write_ply_ascii(pts, out_path)
        return str(out_path)

    # 5. .las / .laz
    elif ext in (".las", ".laz"):
        if not HAS_LASPY:
            raise ImportError("Cần thư viện laspy để ghi .las/.laz: `pip install laspy[lazrs]`")
        header = laspy.LasHeader(point_format=2, version="1.2")
        las = laspy.LasData(header)
        las.x = pts[:, 0]
        las.y = pts[:, 1]
        las.z = pts[:, 2]
        if pts.shape[1] >= 4:
            las.intensity = np.clip(pts[:, 3] * 256.0, 0, 65535).astype(np.uint16)
        las.write(str(out_path))
        return str(out_path)

    # 6. .xyz / .csv / .txt
    elif ext in (".xyz", ".csv", ".txt"):
        delimiter = "," if ext == ".csv" else " "
        fmt = "%.4f"
        np.savetxt(str(out_path), pts, delimiter=delimiter, fmt=fmt)
        return str(out_path)

    else:
        raise ValueError(f"Định dạng xuất không được hỗ trợ: {ext}")
