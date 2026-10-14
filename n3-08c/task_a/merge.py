"""Merge: transform template tới vị trí mới, concat vào target point cloud, tạo annotation mới.

Chạy demo: python task_a/merge.py
"""
import sys
import uuid
from pathlib import Path
from typing import Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from task_a.ground_plane import GroundModel, Plane, project_z_to_ground  # noqa: E402
from utils.transforms import from_local_frame, yaw_to_quaternion  # noqa: E402


def insert_object(target_points: np.ndarray, object_template: dict, position: Tuple[float, float, float],
                  ground_plane: Union[Plane, GroundModel]) -> np.ndarray:
    """Đặt template tại position = (x, y, yaw), đáy box chạm mặt đất -> augmented points (N+M, 5).
    Xử lý xoá điểm background bị che khuất (occlusion) và điểm bên trong bounding box.
    """
    from utils.transforms import points_in_cuboid
    x, y, yaw = position
    size = object_template["size"]
    h = size[2]
    
    # +- 15cm margin used for depth occlusion checking
    ground_z = project_z_to_ground(x, y, ground_plane)
    center = [x, y, ground_z + h / 2]
    rot_quat = yaw_to_quaternion(yaw)
    placed = from_local_frame(object_template["points"], center, rot_quat)
    
    # 1. Xoá điểm target nằm trong box mới
    in_box_mask = points_in_cuboid(target_points, center, size, rot_quat)
    target_points = target_points[~in_box_mask]
    
    # 2. Xử lý che khuất (Occlusion) bằng Range Image
    all_points = np.concatenate([target_points, placed.astype(target_points.dtype)], axis=0)
    
    xyz = all_points[:, :3]
    r = np.linalg.norm(xyz, axis=1)
    # Lọc bỏ các điểm ở gốc tọa độ (r=0) để tránh lỗi chia 0
    valid_r = r > 0.01
    
    xyz_valid = xyz[valid_r]
    r_valid = r[valid_r]
    
    azimuth = np.arctan2(xyz_valid[:, 1], xyz_valid[:, 0])
    elevation = np.arcsin(np.clip(xyz_valid[:, 2] / r_valid, -1.0, 1.0))
    
    num_azimuth = 3600  # 0.1 degree resolution
    num_elevation = 64  # cover FOV
    
    az_bins = np.floor((azimuth + np.pi) / (2 * np.pi) * num_azimuth).astype(int) % num_azimuth
    el_min, el_max = -0.6, 0.3
    el_bins = np.floor((elevation - el_min) / (el_max - el_min) * num_elevation).astype(int)
    el_bins = np.clip(el_bins, 0, num_elevation - 1)
    
    # Xây dựng depth map
    depth_map = np.full((num_azimuth, num_elevation), np.inf)
    
    # Update depth_map with minimum depth for each bin
    # We can use np.minimum.at for unbuffered in-place updates
    np.minimum.at(depth_map, (az_bins, el_bins), r_valid)
    
    # Lọc các điểm bị che khuất: depth lớn hơn depth map + margin (0.15m)
    margin_depth = 0.15
    not_occluded = r_valid <= (depth_map[az_bins, el_bins] + margin_depth)
    
    # Tạo mask cho toàn bộ all_points
    final_mask = np.zeros(len(all_points), dtype=bool)
    final_mask[valid_r] = not_occluded
    
    augmented_points = all_points[final_mask]
    
    # Cập nhật số điểm của object (phần tử cuối cùng thuộc placed)
    placed_mask = final_mask[-len(placed):]
    object_template["final_num_pts"] = np.sum(placed_mask)
    
    return augmented_points


def create_annotation(object_template: dict, position: Tuple[float, float, float], ground_z: float) -> dict:
    """Annotation dict (sensor frame) cho object vừa chèn."""
    x, y, yaw = position
    size: Sequence[float] = object_template["size"]
    num_lidar_pts = object_template.get("final_num_pts", int(len(object_template["points"])))
    w, l, h = float(size[0]), float(size[1]), float(size[2])
    return {
        "token": uuid.uuid4().hex,
        "translation": [float(x), float(y), float(ground_z + h / 2)],
        "size": [w, l, h],
        "width": w,
        "length": l,
        "height": h,
        "yaw": float(yaw),
        "rotation": yaw_to_quaternion(yaw),
        "category": object_template["category"],
        "num_lidar_pts": int(num_lidar_pts),
        "synthetic": True,
        "source_ann_token": object_template.get("ann_token"),
    }


if __name__ == "__main__":
    from utils.transforms import points_in_cuboid

    rng = np.random.default_rng(0)
    target = np.c_[rng.uniform(-30, 30, (5000, 2)), np.full(5000, -1.8), rng.uniform(0, 255, (5000, 2))]
    template = {
        "points": np.c_[rng.uniform([-2.2, -0.9, -0.8], [2.2, 0.9, 0.8], (300, 3)), rng.uniform(0, 255, (300, 2))],
        "size": [1.9, 4.5, 1.6], "category": "vehicle.car", "ann_token": "demo",
    }
    plane = (0.0, 0.0, 1.0, 1.8)  # z = -1.8
    pos = (12.0, -3.0, np.pi / 4)

    aug = insert_object(target, template, pos, plane)
    ann = create_annotation(template, pos, project_z_to_ground(pos[0], pos[1], plane))
    inside = points_in_cuboid(aug, ann["translation"], ann["size"], ann["rotation"]).sum()
    print(f"target {target.shape} -> augmented {aug.shape}; points trong box mới: {inside}")
    print("annotation:", {k: ann[k] for k in ("translation", "size", "category")})
