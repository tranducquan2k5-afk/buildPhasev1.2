"""Sample vị trí hợp lệ (x, y, yaw) trên drivable area + collision check BEV.

Box = annotation dict (translation, size [w, l, h], rotation [w,x,y,z]) ở sensor frame.
Chạy demo: python task_a/placement.py
"""
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from utils.transforms import quaternion_to_yaw, yaw_to_quaternion  # noqa: E402

Position = Tuple[float, float, float]  # (x, y, yaw)


def bev_footprint(center_xy: Sequence[float], size: Sequence[float], yaw: float) -> Polygon:
    """Hình chữ nhật BEV của box (size = [w, l, h])."""
    w, l = size[0], size[1]
    corners = np.array([[l / 2, w / 2], [l / 2, -w / 2], [-l / 2, -w / 2], [-l / 2, w / 2]])
    c, s = np.cos(yaw), np.sin(yaw)
    return Polygon(corners @ np.array([[c, s], [-s, c]]) + np.asarray(center_xy[:2]))


def box_footprint(box: dict) -> Polygon:
    return bev_footprint(box["translation"], box["size"], quaternion_to_yaw(box["rotation"]))


def bev_iou(a: Polygon, b: Polygon) -> float:
    inter = a.intersection(b).area
    return inter / (a.area + b.area - inter + 1e-9)


def check_collision_bev(new_box: dict, existing_boxes: Sequence[dict], buffer_margin: float = 1.5) -> bool:
    """True nếu new_box (buffer an toàn +-1.5m) có va chạm hoặc quá sát với bất kỳ box nào."""
    fp = box_footprint(new_box)
    fp_buffered = fp.buffer(buffer_margin) if buffer_margin > 0 else fp
    for b in existing_boxes:
        b_fp = box_footprint(b)
        if fp_buffered.intersects(b_fp):
            return True
        if bev_iou(fp, b_fp) > config.COLLISION_IOU_THRESH:
            return True
    return False


def is_mutually_separated(x: float, y: float, inserted_boxes: Sequence[dict],
                         min_dist: float = 15.0, min_azimuth_deg: float = 40.0) -> bool:
    """
    Quy tắc cách ly giữa các xe mới chèn trong cùng scene:
    a) Cách tâm mọi xe chèn trước đó >= 15.0m.
    b) Lệch góc phương vị azimuth >= 40 độ (chia đều vào các góc phần tư khác nhau).
    Tránh tuyệt đối 2 xe cùng nằm trên 1 đường ngắm ray cast gây tự xoá điểm của nhau.
    """
    if not inserted_boxes:
        return True
    from utils.transforms import wrap_angle
    theta = np.arctan2(y, x)
    min_az_rad = np.radians(min_azimuth_deg)
    for b in inserted_boxes:
        bx, by = b["translation"][0], b["translation"][1]
        dist = np.hypot(x - bx, y - by)
        if dist < min_dist:
            return False
        b_theta = np.arctan2(by, bx)
        d_theta = abs(wrap_angle(theta - b_theta))
        if d_theta < min_az_rad:
            return False
    return True


def estimate_road_heading(x: float, y: float, existing_boxes: Sequence[dict],
                          road_polygon: Optional[BaseGeometry] = None,
                          points: Optional[np.ndarray] = None) -> float:
    """
    Xác định góc hướng đường yaw_road:
    Ưu tiên 1: Lấy theo hướng của xe thật (existing_boxes) gần vị trí (x, y) nhất (khoảng cách <= 35m).
    Ưu tiên 2: Xác định trục chính của đám mây điểm mặt đường cục bộ quanh (x, y) bằng PCA.
    """
    # 1. Tìm xe thật gần nhất
    vehicles = [
        b for b in existing_boxes 
        if any(k in b.get("category", "").lower() for k in ("vehicle", "car", "truck", "bus"))
    ]
    if vehicles:
        dists = [np.hypot(b["translation"][0] - x, b["translation"][1] - y) for b in vehicles]
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= 35.0:
            ref_box = vehicles[min_idx]
            yaw_ref = quaternion_to_yaw(ref_box["rotation"])
            return float(yaw_ref)

    # 2. PCA trên các điểm mặt đường cục bộ quanh (x, y)
    if points is not None and len(points) > 0:
        local_mask = (np.hypot(points[:, 0] - x, points[:, 1] - y) <= 8.0) & (points[:, 2] < -1.0)
        local_pts = points[local_mask]
        if len(local_pts) >= 20:
            cov = np.cov(local_pts[:, :2], rowvar=False)
            evals, evecs = np.linalg.eigh(cov)
            primary_axis = evecs[:, 1]
            yaw_pca = float(np.arctan2(primary_axis[1], primary_axis[0]))
            return yaw_pca

    # 3. Fallback: tiếp tuyến với tâm LiDAR (hướng luồng giao thông)
    azimuth = np.arctan2(y, x)
    from utils.transforms import wrap_angle
    return float(wrap_angle(azimuth + np.pi / 2))


def is_on_drivable(point_xy: Sequence[float], drivable_polygon: BaseGeometry) -> bool:
    """True nếu điểm (x, y) nằm trong drivable area."""
    return drivable_polygon.contains(Point(point_xy[0], point_xy[1]))


def sample_valid_position(drivable_polygon: BaseGeometry, existing_boxes: Sequence[dict],
                          object_size: Sequence[float], n_attempts: int = 100,
                          rng: Optional[np.random.Generator] = None,
                          distance_range: Optional[Tuple[float, float]] = None,
                          points: Optional[np.ndarray] = None,
                          inserted_boxes: Optional[Sequence[dict]] = None) -> Optional[Position]:
    """Rejection sampling: (x, y) trên drivable area (cách tâm >= 8.0m), yaw bám theo hướng đường.
    Tuân thủ khoảng cách an toàn buffer 1.5m và cách ly góc phương vị >= 40 deg.
    
    Hợp lệ khi: tâm + toàn bộ footprint nằm trong drivable area và không va chạm BEV.
    Returns (x, y, yaw) hoặc None nếu hết n_attempts.
    """
    from utils.transforms import wrap_angle
    rng = rng or np.random.default_rng()
    if distance_range is not None:
        r_min, r_max = distance_range
    else:
        r_min, r_max = config.MIN_DISTANCE, config.MAX_DISTANCE
        
    # Đảm bảo r_min tối thiểu 8.0m theo yêu cầu để không chạm vòng tròn tâm LiDAR
    r_min = max(8.0, r_min)
    r_max = max(r_min + 2.0, r_max)

    for _ in range(n_attempts):
        r = np.sqrt(rng.uniform(r_min ** 2, r_max ** 2))
        phi = rng.uniform(-np.pi, np.pi)
        x, y = r * np.cos(phi), r * np.sin(phi)
        if not is_on_drivable((x, y), drivable_polygon):
            continue

        # Kiểm tra cách ly tối thiểu dist >= 15m và |delta theta| >= 40 deg so với các xe mới chèn
        if inserted_boxes and not is_mutually_separated(x, y, inserted_boxes, min_dist=15.0, min_azimuth_deg=40.0):
            continue

        # Xác định góc hướng đường từ xe thật gần nhất hoặc PCA
        base_yaw = estimate_road_heading(x, y, existing_boxes, drivable_polygon, points)
        
        # Thử 2 chiều (cùng chiều hoặc ngược chiều làn) với độ lệch góc nhỏ ngẫu nhiên +- 5 độ
        candidate_yaws = [
            wrap_angle(base_yaw + rng.uniform(-np.radians(5), np.radians(5))),
            wrap_angle(base_yaw + np.pi + rng.uniform(-np.radians(5), np.radians(5)))
        ]
        
        for yaw in candidate_yaws:
            if not drivable_polygon.contains(bev_footprint((x, y), object_size, yaw)):
                continue
            new_box = {"translation": [x, y, 0.0], "size": list(object_size), "rotation": yaw_to_quaternion(yaw)}
            # Buffer 1.5m để tạo khoảng cách an toàn giao thông tự nhiên
            if check_collision_bev(new_box, existing_boxes, buffer_margin=1.5):
                continue
            return float(x), float(y), float(yaw)
            
    return None


def sample_many(drivable_polygon: BaseGeometry, existing_boxes: Sequence[dict],
                sizes: Sequence[Sequence[float]], n_attempts: int = 100,
                rng: Optional[np.random.Generator] = None,
                distance_ranges: Optional[Sequence[Tuple[float, float]]] = None) -> List[Optional[Position]]:
    """Đặt lần lượt nhiều object; object đã đặt được tính vào collision check của object sau."""
    boxes, out = list(existing_boxes), []
    for i, size in enumerate(sizes):
        d_range = distance_ranges[i] if distance_ranges else None
        pos = sample_valid_position(drivable_polygon, boxes, size, n_attempts, rng, distance_range=d_range)
        out.append(pos)
        if pos is not None:
            boxes.append({"translation": [pos[0], pos[1], 0.0], "size": list(size),
                          "rotation": yaw_to_quaternion(pos[2])})
    return out


if __name__ == "__main__":
    from shapely.geometry import box as shp_box

    rng = np.random.default_rng(config.SEED)
    road = shp_box(-60, -4, 60, 4).union(shp_box(-4, -60, 4, 60))  # ngã tư giả
    existing = [{"translation": [10.0, 1.5, 0.0], "size": [1.9, 4.5, 1.6], "rotation": yaw_to_quaternion(0.0)}]
    car = [1.9, 4.5, 1.6]

    print("is_on_drivable((10, 0)):", is_on_drivable((10, 0), road))
    print("collision với chính nó:", check_collision_bev(existing[0], existing))
    positions = sample_many(road, existing, [car] * 5, rng=rng)
    for p in positions:
        print("  ->", None if p is None else tuple(round(v, 2) for v in p))
