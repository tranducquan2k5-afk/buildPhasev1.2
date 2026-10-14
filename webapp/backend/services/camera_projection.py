"""
Multi-modal Camera - LiDAR Synchronization & 3D to 2D Projection Engine
========================================================================
Projects 3D Bounding Boxes and LiDAR point clouds onto 6 nuScenes surround cameras:
- CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT
- CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT

Features:
1. Mathematical 3D-to-2D Coordinate Transformation:
   T_cam_from_lidar = inv(T_global_from_cam) @ T_global_from_lidar
   Projects 8 box corners via intrinsic matrix K with z_cam > 0.5m near-plane filtering.
2. 3D Bounding Box Rendering:
   Vibrant orange 12-edge wireframe with front-face 'X' cross indicator and distance badges.
3. LiDAR Point Cloud Overlay:
   Depth-colored points on synthetic objects overlaid onto camera lenses.
4. Task B Physical 2D Weather Simulation:
   - Rain: Semi-transparent tilted rain streaks + lens water spots + contrast reduction.
   - Fog: Koschmieder atmospheric transmission model: I(x) = J(x)*t(x) + A*(1 - t(x)).
"""

import base64
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from pyquaternion import Quaternion

# Add n3-08c and webapp/backend to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "n3-08c"))
sys.path.insert(0, str(PROJECT_ROOT / "webapp" / "backend"))

try:
    from server_config import get_cached_nuscenes
except (ImportError, ValueError):
    from webapp.backend.server_config import get_cached_nuscenes

from utils.transforms import points_in_cuboid, sensor_to_global_matrix

logger = logging.getLogger("camera_projection")

# 6 Surround Cameras ordered in cockpit layout:
# Row 1: Front hemisphere (Left, Center, Right)
# Row 2: Rear hemisphere (Left, Center, Right)
CAMERA_CHANNELS = [
    {
        "channel": "CAM_FRONT_LEFT",
        "name": "Front Left (Trước Trái)",
        "row": 1,
        "col": 1,
        "angle_deg": 55.0
    },
    {
        "channel": "CAM_FRONT",
        "name": "Front Center (Chính Diện Trước)",
        "row": 1,
        "col": 2,
        "angle_deg": 0.0
    },
    {
        "channel": "CAM_FRONT_RIGHT",
        "name": "Front Right (Trước Phải)",
        "row": 1,
        "col": 3,
        "angle_deg": -55.0
    },
    {
        "channel": "CAM_BACK_LEFT",
        "name": "Back Left (Sau Trái)",
        "row": 2,
        "col": 1,
        "angle_deg": 125.0
    },
    {
        "channel": "CAM_BACK",
        "name": "Back Center (Chính Diện Sau)",
        "row": 2,
        "col": 2,
        "angle_deg": 180.0
    },
    {
        "channel": "CAM_BACK_RIGHT",
        "name": "Back Right (Sau Phải)",
        "row": 2,
        "col": 3,
        "angle_deg": -125.0
    }
]


def apply_rain_2d(img: np.ndarray, level: str = "moderate", rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Mô phỏng mưa 2D: Vệt mưa rơi (Rain streaks) + mờ ống kính + giảm độ tương phản.
    """
    rng = rng or np.random.default_rng(42)
    h, w = img.shape[:2]
    out = img.copy()

    # Cấu hình vệt mưa theo cấp độ
    streak_counts = {"light": 600, "moderate": 1800, "heavy": 4500}
    n_streaks = streak_counts.get(level.lower(), 1800)

    # Lớp overlay trong suốt cho vệt mưa
    rain_layer = np.zeros((h, w, 3), dtype=np.uint8)

    # Góc rơi chéo do gió (75 đến 85 độ)
    angle_rad = np.radians(rng.uniform(76.0, 84.0))
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    lengths = rng.integers(18, 42, size=n_streaks)
    xs = rng.integers(0, w, size=n_streaks)
    ys = rng.integers(0, h, size=n_streaks)

    for i in range(n_streaks):
        x1, y1 = xs[i], ys[i]
        length = lengths[i]
        x2 = int(x1 + length * cos_a)
        y2 = int(y1 + length * sin_a)
        color = int(rng.integers(190, 245))
        thickness = 1 if level != "heavy" or rng.random() > 0.25 else 2
        cv2.line(rain_layer, (x1, y1), (x2, y2), (color, color, color), thickness, cv2.LINE_AA)

    # Đọng giọt nước trên thấu kính đối với mưa to (heavy)
    if level.lower() == "heavy":
        n_droplets = rng.integers(20, 45)
        for _ in range(n_droplets):
            cx = rng.integers(20, w - 20)
            cy = rng.integers(20, h - 20)
            rad = rng.integers(4, 16)
            cv2.circle(rain_layer, (cx, cy), rad, (220, 220, 230), 1, cv2.LINE_AA)
            cv2.circle(rain_layer, (cx, cy), max(1, rad // 2), (240, 240, 250), -1)

    # Hòa trộn alpha overlay
    alpha = 0.55 if level.lower() == "heavy" else (0.42 if level.lower() == "moderate" else 0.3)
    out = cv2.addWeighted(out, 1.0, rain_layer, alpha, 0)

    # Giảm tương phản và làm mờ quang học nhẹ
    contrast_factors = {"light": 0.95, "moderate": 0.88, "heavy": 0.78}
    cf = contrast_factors.get(level.lower(), 0.88)
    out = np.clip(out.astype(np.float32) * cf + 15 * (1.0 - cf), 0, 255).astype(np.uint8)

    blur_sigmas = {"light": 0.0, "moderate": 0.6, "heavy": 1.2}
    sigma = blur_sigmas.get(level.lower(), 0.6)
    if sigma > 0:
        ksize = 3 if sigma < 1.0 else 5
        out = cv2.GaussianBlur(out, (ksize, ksize), sigma)

    return out


def apply_fog_2d(img: np.ndarray, visibility: float = 50.0) -> np.ndarray:
    """
    Mô phỏng sương mù 2D theo mô hình truyền dẫn khí quyển Koschmieder:
    I(x) = J(x) * t(x) + A * (1 - t(x))
    Trong đó:
    - A: Ánh sáng khí quyển (Atmospheric light ~ [225, 225, 230] BGR).
    - t(x) = exp(-beta * d(x)) với beta = 3.912 / visibility.
    - d(x): Bản đồ chiều sâu độ dốc mặt đường theo phối cảnh camera (Inverse Perspective Mapping).
    """
    h, w = img.shape[:2]
    beta = 3.912 / max(5.0, visibility)
    A = np.array([225.0, 225.0, 230.0], dtype=np.float32)

    # Tạo gradient độ sâu theo phối cảnh:
    # Đường chân trời ở ~48% chiều cao ảnh, đáy ảnh là mặt đường ngay trước mũi xe (~3m)
    y_horizon = int(h * 0.48)
    y_indices = np.arange(h)
    delta_y = np.maximum(y_indices - y_horizon, 1)

    # Chiều sâu mặt đường tăng dần về phía đường chân trời
    d_road = np.clip(1.7 * 1266.0 / delta_y, 3.0, 220.0)
    d_sky = np.full(y_horizon, 220.0)
    depth_profile = np.concatenate([d_sky, d_road[y_horizon:]])

    depth_map = np.tile(depth_profile[:, None], (1, w))
    t = np.exp(-beta * depth_map)[:, :, None]

    foggy = img.astype(np.float32) * t + A * (1.0 - t)
    foggy = np.clip(foggy, 0, 255).astype(np.uint8)

    # Tán xạ làm mờ (Atmospheric scattering blur)
    ksize = 3 if visibility >= 50.0 else 5
    foggy = cv2.GaussianBlur(foggy, (ksize, ksize), 0)

    return foggy


class CameraProjectionEngine:
    def __init__(self):
        self.nusc = get_cached_nuscenes()

    def get_lidar_and_cam_transforms(self, sample_token: str, cam_channel: str) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Lấy ma trận chuyển đổi T_cam_from_lidar (4x4), K (3x3), và đường dẫn ảnh camera.
        """
        sample = self.nusc.get("sample", sample_token)

        # Lấy bản ghi LIDAR_TOP
        lidar_sd = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        cal_lidar = self.nusc.get("calibrated_sensor", lidar_sd["calibrated_sensor_token"])
        pose_lidar = self.nusc.get("ego_pose", lidar_sd["ego_pose_token"])

        # Lấy bản ghi Camera
        cam_sd = self.nusc.get("sample_data", sample["data"][cam_channel])
        cal_cam = self.nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
        pose_cam = self.nusc.get("ego_pose", cam_sd["ego_pose_token"])

        K = np.array(cal_cam["camera_intrinsic"], dtype=float)
        img_rel_path = cam_sd["filename"]
        img_abs_path = os.path.join(self.nusc.dataroot, img_rel_path)

        # T_cam_from_lidar = inv(T_global_from_cam) @ T_global_from_lidar
        T_global_lidar = sensor_to_global_matrix(cal_lidar, pose_lidar)
        T_global_cam = sensor_to_global_matrix(cal_cam, pose_cam)
        T_cam_from_lidar = np.linalg.inv(T_global_cam) @ T_global_lidar

        return T_cam_from_lidar, K, img_abs_path

    def project_scene_to_camera(self,
                                sample_token: str,
                                cam_info: dict,
                                boxes: List[dict],
                                points: Optional[np.ndarray] = None,
                                weather_type: str = "none",
                                weather_level: str = "moderate") -> dict:
        """
        Chiếu các 3D Bounding Box và điểm LiDAR mới chèn lên 1 camera duy nhất,
        sau đó áp hiệu ứng thời tiết 2D nếu có.
        """
        channel = cam_info["channel"]
        T_cam_from_lidar, K, img_path = self.get_lidar_and_cam_transforms(sample_token, channel)

        if not os.path.isfile(img_path):
            logger.warning(f"Image not found on disk: {img_path}")
            return {
                "channel": channel,
                "name": cam_info["name"],
                "row": cam_info["row"],
                "col": cam_info["col"],
                "has_objects": False,
                "num_objects": 0,
                "detected_objects": [],
                "image_base64": ""
            }

        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        detected_objects = []

        # 1. Vẽ các điểm LiDAR 3D thuộc các vật thể mới chèn (nếu có điểm)
        if points is not None and len(points) > 0 and len(boxes) > 0:
            for b in boxes:
                center = b.get("translation", [0, 0, 0])
                size = b.get("size", [1.9, 4.5, 1.6])
                rot = b.get("rotation", [1, 0, 0, 0])
                in_box_mask = points_in_cuboid(points, center, size, rot)
                p_obj = points[in_box_mask]
                if len(p_obj) == 0:
                    continue

                # Chuyển điểm sang hệ camera
                p_hom = np.hstack([p_obj[:, :3], np.ones((len(p_obj), 1))])
                p_cam = (p_hom @ T_cam_from_lidar.T)[:, :3]

                # Lọc điểm phía trước ống kính
                val = p_cam[:, 2] > 0.5
                p_cam = p_cam[val]
                if len(p_cam) == 0:
                    continue

                u = (K[0, 0] * p_cam[:, 0] / p_cam[:, 2] + K[0, 2]).astype(int)
                v = (K[1, 1] * p_cam[:, 1] / p_cam[:, 2] + K[1, 2]).astype(int)

                in_view = (u >= 0) & (u < w) & (v >= 0) & (v < h)
                u_in, v_in, z_in = u[in_view], v[in_view], p_cam[in_view, 2]

                for px, py, pz in zip(u_in, v_in, z_in):
                    norm_d = np.clip((pz - 4.0) / 35.0, 0.0, 1.0)
                    hue = int((1.0 - norm_d) * 115)  # Đỏ (gần) -> Xanh lục (xa)
                    hsv_pix = np.uint8([[[hue, 255, 255]]])
                    bgr_col = cv2.cvtColor(hsv_pix, cv2.COLOR_HSV2BGR)[0, 0].tolist()
                    cv2.circle(img, (int(px), int(py)), 2, bgr_col, -1)

        # 2. Chiếu 3D Bounding Box của vật thể chèn
        box_color = (0, 140, 255)       # Cam rực rỡ (BGR)
        front_color = (0, 220, 255)     # Vàng sáng viền trước (BGR)
        wire_thickness = 2

        # 12 cạnh khung dây 3D:
        # Đáy: (0,1), (1,2), (2,3), (3,0)
        # Nóc: (4,5), (5,6), (6,7), (7,4)
        # Trụ đứng: (0,4), (1,5), (2,6), (3,7)
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        for b in boxes:
            size = b.get("size", [1.9, 4.5, 1.6])
            trans = b.get("translation", [0, 0, 0])
            rot = b.get("rotation", [1, 0, 0, 0])
            cat = b.get("category", "vehicle.car")

            w_box, l_box, h_box = float(size[0]), float(size[1]), float(size[2])
            cx, cy, cz = float(trans[0]), float(trans[1]), float(trans[2])

            # 8 đỉnh ở hệ tọa độ cục bộ (X: dọc xe / chiều dài, Y: ngang / chiều rộng, Z: đứng)
            corners_local = np.array([
                [ l_box / 2,  w_box / 2, -h_box / 2],  # 0: Front-Left-Bottom
                [ l_box / 2, -w_box / 2, -h_box / 2],  # 1: Front-Right-Bottom
                [-l_box / 2, -w_box / 2, -h_box / 2],  # 2: Rear-Right-Bottom
                [-l_box / 2,  w_box / 2, -h_box / 2],  # 3: Rear-Left-Bottom
                [ l_box / 2,  w_box / 2,  h_box / 2],  # 4: Front-Left-Top
                [ l_box / 2, -w_box / 2,  h_box / 2],  # 5: Front-Right-Top
                [-l_box / 2, -w_box / 2,  h_box / 2],  # 6: Rear-Right-Top
                [-l_box / 2,  w_box / 2,  h_box / 2],  # 7: Rear-Left-Top
            ], dtype=float)

            # Xoay và tịnh tiến sang hệ LIDAR_TOP
            R_lidar = Quaternion(rot).rotation_matrix
            corners_lidar = corners_local @ R_lidar.T + np.array([cx, cy, cz])

            # Chuyển đổi sang hệ Camera
            corners_hom = np.hstack([corners_lidar, np.ones((8, 1))])
            corners_cam = (corners_hom @ T_cam_from_lidar.T)[:, :3]

            # Kiểm tra: nếu tất cả các đỉnh nằm sau ống kính (z <= 0.5m) thì bỏ qua
            if np.all(corners_cam[:, 2] <= 0.5):
                continue

            # Chiếu sang mặt phẳng ảnh 2D
            z_clamped = np.maximum(corners_cam[:, 2], 0.2)
            u = K[0, 0] * corners_cam[:, 0] / z_clamped + K[0, 2]
            v = K[1, 1] * corners_cam[:, 1] / z_clamped + K[1, 2]
            pts_2d = np.column_stack([u, v])

            # Kiểm tra xem box có nằm trong tầm quét ống kính hay không
            min_x, max_x = float(pts_2d[:, 0].min()), float(pts_2d[:, 0].max())
            min_y, max_y = float(pts_2d[:, 1].min()), float(pts_2d[:, 1].max())

            # Điều kiện nhìn thấy: có ít nhất 1 phần giao thoa với khung hình
            if max_x < 0 or min_x >= w or max_y < 0 or min_y >= h:
                continue

            # Khoảng cách từ camera tới vật thể
            dist_cam = float(np.linalg.norm(corners_cam.mean(axis=0)))

            # Vẽ 12 cạnh khung dây 3D
            for p1_idx, p2_idx in edges:
                if corners_cam[p1_idx, 2] > 0.5 and corners_cam[p2_idx, 2] > 0.5:
                    pt1 = (int(pts_2d[p1_idx, 0]), int(pts_2d[p1_idx, 1]))
                    pt2 = (int(pts_2d[p2_idx, 0]), int(pts_2d[p2_idx, 1]))
                    # Cạnh mặt trước (0-1, 1-5, 5-4, 4-0) dùng màu vàng sáng
                    is_front = (p1_idx in (0, 1, 4, 5)) and (p2_idx in (0, 1, 4, 5))
                    edge_col = front_color if is_front else box_color
                    cv2.line(img, pt1, pt2, edge_col, wire_thickness, cv2.LINE_AA)

            # Vẽ dấu chữ X trên mặt trước để người dùng thấy rõ hướng đầu xe
            if (corners_cam[0, 2] > 0.5 and corners_cam[5, 2] > 0.5 and
                corners_cam[1, 2] > 0.5 and corners_cam[4, 2] > 0.5):
                pt0 = (int(pts_2d[0, 0]), int(pts_2d[0, 1]))
                pt5 = (int(pts_2d[5, 0]), int(pts_2d[5, 1]))
                pt1 = (int(pts_2d[1, 0]), int(pts_2d[1, 1]))
                pt4 = (int(pts_2d[4, 0]), int(pts_2d[4, 1]))
                cv2.line(img, pt0, pt5, (0, 200, 255), 1, cv2.LINE_AA)
                cv2.line(img, pt1, pt4, (0, 200, 255), 1, cv2.LINE_AA)

            # Gắn nhãn phân loại và khoảng cách
            label_x = int(np.clip(min_x, 10, w - 180))
            label_y = int(np.clip(min_y - 10, 25, h - 15))
            cat_short = cat.split(".")[-1].upper()
            label_text = f"{cat_short} | {dist_cam:.1f}m"

            (txt_w, txt_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(img, (label_x - 3, label_y - txt_h - 4), (label_x + txt_w + 5, label_y + baseline + 2), (20, 20, 20), -1)
            cv2.rectangle(img, (label_x - 3, label_y - txt_h - 4), (label_x + txt_w + 5, label_y + baseline + 2), box_color, 1)
            cv2.putText(img, label_text, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            detected_objects.append({
                "category": cat,
                "distance_m": round(dist_cam, 1),
                "box_2d": [int(min_x), int(min_y), int(max_x), int(max_y)]
            })

        # 3. Áp dụng hiệu ứng thời tiết 2D
        w_type = weather_type.lower()
        if w_type == "rain":
            img = apply_rain_2d(img, level=weather_level)
        elif w_type in ("fog", "sương mù"):
            vis_val = 20.0 if weather_level in ("dense", "heavy") else (50.0 if weather_level == "moderate" else 200.0)
            img = apply_fog_2d(img, visibility=vis_val)

        # 4. Vẽ tiêu đề góc nhìn camera
        header_text = f"[{channel}] {cam_info['name']}"
        cv2.rectangle(img, (10, 10), (10 + len(header_text) * 11, 40), (0, 0, 0), -1)
        status_color = (0, 220, 0) if len(detected_objects) > 0 else (180, 180, 180)
        cv2.putText(img, header_text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

        # Nén JPEG Base64
        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 78])
        b64_str = base64.b64encode(buffer).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64_str}"

        return {
            "channel": channel,
            "name": cam_info["name"],
            "row": cam_info["row"],
            "col": cam_info["col"],
            "has_objects": len(detected_objects) > 0,
            "num_objects": len(detected_objects),
            "detected_objects": detected_objects,
            "image_base64": data_uri,
            "width": w,
            "height": h
        }

    def generate_surround_view(self,
                               sample_token: str,
                               boxes: List[dict],
                               points: Optional[np.ndarray] = None,
                               weather_type: str = "none",
                               weather_level: str = "moderate") -> dict:
        """
        Sinh toàn bộ 6 góc nhìn Camera vòng quanh xe tự lái (6-Camera Surround View).
        """
        cameras_res = []
        cams_with_objects = 0

        for cam_info in CAMERA_CHANNELS:
            cam_data = self.project_scene_to_camera(
                sample_token=sample_token,
                cam_info=cam_info,
                boxes=boxes,
                points=points,
                weather_type=weather_type,
                weather_level=weather_level
            )
            if cam_data["has_objects"]:
                cams_with_objects += 1
            cameras_res.append(cam_data)

        return {
            "status": "success",
            "sample_token": sample_token,
            "total_cameras": len(cameras_res),
            "cameras_with_objects": cams_with_objects,
            "weather_type": weather_type,
            "weather_level": weather_level,
            "cameras": cameras_res
        }
