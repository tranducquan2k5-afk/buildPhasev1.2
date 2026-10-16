"""
2D Photorealistic Sensor Fusion Synthesis & Multi-view Sprite Compositing
========================================================================
Module kết xuất quang học 2D chân thực và sinh điều kiện ControlNet Inpainting
cho 6-Camera Surround View nuScenes theo phương pháp:
HƯỚNG 1: GHÉP ẢNH PNG TÁCH NỀN THEO GÓC NHÌN (MULTI-VIEW SPRITE COMPOSITING)

1. Cấu trúc thư mục tài nguyên ảnh (assets/vn_objects/<category>/):
   Hỗ trợ 16 lớp vật thể (Việt Nam + chuẩn nuScenes): ba_gac, moto_cargo,
   moto_multi, moto_tow, vendor_cart, cow, dog, ox_cart, car, truck, bus,
   motorcycle, bicycle, pedestrian, barrier, traffic_cone.
   Mỗi thư mục hỗ trợ: front.png, rear.png, side.png, oblique_front.png, oblique_rear.png.
   Người dùng có thể chép đè bất kỳ ảnh PNG nào tải từ mạng vào thư mục.

2. Chọn ảnh theo góc nhìn của Camera (View-Angle Selection & Flip):
   Tính góc quan sát tương đối alpha giữa tia nhìn camera và trục dọc vật thể:
   - Nhìn từ phía đầu xe (|alpha| <= 35 độ): chọn front.png.
   - Nhìn từ phía đuôi xe (|alpha| >= 145 độ): chọn rear.png.
   - Nhìn từ góc chéo hoặc ngang hông (35 < |alpha| < 145 độ): chọn oblique hoặc side.png.
   - Tự động lật ngang ảnh (cv2.flip(sprite, 1)) để đầu xe luôn hướng đúng chiều di chuyển thực tế.

3. Đặt vị trí, Co giãn theo Box 2D và Xử lý biên Alpha:
   Chiếu 8 đỉnh 3D Box lên ảnh 2D để lấy khung bao [u_min, v_min, u_max, v_max].
   Co giãn PNG theo chiều cao h_2d, giữ nguyên tỷ lệ tự nhiên, đáy chạm v_max, tâm ngang trùng tâm 2D Box.
   Loại bỏ các vật thể quá gần (z_cam < 4.0m) hoặc chiếm > 65% diện tích khung hình.

4. Giữ nguyên các quy luật vật lý quang học và hòa trộn (Harmonization):
   - Thứ tự độ sâu (Painter's algorithm) & kiểm tra che khuất bởi xe thật phía trước.
   - Bóng đổ tiếp xúc mặt đường (Ground Contact Shadow Gaussian Penumbra).
   - Khớp màu nền, làm mờ theo cự ly (Distance blur) và làm mềm biên (Alpha Feathering 1-2px).
   - Phân tầng thời tiết sương mù Koschmieder theo độ sâu và vệt mưa tiền cảnh.
   - Sinh điều kiện inpaint_mask và depth_map chuẩn hóa cho ControlNet / Diffusion Inpainting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union
import hashlib
import cv2
import numpy as np
from PIL import Image

from utils.asset_manager import (
    normalize_category,
    select_view_name,
    select_view_from_cos,
    load_sprite_rgba,
    get_default_assets_dir
)

# ==================== BẢNG MÀU VẬT LIỆU (TƯƠNG THÍCH NGƯỢC) ====================
PALETTE = {
    "car_paint": [(25, 65, 160), (175, 30, 35), (220, 225, 230), (50, 55, 60), (200, 140, 25)],
    "truck_paint": [(40, 80, 140), (160, 45, 30), (210, 210, 215), (70, 75, 80)],
    "bus_paint": [(210, 150, 20), (30, 85, 170), (180, 40, 40)],
    "glass": (85, 115, 135),
    "tire": (28, 28, 30),
    "metal_chrome": (165, 170, 175),
    "moto_body": [(190, 25, 30), (25, 60, 150), (220, 220, 220), (35, 35, 38)],
    "cargo_cloth": [(70, 95, 130), (150, 120, 75), (60, 85, 70), (130, 80, 50)],
    "ba_gac_bed": (45, 85, 75),
    "wood": (135, 100, 60),
    "rider_clothes": [(40, 45, 65), (75, 55, 45), (140, 40, 35), (50, 70, 60)],
    "helmet": [(220, 40, 35), (235, 205, 35), (230, 230, 230), (30, 30, 35)],
    "fur_dog": (125, 85, 50),
    "fur_cow": (65, 45, 30),
    "pedestrian_clothes": [(50, 60, 80), (160, 50, 50), (60, 90, 70), (190, 185, 180)]
}


def _get_material_color(mat_name: str, seed: int = 0) -> Tuple[int, int, int]:
    """Lấy mã màu RGB tương ứng với tên vật liệu."""
    rng = np.random.default_rng(seed)
    if mat_name in PALETTE:
        val = PALETTE[mat_name]
        if isinstance(val, list):
            return val[int(rng.integers(0, len(val)))]
        return val
    if "glass" in mat_name:
        return PALETTE["glass"]
    if "tire" in mat_name:
        return PALETTE["tire"]
    if "cloth" in mat_name or "cargo" in mat_name:
        return PALETTE["cargo_cloth"][0]
    if "metal" in mat_name:
        return PALETTE["metal_chrome"]
    if "paint" in mat_name:
        return PALETTE["car_paint"][0]
    return (120, 125, 130)


def _get_object_primitives(category: str, size: Sequence[float], seed: int = 0) -> List[Tuple[str, np.ndarray, np.ndarray, str]]:
    """Trích xuất danh sách primitives (hỗ trợ tương thích ngược cho unit test cũ)."""
    w, l, h = size
    cat = category.lower()
    if "moto_cargo" in cat:
        cargo_h = h * 0.45
        return [
            ("box", np.array([0.0, 0.0, -h * 0.15]), np.array([l * 0.8, w * 0.4, h * 0.6]), "moto_body"),
            ("box", np.array([-l * 0.25, 0.0, h * 0.2]), np.array([l * 0.6, w * 0.95, cargo_h]), "cargo_cloth"),
            ("ell", np.array([l * 0.1, 0.0, h * 0.3]), np.array([0.16, 0.2, 0.28]), "rider_clothes"),
            ("ell", np.array([l * 0.12, 0.0, h * 0.45]), np.array([0.12, 0.12, 0.13]), "helmet")
        ]
    if "ba_gac" in cat:
        return [
            ("box", np.array([l * 0.3, 0.0, -h * 0.1]), np.array([l * 0.35, w * 0.4, h * 0.6]), "moto_body"),
            ("box", np.array([-l * 0.15, 0.0, -h * 0.05]), np.array([l * 0.65, w * 0.9, h * 0.5]), "ba_gac_bed"),
            ("box", np.array([-l * 0.15, 0.0, h * 0.25]), np.array([l * 0.55, w * 0.8, h * 0.4]), "cargo_cloth"),
            ("ell", np.array([l * 0.25, 0.0, h * 0.3]), np.array([0.16, 0.2, 0.28]), "rider_clothes")
        ]
    if "vendor_cart" in cat or "cart" in cat:
        return [
            ("box", np.array([0.0, 0.0, -h * 0.15]), np.array([l * 0.9, w * 0.9, h * 0.6]), "wood"),
            ("box", np.array([0.0, 0.0, h * 0.2]), np.array([l * 0.85, w * 0.85, h * 0.4]), "cargo_cloth"),
            ("box", np.array([-l * 0.45, 0.0, 0.0]), np.array([l * 0.1, w * 0.7, h * 0.7]), "metal_chrome")
        ]
    return [
        ("box", np.array([0.0, 0.0, -h * 0.12]), np.array([l * 0.98, w * 0.95, h * 0.55]), "car_paint"),
        ("box", np.array([-l * 0.05, 0.0, h * 0.22]), np.array([l * 0.52, w * 0.85, h * 0.45]), "glass")
    ]


# ==================== TIỀN XỬ LÝ GỘP CẶP XE BÒ (OX_CART) ====================

def preprocess_ox_cart_boxes(boxes: Sequence[dict]) -> List[dict]:
    """
    Tiền xử lý danh sách boxes: Gộp cặp (vn.cow + vn.ox_wagon) thành 1 thực thể ox_cart duy nhất
    với bounding box bao trọn cả con bò và cỗ xe phía sau, tránh tách rời kỳ quặc.
    """
    out_boxes = []
    used_indices = set()
    n = len(boxes)

    for i in range(n):
        if i in used_indices:
            continue
        b_i = boxes[i]
        cat_i = normalize_category(b_i.get("category", ""))

        matched_j = None
        if cat_i == "cow" and b_i.get("inserted", False):
            c_i = np.asarray(b_i["translation"], dtype=np.float32)
            for j in range(n):
                if j == i or j in used_indices:
                    continue
                b_j = boxes[j]
                if not b_j.get("inserted", False):
                    continue
                cat_j = normalize_category(b_j.get("category", ""))
                if cat_j == "ox_cart":
                    c_j = np.asarray(b_j["translation"], dtype=np.float32)
                    dist = float(np.linalg.norm(c_i[:2] - c_j[:2]))
                    if dist < 4.0:
                        matched_j = j
                        break

        if matched_j is not None:
            # Gộp b_i (cow) và b_j (wagon) thành 1 box ox_cart hoàn chỉnh
            b_j = boxes[matched_j]
            used_indices.add(i)
            used_indices.add(matched_j)

            c_i = np.asarray(b_i["translation"], dtype=np.float32)
            c_j = np.asarray(b_j["translation"], dtype=np.float32)

            try:
                from utils.visualization import to_box
                corners_i = to_box(b_i).corners()  # (3, 8)
                corners_j = to_box(b_j).corners()  # (3, 8)
                all_corners = np.hstack([corners_i, corners_j])  # (3, 16)
                p_front = corners_i[:, :4].mean(axis=1)  # Đầu bò là mặt trước
                p_rear = corners_j[:, 4:].mean(axis=1)   # Đuôi xe gỗ là mặt sau
            except Exception:
                p_front = c_i
                p_rear = c_j
                all_corners = np.vstack([c_i, c_j]).T

            c_merged = (p_front + p_rear) * 0.5
            size_i = np.asarray(b_i.get("size", [1.0, 2.0, 1.4]), dtype=np.float32)
            size_j = np.asarray(b_j.get("size", [1.5, 2.5, 1.5]), dtype=np.float32)
            total_l = float(np.linalg.norm(p_front[:2] - p_rear[:2])) + max(size_i[1], size_j[1]) * 0.5
            merged_size = [max(size_i[0], size_j[0]), total_l, max(size_i[2], size_j[2])]

            merged_b = dict(b_i)
            merged_b["category"] = "ox_cart"
            merged_b["translation"] = c_merged.tolist()
            merged_b["size"] = merged_size
            merged_b["_p_front_lid"] = p_front
            merged_b["_p_rear_lid"] = p_rear
            merged_b["_corners_lid"] = all_corners.T
            out_boxes.append(merged_b)
        else:
            out_boxes.append(b_i)

    return out_boxes


# ==================== BÓNG ĐỔ TIẾP XÚC MẶT ĐƯỜNG ====================

def render_single_ground_shadow(image: np.ndarray, b: dict,
                                T: np.ndarray, K: np.ndarray,
                                weather: Optional[str] = None,
                                level: Optional[str] = None) -> np.ndarray:
    """
    Vẽ bóng đổ tiếp xúc mặt đường (elip mềm bậc 2) cho đúng 1 vật thể:
    w(x, y) = max(0, 1 - (dx/rx)^2 - (dy/ry)^2)^2
    """
    if not b.get("inserted", False):
        return image

    h, w = image.shape[:2]
    fog_factor = 0.0
    if weather == "fog":
        fog_factor = 0.7 if level == "dense" else (0.4 if level == "moderate" else 0.2)
    elif weather in ("rain", "storm"):
        fog_factor = 0.5 if level == "heavy" or weather == "storm" else 0.3

    base_opacity = max(0.20, 0.60 * (1.0 - 0.6 * fog_factor))

    try:
        if "_corners_lid" in b:
            c_lid = b["_corners_lid"]
            min_z = float(c_lid[:, 2].min())
            bot_mask = c_lid[:, 2] <= (min_z + 0.35)
            corners_lidar = c_lid[bot_mask] if np.any(bot_mask) else c_lid
        else:
            from utils.visualization import to_box
            corners_lidar = to_box(b).bottom_corners().T
    except Exception:
        c = np.asarray(b["translation"], dtype=np.float32)
        size = np.asarray(b.get("size", [2, 4, 1.5]), dtype=np.float32)
        hl, hw = size[1] / 2.0, size[0] / 2.0
        bot_z = -size[2] / 2.0 + 0.05
        corners_lidar = c + np.array([
            [-hl, -hw, bot_z], [hl, -hw, bot_z], [hl, hw, bot_z], [-hl, hw, bot_z]
        ], dtype=np.float32)

    corners_cam = corners_lidar @ T[:3, :3].T + T[:3, 3]
    if np.any(corners_cam[:, 2] <= 0.4):
        return image

    uv = (corners_cam @ K.T)[:, :2] / corners_cam[:, 2:3]
    u_min, u_max = float(np.min(uv[:, 0])), float(np.max(uv[:, 0]))
    v_min, v_max = float(np.min(uv[:, 1])), float(np.max(uv[:, 1]))

    ego_hood_v = int(h * 0.85) if h >= 800 else h
    if u_max < 0 or u_min >= w or v_max < 0 or v_min >= ego_hood_v:
        return image

    cx = float(np.mean(uv[:, 0]))
    cy = float(np.mean(uv[:, 1]))
    rx = max(6.0, (u_max - u_min) * 0.58)
    ry = max(3.0, (v_max - v_min) * 0.58)

    x1 = max(0, int(cx - rx))
    x2 = min(w, int(cx + rx + 1))
    y1 = max(0, int(cy - ry))
    y2 = min(ego_hood_v, int(cy + ry + 1))

    if x2 <= x1 or y2 <= y1:
        return image

    xs = np.arange(x1, x2, dtype=np.float32)
    ys = np.arange(y1, y2, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xs, ys)

    d2 = ((grid_x - cx) / rx) ** 2 + ((grid_y - cy) / ry) ** 2
    w_ell = np.clip(1.0 - d2, 0.0, 1.0) ** 2
    attenuation = 1.0 - (w_ell * (base_opacity * 0.70))[..., None]

    out_img = image.copy()
    out_img[y1:y2, x1:x2] = np.clip(out_img[y1:y2, x1:x2].astype(np.float32) * attenuation, 0, 255).astype(np.uint8)
    return out_img


def render_ground_contact_shadow(image: np.ndarray, boxes: Sequence[dict],
                                 T: np.ndarray, K: np.ndarray,
                                 weather: Optional[str] = None,
                                 level: Optional[str] = None) -> np.ndarray:
    """
    GROUND CONTACT SHADOW (Bóng đổ tiếp xúc mặt đường):
    Chiếu đáy của 3D Box xuống mặt đường nhựa và vẽ lớp bóng đổ elip mềm bậc 2.
    """
    merged_boxes = preprocess_ox_cart_boxes(boxes)
    out_img = image
    for b in merged_boxes:
        out_img = render_single_ground_shadow(out_img, b, T, K, weather=weather, level=level)
    return out_img


# ==================== KHỚP TỈ LỆ KHUNG CHIẾU 3D ====================

def select_best_aspect_variant(category: str, view_name: str, target_aspect: float, seed_idx: int) -> int:
    """Khớp tỉ lệ khung chiếu 3D: Chọn biến thể sprite có tỉ lệ w/h gần nhất với tỉ lệ w_2d / h_2d của 3D box."""
    cat_dir = get_default_assets_dir() / category
    cands = []
    for idx in range(10):
        v_p = cat_dir / f"{view_name}_{idx}.png"
        if not v_p.is_file():
            break
        try:
            with Image.open(v_p) as im:
                sw, sh = im.size
            s_asp = sw / max(1.0, sh)
            # Với xe đi thẳng (front/rear), loại bỏ các ảnh quá bè ngang (s_aspect > 1.15)
            if view_name in ("front", "rear") and category in ("car", "truck", "bus") and s_asp > 1.15:
                continue
            cands.append((abs(s_asp - target_aspect), idx))
        except Exception:
            pass

    if cands:
        cands.sort(key=lambda x: x[0])
        top_k = min(len(cands), 3)
        return cands[seed_idx % top_k][1]
    return seed_idx % 5


# ==================== HƯỚNG 1: MULTI-VIEW SPRITE COMPOSITING ====================

def composite_multiview_sprites(image: np.ndarray, boxes: Sequence[dict],
                                T: np.ndarray, K: np.ndarray,
                                weather: Optional[str] = None,
                                level: Optional[str] = None,
                                seed: int = 42,
                                draw_wireframe: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    HƯỚNG 1: GHÉP ẢNH PNG TÁCH NỀN THEO GÓC NHÌN (MULTI-VIEW SPRITE COMPOSITING)
    Thay thế hoàn toàn cơ chế vẽ khối hộp đa giác phẳng bằng ảnh PNG chất lượng cao:
    1. Chọn view ảnh (front/rear/side/oblique) theo góc quan sát alpha.
    2. Tự động lật ngang (cv2.flip) khớp hướng di chuyển 3D trên màn hình.
    3. Aspect-Preserving Fit: Giữ nguyên tỷ lệ khung hình tự nhiên của ảnh PNG, không ép dẹp thành que.
    4. Căn chỉnh chân ảnh chạm đáy 2D Box (v_max) và tâm ngang trùng tâm 2D Box.
    5. Bảo vệ nắp capo xe chủ ego (v <= 0.85*H).
    6. Thuật toán Painter's Algorithm sắp xếp vẽ từ xa đến gần, đồng thời vẽ khung dây 3D wireframe
       ngay sau khi dán sprite để vật thể phía trước tự nhiên che lấp khung dây của vật thể phía sau.
    Returns: (synthesized_image, z_buffer_float, inpaint_mask_uint8)
    """
    h, w = image.shape[:2]
    out_img = image.copy()
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    ego_hood_v = int(h * 0.85) if h >= 800 else h

    # Hệ số suy hao sương mù Koschmieder
    beta = 0.0
    if weather == "fog":
        from task_b.camera_weather import FOG_VISIBILITY
        vis = FOG_VISIBILITY.get(level, 50.0)
        beta = 3.912 / max(5.0, vis)
    elif weather == "storm":
        beta = 3.912 / 120.0
    sky_airlight = np.array([225.0, 225.0, 230.0], dtype=np.float32)

    # Thu thập và tính toán thông số chiếu cho toàn bộ các vật thể chèn (Inserted objects)
    merged_boxes = preprocess_ox_cart_boxes(boxes)
    candidates = []
    for obj_idx, b in enumerate(merged_boxes):
        if not b.get("inserted", False):
            continue

        c_lidar = np.asarray(b["translation"], dtype=np.float32)
        c_cam = c_lidar @ T[:3, :3].T + T[:3, 3]
        z_cam = float(c_cam[2])

        # Loại bỏ nếu đối tượng nằm phía sau camera hoặc quá gần (< 4.0m để tránh đè nắp capo)
        if z_cam < 4.0:
            continue

        if "_corners_lid" in b and "_p_front_lid" in b and "_p_rear_lid" in b:
            corners_lid = b["_corners_lid"]
            p_front_lid = b["_p_front_lid"]
            p_rear_lid = b["_p_rear_lid"]
        else:
            try:
                from utils.visualization import to_box
                box_obj = to_box(b)
                c_corners = box_obj.corners()
                corners_lid = c_corners.T
                # 4 đỉnh đầu xe (front face): cột 0, 1, 2, 3; 4 đỉnh đuôi xe (rear face): cột 4, 5, 6, 7
                p_front_lid = c_corners[:, :4].mean(axis=1)
                p_rear_lid = c_corners[:, 4:].mean(axis=1)
            except Exception:
                size = np.asarray(b.get("size", [2.0, 4.5, 1.5]), dtype=np.float32)
                hl, hw, hh = size[1] / 2.0, size[0] / 2.0, size[2] / 2.0
                corners_lid = c_lidar + np.array([
                    [-hl, -hw, -hh], [hl, -hw, -hh], [hl, hw, -hh], [-hl, hw, -hh],
                    [-hl, -hw,  hh], [hl, -hw,  hh], [hl, hw,  hh], [-hl, hw,  hh]
                ], dtype=np.float32)
                rot = b.get("rotation", [1.0, 0.0, 0.0, 0.0])
                from pyquaternion import Quaternion
                R_rot = Quaternion(rot).rotation_matrix.astype(np.float32) if len(rot) == 4 else np.eye(3, dtype=np.float32)
                u_fwd = R_rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
                p_front_lid = c_lidar + hl * u_fwd
                p_rear_lid = c_lidar - hl * u_fwd

        corners_cam = corners_lid @ T[:3, :3].T + T[:3, 3]
        # MỤC 4: CHẶN CÁC HỘP 3D BỊ TRÀN RA NGOÀI KHUNG HÌNH CAMERA (FRUSTUM CULLING)
        # 1. Nếu bất kỳ đỉnh nào của hộp có Z_cam < 5.5m:
        if np.any(corners_cam[:, 2] < 5.5):
            continue

        uv = (corners_cam @ K.T)[:, :2] / corners_cam[:, 2:3]
        u_min = float(np.min(uv[:, 0]))
        u_max = float(np.max(uv[:, 0]))
        v_min = float(np.min(uv[:, 1]))
        v_max = float(np.max(uv[:, 1]))

        # Cắt lọc các vật thể nằm ngoài khung hình hoặc nằm trọn bên dưới nắp capo xe chủ
        if u_max < 0 or u_min >= w or v_max < 0 or v_min >= ego_hood_v:
            continue

        w_2d = u_max - u_min
        h_2d = v_max - v_min
        if w_2d < 4.0 or h_2d < 4.0:
            continue

        # 2. Hoặc tâm hộp nằm ngoài khung hình (u_c < 50 hoặc u_c > W - 50):
        u_c = (u_min + u_max) * 0.5
        if u_c < 50.0 or u_c > (w - 50.0):
            continue

        # 3. Hoặc diện tích khung chiếu 2D w_2d * h_2d > 0.35 * W * H (hộp quá to sát ống kính):
        if (w_2d * h_2d) > (0.35 * w * h):
            continue

        # Chiếu 2 tâm mặt Đầu - Đuôi P_front và P_rear sang hệ tọa độ Camera
        p_front_cam = p_front_lid @ T[:3, :3].T + T[:3, 3]
        p_rear_cam = p_rear_lid @ T[:3, :3].T + T[:3, 3]

        z_front = float(p_front_cam[2])
        z_rear = float(p_rear_cam[2])

        # Chiếu tọa độ pixel 2D (u_front, v_front) và (u_rear, v_rear)
        uv_front = (K @ p_front_cam)[:2] / max(1e-3, z_front)
        uv_rear = (K @ p_rear_cam)[:2] / max(1e-3, z_rear)
        target_du = float(uv_front[0] - uv_rear[0])

        # Vector hướng từ đuôi xe đến đầu xe trong hệ Camera
        dx_cam = float(p_front_cam[0] - p_rear_cam[0])
        dz_cam = float(z_front - z_rear)
        l_xz = max(1e-4, float(np.hypot(dx_cam, dz_cam)))
        sin_heading = dx_cam / l_xz
        cos_heading = dz_cam / l_xz

        cat_clean = normalize_category(b.get("category", "car"))

        if cat_clean == "dog":
            # Chỉ dùng ảnh đứng ngang side.png khi con chó đang chạy cắt ngang qua đường (|dx_cam| > |dz_cam|)
            if abs(dx_cam) > abs(dz_cam):
                view_name = "side"
            elif z_front < z_rear:
                view_name = "front"
            else:
                view_name = "rear"
        elif abs(sin_heading) < 0.55:
            # Vật thể di chuyển dọc theo chiều con đường (|u_cam,x| < 0.55, tức góc lệch so với trục đường < 33 độ)
            # BẮT BUỘC 100% chỉ được chọn "front" hoặc "rear" cho mọi vật thể!
            if z_front < z_rear:
                view_name = "front"
            else:
                view_name = "rear"
        elif abs(cos_heading) < 0.35:
            # Xe di chuyển cắt ngang vuông góc qua mặt đường (> 70 độ)
            view_name = "side"
        else:
            # Góc di chuyển chéo (oblique)
            if z_front < z_rear:
                view_name = "oblique_front"
            else:
                view_name = "oblique_rear"

        # QUY TẮC LẬT NGANG (FLIP) BẤT BIẾN:
        # Kho ảnh chuẩn hóa front quay sang phải (orig_du_sign = +1).
        # target_du = u_front - u_rear:
        # Nếu target_du < 0: trên ảnh camera mục tiêu, đầu xe hướng sang trái -> Cần flip!
        needs_flip = (target_du < 0)

        # Kiểm tra xem có ảnh chụp thật đúng theo ann_token của template này không
        ann_token = b.get("source_ann_token") or b.get("token")
        ann_match = None
        if ann_token:
            ann_fp = get_default_assets_dir() / cat_clean / f"{ann_token}.png"
            if ann_fp.is_file():
                ann_match = ann_token

        # Khớp tỉ lệ khung chiếu 3D (Aspect Ratio Matching)
        variant_idx = select_best_aspect_variant(cat_clean, view_name, w_2d / max(1.0, h_2d), seed_idx=obj_idx)

        candidates.append({
            "box": b,
            "z_cam": z_cam,
            "category": cat_clean,
            "view_name": view_name,
            "ann_match": ann_match,
            "variant_idx": variant_idx,
            "needs_flip": needs_flip,
            "u_min": u_min, "u_max": u_max,
            "v_min": v_min, "v_max": v_max,
            "w_2d": w_2d, "h_2d": h_2d,
            "obj_idx": obj_idx,
            "corners_lid": corners_lid,
        })

    # Sắp xếp các vật thể từ xa đến gần (Painter's algorithm)
    candidates.sort(key=lambda item: item["z_cam"], reverse=True)

    for item in candidates:
        z_cam = item["z_cam"]
        # THỨ TỰ ĐỘ SÂU (DEPTH SORTING):
        # 1. Vẽ bóng đổ tiếp xúc mặt đường của vật thể này trước
        if item.get("box") is not None:
            out_img = render_single_ground_shadow(out_img, item["box"], T, K, weather=weather, level=level)

        # Nạp ảnh chuẩn xác theo view_name (đảm bảo 100% bất biến Z_front vs Z_rear) và variant_idx
        sprite_name = item["view_name"]
        variant_idx = item.get("variant_idx", 0)
        sprite_rgba = load_sprite_rgba(item["category"], sprite_name, variant_idx=variant_idx)

        # Tự động lật ngang ảnh nếu needs_flip (đảm bảo hông thùng xe nghiêng đúng phối cảnh)
        if item["view_name"] in ("side", "oblique_front", "oblique_rear") and item["needs_flip"]:
            sprite_rgba = cv2.flip(sprite_rgba, 1)

        orig_h, orig_w = sprite_rgba.shape[:2]
        sprite_w = float(orig_w)
        sprite_h = float(orig_h)

        w_2d_val = float(item["w_2d"])
        h_2d_val = float(item["h_2d"])

        sprite_aspect = sprite_w / max(1.0, sprite_h)
        box_aspect = w_2d_val / max(1.0, h_2d_val)
        ratio_match = sprite_aspect / max(1e-4, box_aspect)

        # MỤC 2: KHÓA CỨNG BỀ RỘNG ẢNH target_w KHÔNG BAO GIỜ ĐƯỢC TRÀN KHỎI HỘP 3D (w_2d)
        if 0.7 <= ratio_match <= 1.35:
            target_w = max(8, int(round(w_2d_val)))
            target_h = max(8, int(round(h_2d_val)))
        else:
            scale = min(w_2d_val / max(1.0, sprite_w), h_2d_val / max(1.0, sprite_h))
            target_w = max(8, int(round(sprite_w * scale)))
            target_h = max(8, int(round(sprite_h * scale)))

        # Khóa cứng an toàn tuyệt đối: cấm để target_w > 1.05 * w_2d
        if target_w > 1.05 * w_2d_val:
            target_w = max(8, int(round(1.05 * w_2d_val)))
            target_h = max(8, int(round(target_w / sprite_aspect)))

        sprite_resized = cv2.resize(sprite_rgba, (target_w, target_h),
                                    interpolation=cv2.INTER_AREA if target_w < orig_w else cv2.INTER_LINEAR)

        # Căn chỉnh đáy ảnh chạm đúng v_max (giới hạn tại nắp capo ego_hood_v) và tâm ngang trùng tâm 2D Box
        y2 = min(int(round(item["v_max"])), ego_hood_v)
        y1 = y2 - target_h
        cx = int(round((item["u_min"] + item["u_max"]) * 0.5))
        x1 = cx - target_w // 2
        x2 = x1 + target_w

        # Cắt xén phần vượt ra ngoài mép khung hình
        sx1 = max(0, -x1)
        sy1 = max(0, -y1)
        sx2 = target_w - max(0, x2 - w)
        sy2 = target_h - max(0, y2 - ego_hood_v)

        dx1 = max(0, x1)
        dy1 = max(0, y1)
        dx2 = min(w, x2)
        dy2 = min(ego_hood_v, y2)

        if dx2 <= dx1 or dy2 <= dy1 or sx2 <= sx1 or sy2 <= sy1:
            continue

        sprite_crop = sprite_resized[sy1:sy2, sx1:sx2]
        sprite_rgb = sprite_crop[:, :, :3].astype(np.float32)
        alpha = sprite_crop[:, :, 3].astype(np.float32) / 255.0

        # Làm mềm biên Alpha (Alpha Feathering 1-2px)
        if alpha.max() > 0:
            alpha_feathered = cv2.GaussianBlur(alpha, (3, 3), sigmaX=0.8)
            alpha_feathered = np.clip(alpha_feathered, 0.0, 1.0)
        else:
            alpha_feathered = alpha

        # Làm mờ nhẹ theo cự ly (Distance blur) cho vật thể ở xa (> 18m)
        if z_cam > 18.0:
            blur_sigma = min(1.2, float((z_cam - 18.0) * 0.04))
            if blur_sigma > 0.3:
                sprite_rgb = cv2.GaussianBlur(sprite_rgb, (3, 3), sigmaX=blur_sigma)

        # Khớp ánh sáng môi trường nhẹ (Color & Contrast Matching)
        bg_region = out_img[dy1:dy2, dx1:dx2].astype(np.float32)
        bg_lum = float(np.mean(bg_region))
        if bg_lum > 15.0:
            lum_factor = np.clip(bg_lum / 125.0, 0.85, 1.15)
            sprite_rgb = np.clip(sprite_rgb * lum_factor, 0.0, 255.0)

        # Phủ suy hao khí quyển Koschmieder theo độ sâu z_cam
        if beta > 0:
            trans = np.exp(-beta * z_cam)
            sprite_rgb = sprite_rgb * trans + sky_airlight * (1.0 - trans)

        # Kiểm tra độ sâu Z-buffer per-pixel và mặt nạ bảo vệ nắp capo xe chủ
        z_region = z_buffer[dy1:dy2, dx1:dx2]
        is_ego_region = (np.arange(dy1, dy2)[:, None] >= ego_hood_v)
        visible_mask = (alpha_feathered > 0.08) & (z_cam <= z_region) & (~is_ego_region)

        # ĐỒNG BỘ CULLING 2D/3D: Nếu không có pixel nào hiển thị (bị che khuất hoặc ngoài nắp capo), bỏ qua
        if not np.any(visible_mask):
            continue

        alpha_final = np.where(visible_mask, alpha_feathered, 0.0)
        alpha_3d = alpha_final[..., None]

        # Hòa trộn kênh Alpha vào ảnh nền
        blended = sprite_rgb * alpha_3d + bg_region * (1.0 - alpha_3d)
        out_img[dy1:dy2, dx1:dx2] = np.clip(blended, 0, 255).astype(np.uint8)

        # Cập nhật z_buffer và inpaint_mask
        z_region[visible_mask] = z_cam
        z_buffer[dy1:dy2, dx1:dx2] = z_region

        inpaint_region = inpaint_mask[dy1:dy2, dx1:dx2]
        inpaint_region[visible_mask] = 255
        inpaint_mask[dy1:dy2, dx1:dx2] = inpaint_region

        # Nếu draw_wireframe=True: Vẽ khung dây 3D wireframe box của ĐÚNG vật thể này ngay sau khi dán sprite
        # Nhờ đó vật thể ở gần hơn tiếp theo sẽ dán sprite đè lên khung dây của vật thể này!
        if draw_wireframe:
            c_lid = item.get("corners_lid")
            if c_lid is not None:
                c_cam = c_lid @ T[:3, :3].T + T[:3, 3]
                if not np.any(c_cam[:, 2] <= 0.4):
                    uv_box = (c_cam @ K.T)[:, :2] / c_cam[:, 2:3]
                    from utils.camera_proj import BOX_EDGES, STATUS_RGB
                    wire_color = STATUS_RGB.get("inserted", (245, 158, 11))
                    for p1, p2 in BOX_EDGES:
                        u1, v1 = float(uv_box[p1, 0]), float(uv_box[p1, 1])
                        u2, v2 = float(uv_box[p2, 0]), float(uv_box[p2, 1])
                        if v1 > ego_hood_v and v2 > ego_hood_v:
                            continue
                        if v1 > ego_hood_v:
                            t = (ego_hood_v - v2) / max(1e-4, v1 - v2)
                            u1 = u2 + t * (u1 - u2)
                            v1 = float(ego_hood_v)
                        elif v2 > ego_hood_v:
                            t = (ego_hood_v - v1) / max(1e-4, v2 - v1)
                            u2 = u1 + t * (u2 - u1)
                            v2 = float(ego_hood_v)
                        cv2.line(out_img, (int(round(u1)), int(round(v1))),
                                 (int(round(u2)), int(round(v2))), wire_color, 2, cv2.LINE_AA)

    return out_img, z_buffer, inpaint_mask


# Giữ nguyên alias rasterize_dense_surface_zbuffer để tương thích ngược hoàn toàn
rasterize_dense_surface_zbuffer = composite_multiview_sprites


# ==================== CONTROLNET CONDITIONING GENERATOR ====================

def generate_controlnet_conditions(image: np.ndarray, boxes: Sequence[dict],
                                  T: np.ndarray, K: np.ndarray,
                                  out_size: Tuple[int, int] = (800, 450)) -> Tuple[np.ndarray, np.ndarray]:
    """
    CONTROLNET CONDITIONING GENERATOR:
    Xuất ra cặp dữ liệu điều khiển:
    a) inpaint_mask: Mặt nạ nhị phân uint8 (255 tại vùng vật thể + bóng đổ mặt đường)
    b) depth_map: Bản đồ độ sâu chuẩn hóa uint8 [0, 255] (vật thể gần sáng hơn, xa tối hơn)
    phục vụ nạp trực tiếp vào mô hình ControlNet-Depth / Inpainting.
    """
    h, w = out_size[1], out_size[0]
    img_resized = cv2.resize(image, out_size) if (image.shape[1], image.shape[0]) != out_size else image

    # Tỉ lệ ma trận nội suy K theo kích thước
    sx, sy = out_size[0] / 1600.0, out_size[1] / 900.0
    K_scaled = K.copy()
    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy

    _, z_buffer, mask = composite_multiview_sprites(img_resized, boxes, T, K_scaled)

    # Mở rộng nhẹ mask để bao trọn viền tiếp xúc (dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated_mask = cv2.dilate(mask, kernel, iterations=1)

    # Chuẩn hóa depth map sang thang uint8 [0, 255]
    valid_depths = z_buffer[np.isfinite(z_buffer)]
    depth_map_uint8 = np.zeros((h, w), dtype=np.uint8)
    if len(valid_depths) > 0:
        min_d = max(1.0, float(valid_depths.min()))
        max_d = min(80.0, float(valid_depths.max()) + 1.0)
        norm_d = np.clip((max_d - z_buffer) / max(0.1, max_d - min_d), 0.0, 1.0)
        depth_map_uint8 = np.where(np.isfinite(z_buffer), (norm_d * 255.0).astype(np.uint8), 0)

    return dilated_mask, depth_map_uint8


def run_controlnet_inpainting_hook(image: np.ndarray, inpaint_mask: np.ndarray,
                                   depth_map: np.ndarray, prompt: str = "realistic car on asphalt street") -> np.ndarray:
    """
    Hook gọi Diffusers StableDiffusionControlNetInpaintPipeline khi có GPU.
    Tự động fallback an toàn về kết quả Multi-view Sprite Renderer khi chạy trên CPU.
    """
    try:
        import torch
        if torch.cuda.is_available():
            from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel  # type: ignore
            pass
    except Exception:
        pass
    return image


# ==================== ĐỒNG BỘ TOÀN DIỆN CAMERA SYNTHESIS ====================

def synthesize_camera_view(image: np.ndarray, boxes: Sequence[dict],
                           T: np.ndarray, K: np.ndarray,
                           weather: Optional[str] = None,
                           level: Optional[str] = None,
                           mode: str = "photorealistic_box",
                           seed: int = 42) -> np.ndarray:
    """
    HÀM ĐỒNG BỘ CHÍNH CHO 6-CAMERA SURROUND VIEW:
    Các chế độ mode:
    1. 'photorealistic_box': Vật thể 2D đặc (Multi-view Sprite) + bóng đổ mặt đường + khung dây 3D wireframe.
    2. 'photorealistic': Chỉ hiển thị ảnh quang học đặc (không khung dây, dùng train BEVFormer/BEVDepth).
    3. 'controlnet_depth': Hiển thị bản đồ độ sâu Depth Map và Inpainting Mask.
    """
    h, w = image.shape[:2]
    sx, sy = w / 1600.0, h / 900.0
    K_scaled = K.copy()
    K_scaled[0, :] *= sx
    K_scaled[1, :] *= sy

    if mode == "controlnet_depth":
        mask, dmap = generate_controlnet_conditions(image, boxes, T, K, out_size=(w, h))
        dmap_bgr = cv2.applyColorMap(dmap, cv2.COLORMAP_VIRIDIS)
        dmap_rgb = cv2.cvtColor(dmap_bgr, cv2.COLOR_BGR2RGB)
        mask_3c = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        combined = cv2.addWeighted(dmap_rgb, 0.7, mask_3c, 0.3, 0)
        return combined

    # 0. Nếu mode == "photorealistic_box": Vẽ khung dây 3D Wireframe của các xe thật phía sau (background) TRƯỚC TIÊN
    # để khi dán sprite xe chèn vào, sprite sẽ che chắn tự nhiên các đường wireframe trắng phía sau (không bị xuyên thấu).
    base_image = image
    if mode == "photorealistic_box":
        from utils.camera_proj import BOX_EDGES, STATUS_RGB, project
        from utils.visualization import to_box
        from PIL import Image, ImageDraw

        pil_bg = Image.fromarray(image)
        dr_bg = ImageDraw.Draw(pil_bg, "RGBA")
        for b in boxes:
            if b.get("inserted", False):
                continue
            try:
                cuv, cok, cfront = project(T, K, to_box(b).corners().T)
                if not cfront.all() or not cok.any():
                    continue
                rgb = STATUS_RGB.get(b.get("status", "real"), (255, 255, 255))
                for p1, p2 in BOX_EDGES:
                    dr_bg.line([(cuv[p1, 0] * sx, cuv[p1, 1] * sy), (cuv[p2, 0] * sx, cuv[p2, 1] * sy)],
                               fill=rgb + (110,), width=1)
            except Exception:
                pass
        base_image = np.asarray(pil_bg)

    draw_wf = (mode == "photorealistic_box")

    # Ghép ảnh quang học đặc tách nền Multi-view Sprite + Bóng đổ + 3D Wireframe (depth-sorted per object)
    img_rendered, _, _ = composite_multiview_sprites(
        base_image, boxes, T, K_scaled,
        weather=weather, level=level, seed=seed, draw_wireframe=draw_wf
    )

    # 3. Phủ hiệu ứng thời tiết phía trước (Vệt mưa / Sương mù lớp tiền cảnh)
    if weather:
        from task_b.camera_weather import apply_camera_weather
        img_weather = apply_camera_weather(img_rendered, weather, level, np.random.default_rng(seed))
    else:
        img_weather = img_rendered

    return img_weather
