"""Vật thể VN không có trong nuScenes (xe ba gác, xe máy chở hàng, động vật...):
Dựng hình từ khối cơ bản theo kích thước công bố / giả định, kết hợp cơ chế
HYBRID MESH / TEMPLATE RAY-CASTING:
- Kết hợp khung xe máy / người thật từ Template Bank với các phần mở rộng đặc thù VN
  (hàng cồng kềnh 1.0-1.8m, chở 2-3 người, xe ba gác, xe lôi, hàng rong).
- Tính toán pháp tuyến bề mặt (surface normals) và góc tới laser cos(theta).
- Mô phỏng độ nhám micro-roughness và suy hao phản xạ vật lý theo định luật Lambert.
- Tự động tích hợp vào luồng 10-sweep velocity compensation và Multi-Frame Tracking.

Cảm biến: Velodyne HDL-32E (LIDAR_TOP) — 32 tia dọc, góc ngước theo thông số (HDL32E_ELEVATIONS, 1.33°),
~1084 bước azimuth / vòng. Mỗi tia lấy điểm chạm gần nhất => mật độ theo khoảng cách, mặt nhìn thấy, tự che.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.ground_plane import project_z_to_ground  # noqa: E402
from task_a.merge import points_in_new_box, resolve_occlusion  # noqa: E402
from utils.transforms import yaw_to_quaternion  # noqa: E402

HDL32E_ELEVATIONS = np.radians(np.linspace(-30.67, 10.67, 32))  # 32 tia đều 1.33° (thông số Velodyne HDL-32E)
RANGE_NOISE = 0.02     # (m) sai số đo khoảng cách HDL-32E ~ 2 cm
DROPOUT = 0.03         # tỉ lệ tia không trả về cơ bản (bề mặt xiên, vật liệu tối)

# Intensity tham chiếu (thang 0-255 nuScenes) theo từng loại vật liệu:
REFL = {
    "metal": 35.0,     # Kim loại lốc máy, khung xe
    "paint": 26.0,     # Vỏ sơn xe
    "cloth": 11.0,     # Vải quần áo, bạt phủ hàng
    "fur": 7.0,        # Lông động vật (hấp thụ laser mạnh)
    "wood": 14.0,      # Gỗ thùng xe, xe bò
    "tire": 5.0,       # Cao su lốp xe (đen, hấp thụ cao)
    "plastic": 16.0,   # Nhựa yếm xe, nón bảo hiểm
    "glass": 5.0       # Kính phản xạ gương / khúc xạ
}

# Tỉ lệ tia không trả về theo vật liệu:
DROP = {"glass": 0.8, "tire": 0.3, "fur": 0.1, "cloth": 0.05}


def _box(c, l, w, h, mat):
    return ("box", np.asarray(c, float), np.array([l, w, h], float), REFL[mat], DROP.get(mat, 0.0))


def _ell(c, rx, ry, rz, mat):
    return ("ell", np.asarray(c, float), np.array([rx, ry, rz], float), REFL[mat], DROP.get(mat, 0.0))


# ----- Hình các chi tiết vật thể (z tính từ mặt đất, chuyển về gốc tâm box ở build_model) -----
def _wheels(x_front, x_back, r=0.3, y=0.0, thick=0.06):
    return [_ell((x, y, r), r, thick, r, "tire") for x in (x_front, x_back)]


def _rider(x, seat_z=0.75, mat="cloth"):
    return [
        _box((x + 0.15, 0, seat_z - 0.25), 0.35, 0.45, 0.45, mat),       # Chân / đùi
        _ell((x, 0, seat_z + 0.32), 0.17, 0.22, 0.32, mat),              # Thân người
        _ell((x + 0.03, 0, seat_z + 0.78), 0.12, 0.11, 0.13, "plastic")  # Đầu + mũ bảo hiểm
    ]


def _moto_frame():
    """
    Khung xe máy chi tiết cao (Honda Wave Alpha / Vision):
    Bao gồm bánh xe, ốp thân, yếm, ghi-đông, lốc máy và ống xả kim loại.
    """
    return (
        _wheels(0.66, -0.66, 0.28, 0.0, 0.08)
        + [
            _box((-0.15, 0, 0.58), 0.95, 0.58, 0.42, "plastic"),      # Ốp thân + yên xe
            _box((0.18, 0, 0.30), 0.55, 0.45, 0.12, "plastic"),       # Sàn để chân
            _box((0.48, 0, 0.62), 0.22, 0.62, 0.62, "plastic"),       # Yếm xe (leg shield)
            _box((0.60, 0, 0.98), 0.25, 0.45, 0.18, "plastic"),       # Đầu đèn
            _box((0.58, 0, 1.06), 0.08, 0.69, 0.06, "metal"),         # Tay lái (ghi-đông)
            # Các chi tiết kim loại tăng độ tương phản LiDAR:
            _box((0.05, 0.0, 0.30), 0.38, 0.28, 0.22, "metal"),       # Lốc máy
            _box((-0.25, 0.22, 0.26), 0.65, 0.10, 0.10, "metal")      # Ống xả (bên phải)
        ]
    )


def shape(name: str, rng: Optional[np.random.Generator] = None) -> List[tuple]:
    """Tạo danh sách primitives (z tính từ mặt đất) của vật thể."""
    rng = rng or np.random.default_rng(0)
    if name == "car":  # Ô tô 4-5 chỗ ~ Toyota Vios
        return (
            _wheels(1.35, -1.35, 0.31, 0.75, 0.10) + _wheels(1.35, -1.35, 0.31, -0.75, 0.10)
            + [
                _box((0, 0, 0.62), 4.40, 1.72, 0.66, "paint"),
                _box((-0.25, 0, 1.18), 2.30, 1.50, 0.58, "glass"),       # Cabin kính
                _box((-0.25, 0, 1.45), 2.00, 1.40, 0.05, "paint")        # Nóc xe
            ]
        )
    if name == "moto_1":
        return _moto_frame() + _rider(-0.10)
    if name == "moto_multi":  # Xe máy kẹp 2-3 người nối tiếp
        n = int(rng.integers(2, 4))
        return _moto_frame() + sum((_rider(x) for x in (0.0, -0.40, -0.75)[:n]), [])
    if name == "moto_cargo":  # Xe máy chở hàng cồng kềnh buộc sau yên (rộng 1.0-1.8m, cao tới 2m)
        w, h, l = float(rng.uniform(1.0, 1.8)), float(rng.uniform(0.7, 1.1)), float(rng.uniform(0.7, 1.1))
        return _moto_frame() + _rider(0.05) + [_box((-0.55, 0, 0.85 + h / 2), l, w, h, "cloth")]
    if name == "ba_gac":  # Xe ba gác Nam Định: đầu xe máy + thùng sau hở 2.6 x 1.7m
        hh = float(rng.uniform(0.3, 0.9))
        return (
            _wheels(1.65, 1.65) + _wheels(-1.2, -1.2, 0.28, 0.75) + _wheels(-1.2, -1.2, 0.28, -0.75)
            + _rider(1.25)
            + [
                _box((1.45, 0, 0.55), 0.8, 0.34, 0.4, "plastic"),
                _box((-0.6, 0, 0.62), 2.6, 1.70, 0.12, "metal"),      # Sàn thùng
                _box((-0.6, 0.83, 0.85), 2.6, 0.04, 0.35, "metal"),   # Thành trái
                _box((-0.6, -0.83, 0.85), 2.6, 0.04, 0.35, "metal"),  # Thành phải
                _box((-0.6, 0, 0.68 + hh / 2), 2.2, 1.5, hh, "cloth")  # Hàng trên thùng
            ]
        )
    if name == "moto_tow":  # Xe máy kéo thùng hàng / xe lôi
        return (
            [_box(p[1] + [1.0, 0, 0], *p[2], "plastic") if p[0] == "box" else
             _ell(p[1] + [1.0, 0, 0], *p[2], "tire") for p in _moto_frame()]
            + _rider(0.9) + _wheels(-0.9, -0.9, 0.25, 0.55) + _wheels(-0.9, -0.9, 0.25, -0.55)
            + [_box((-0.9, 0, 0.75), 1.8, 1.2, 0.95, "wood")]
        )
    if name == "vendor_cart":  # Xe đẩy hàng rong 1.5 x 0.8m + mái che + người đẩy
        return (
            _wheels(0.5, -0.5, 0.2, 0.35) + _wheels(0.5, -0.5, 0.2, -0.35)
            + [
                _box((0, 0, 0.75), 1.5, 0.8, 0.5, "wood"),
                _box((0, 0, 1.75), 1.6, 1.0, 0.05, "plastic"),
                _box((0.7, 0.45, 1.25), 0.04, 0.04, 1.0, "metal"),
                _box((-0.7, -0.45, 1.25), 0.04, 0.04, 1.0, "metal"),
                _ell((-1.05, 0, 1.05), 0.17, 0.22, 0.32, "cloth"),
                _ell((-1.05, 0, 1.52), 0.11, 0.11, 0.12, "cloth"),
                _box((-1.05, 0, 0.42), 0.25, 0.35, 0.84, "cloth")
            ]
        )
    if name == "dog":  # Chó chạy qua đường (0.9 x 0.3 x 0.6m)
        return (
            [_ell((0, 0, 0.42), 0.33, 0.13, 0.14, "fur"), _ell((0.40, 0, 0.52), 0.11, 0.09, 0.09, "fur")]
            + [_box((x, y, 0.16), 0.06, 0.06, 0.32, "fur") for x in (0.22, -0.22) for y in (0.08, -0.08)]
        )
    if name == "cow":  # Trâu / bò đi trên đường (2.2 x 0.7 x 1.4m)
        return (
            [_ell((0, 0, 1.0), 0.85, 0.33, 0.36, "fur"), _ell((0.95, 0, 1.15), 0.22, 0.14, 0.18, "fur")]
            + [_box((x, y, 0.33), 0.12, 0.12, 0.66, "fur") for x in (0.6, -0.6) for y in (0.18, -0.18)]
        )
    if name == "ox_wagon":  # Xe bò kéo: xe gỗ 2.5 x 1.5m
        return (
            _wheels(0.0, 0.0, 0.5, 0.75, 0.08) + _wheels(0.0, 0.0, 0.5, -0.75, 0.08)
            + [
                _box((0, 0, 0.75), 2.5, 1.4, 0.10, "wood"),
                _box((0, 0, 0.95), 2.5, 1.5, 0.35, "wood"),
                _box((1.6, 0, 0.75), 0.9, 0.08, 0.06, "wood")
            ]
        )
    raise KeyError(name)


SHAPES = ("car", "moto_1", "moto_multi", "moto_cargo", "ba_gac", "moto_tow", "vendor_cart", "dog", "cow", "ox_wagon")


def build_model(name: str, rng: Optional[np.random.Generator] = None, category: Optional[str] = None) -> dict:
    """Tạo template ray-cast cơ bản: primitives trong object frame và kích thước bao [w, l, h]."""
    prims = shape(name, rng)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for kind, c, s, _, _ in prims:
        half = s / 2 if kind == "box" else s
        lo, hi = np.minimum(lo, c - half), np.maximum(hi, c + half)
    lo[2] = 0.0  # Đáy box chạm mặt đất z = 0
    center = (lo + hi) / 2
    size_lwh = hi - lo
    prims = [(k, c - center, s, refl, drop) for k, c, s, refl, drop in prims]
    return {
        "name": name,
        "category": category or f"vn.{name}",
        "primitives": prims,
        "size": [float(size_lwh[1]), float(size_lwh[0]), float(size_lwh[2])],
        "ann_token": f"raycast_{name}_{id(prims)}",
        "points": np.zeros((0, 5), np.float32),
        "distance": None,
        "base_template": None
    }


def build_hybrid_model(name: str, base_template: Optional[dict] = None,
                       rng: Optional[np.random.Generator] = None, category: Optional[str] = None) -> dict:
    """
    CƠ CHẾ HYBRID TEMPLATE + PROCEDURAL RAYCAST:
    Kết hợp khung xe máy / người đi bộ thật từ nuScenes với phần mở rộng raycast đặc thù VN
    (khối hàng cồng kềnh cho moto_cargo, người thứ 2/3 cho moto_multi, thùng xe cho ba_gac).
    Khắc phục triệt để hiện tượng mô hình thuần elip/hộp chỉ đạt 64% recall so với 94% của xe thật.
    """
    rng = rng or np.random.default_rng()
    base_model = build_model(name, rng=rng, category=category)

    if base_template is not None and "points" in base_template and len(base_template["points"]) > 0:
        base_model["base_template"] = base_template
        # Cập nhật bounding box mở rộng để bao cả base template và phần raycast
        if "size" in base_template:
            bw, bl, bh = base_template["size"]
            mw, ml, mh = base_model["size"]
            base_model["size"] = [max(bw, mw), max(bl, ml), max(bh, mh)]

    return base_model


# ==================== DÒ TIA CHÍNH XÁC KÈM PHÁP TUYẾN & GÓC TỚI ====================

def _hit_box_with_normal(o, d, size):
    """
    Slab method tìm điểm chạm gần nhất và vector pháp tuyến bề mặt (normal)
    để tính góc tới laser cos(theta).
    """
    half = size / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = (-half - o) / d
        t2 = (half - o) / d
    tmin_xyz = np.minimum(t1, t2)
    tmax_xyz = np.maximum(t1, t2)
    tmin = np.nanmax(tmin_xyz, axis=1)
    tmax = np.nanmin(tmax_xyz, axis=1)
    valid = (tmax >= tmin) & (tmax > 0)
    t = np.where(valid, np.where(tmin > 0, tmin, np.inf), np.inf)

    # Pháp tuyến tại mặt chạm
    hit_face = (tmin_xyz == tmin[:, None]) & np.isfinite(t)[:, None]
    normal = -np.sign(d) * hit_face.astype(float)
    norm = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6
    normal = normal / norm
    return t, normal


def _hit_ell_with_normal(o, d, radii):
    """Tìm điểm chạm trên ellipsoid và vector pháp tuyến gradient."""
    o_sc, d_sc = o / radii, d / radii
    a = (d_sc * d_sc).sum(1)
    b = 2 * (o_sc * d_sc).sum(1)
    c = (o_sc * o_sc).sum(-1) - 1
    disc = b * b - 4 * a * c
    t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
    valid = (disc >= 0) & (t > 0)
    t = np.where(valid, t, np.inf)

    # Pháp tuyến = gradient F(p) = 2 * p / radii^2
    p = o + np.where(np.isfinite(t), t, 0)[:, None] * d
    normal = p / (radii ** 2)
    norm = np.linalg.norm(normal, axis=1, keepdims=True) + 1e-6
    normal = np.where(np.isfinite(t)[:, None], normal / norm, 0.0)
    return t, normal


def raycast(model: dict, center: Sequence[float], yaw: float, ring_table: np.ndarray,
            rng: Optional[np.random.Generator] = None, n_az: int = config.LIDAR_AZIMUTH_BINS) -> np.ndarray:
    """
    Bắn tia laser từ gốc cảm biến (0, 0, 0) vào vật thể tại vị trí center, hướng yaw.
    Mô phỏng đầy đủ:
    1. Góc tới laser cos(theta) = |n . (-d)|
    2. Độ nhám micro-roughness và phản xạ Lambertian
    3. Tỷ lệ dropout phụ thuộc góc xiên (glancing angle)
    4. Cơ chế Hybrid nạp điểm thật từ base_template nếu có.
    """
    rng = rng or np.random.default_rng()
    center = np.asarray(center, float)
    w, l, h = model["size"]

    # Bán kính góc quét bao trọn vật thể
    r_xy = np.hypot(*center[:2])
    half_ang = np.arctan2(np.hypot(l, w) / 2 + 0.25, max(r_xy, 0.5))
    az0 = np.arctan2(center[1], center[0])
    step = 2 * np.pi / n_az
    az = (np.round(az0 / step) + np.arange(-int(half_ang / step) - 1, int(half_ang / step) + 2)) * step
    ring_idx, az_g = np.meshgrid(np.arange(len(ring_table)), az, indexing="ij")
    el = ring_table[ring_idx.ravel()]
    a = az_g.ravel()
    dirs = np.c_[np.cos(el) * np.cos(a), np.cos(el) * np.sin(a), np.sin(el)]

    # Chuyển tia sang hệ tọa độ cục bộ của vật thể (object frame)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    o_loc = (np.zeros(3) - center) @ R
    d_loc = dirs @ R

    best_t = np.full(len(dirs), np.inf)
    best_refl = np.zeros(len(dirs))
    best_drop = np.zeros(len(dirs))
    best_normal = np.zeros((len(dirs), 3))

    for kind, pc, size, refl, drop in model["primitives"]:
        if kind == "box":
            t, nml = _hit_box_with_normal(o_loc - pc, d_loc, size)
        else:
            t, nml = _hit_ell_with_normal(o_loc - pc, d_loc, size)

        closer = t < best_t
        best_t[closer] = t[closer]
        best_refl[closer] = refl
        best_drop[closer] = drop
        best_normal[closer] = nml[closer]

    # Tính góc tới cos(theta) giữa hướng tia và pháp tuyến
    cos_theta = np.clip(np.abs(np.sum(best_normal * (-d_loc), axis=1)), 0.05, 1.0)

    # Hiệu ứng góc xiên (glancing dropout): góc càng nghiêng thì laser càng dễ phân tán không quay lại
    glancing_drop = np.where(cos_theta < 0.15, 0.20, 0.0)
    effective_drop = np.maximum(DROPOUT, np.maximum(best_drop, glancing_drop))

    hit = np.isfinite(best_t) & (rng.random(len(dirs)) >= effective_drop)
    t = best_t[hit] + rng.normal(0, RANGE_NOISE, hit.sum())
    pts_procedural = dirs[hit] * t[:, None]

    # Cường độ phản xạ vật lý: Lambertian + nhiễu log-normal micro-roughness
    cos_term = 0.35 + 0.65 * cos_theta[hit]
    inten = np.clip(best_refl[hit] * cos_term * rng.lognormal(0, 0.28, hit.sum()), 0, 255)
    rings = ring_idx.ravel()[hit]
    procedural_cloud = np.c_[pts_procedural, inten, rings].astype(np.float32)

    # ==================== KẾT HỢP HYBRID VỚI BASE TEMPLATE ====================
    base_tpl = model.get("base_template")
    if base_tpl is not None and "points" in base_tpl and len(base_tpl["points"]) > 0:
        base_pts = np.asarray(base_tpl["points"], dtype=np.float32)
        # Chuyển base_pts từ object frame sang sensor frame
        # p_sensor = center + p_loc @ R.T
        base_xyz = center + base_pts[:, :3] @ R.T
        base_int = base_pts[:, 3:4]
        base_ring = base_pts[:, 4:5] if base_pts.shape[1] >= 5 else np.zeros((len(base_pts), 1), np.float32)

        # Lọc bỏ các điểm của xe thật bị phần mở rộng che khuất (vd hàng buộc sau che phần đuôi xe x < -0.2)
        if model["name"] == "moto_cargo":
            keep_base = (base_pts[:, 0] > -0.20) | (np.abs(base_pts[:, 1]) > 0.40)
        elif model["name"] == "moto_multi":
            keep_base = base_pts[:, 0] > -0.30
        elif model["name"] == "ba_gac":
            keep_base = base_pts[:, 0] > 0.30
        else:
            keep_base = np.ones(len(base_pts), dtype=bool)

        base_cloud = np.hstack([base_xyz[keep_base], base_int[keep_base], base_ring[keep_base]]).astype(np.float32)
        # Gộp cả 2 nguồn: Khung xe thật + hàng hoá / mở rộng raycast
        return np.concatenate([base_cloud, procedural_cloud], axis=0)

    return procedural_cloud


def insert_raycast(target_points: np.ndarray, model: dict, position: Tuple[float, float, float], ground,
                   rng: Optional[np.random.Generator] = None, **_) -> Tuple[np.ndarray, dict]:
    """
    Chèn vật thể ray-cast (hoặc Hybrid) vào đám mây điểm mục tiêu:
    - Sinh đám mây điểm tại đúng tọa độ, áp dụng mặt đất cục bộ.
    - Xóa các điểm nền nằm trong thể tích box (Ground Clutter Cleaner).
    - Khử che khuất hai chiều (Resolve Occlusion).
    """
    x, y, yaw = position
    h = model["size"][2]
    center = [x, y, project_z_to_ground(x, y, ground) + h / 2]

    placed = raycast(model, center, yaw, HDL32E_ELEVATIONS, rng).astype(target_points.dtype)
    box = {"translation": center, "size": model["size"], "rotation": yaw_to_quaternion(yaw)}
    in_box = points_in_new_box(target_points, box)
    target = target_points[~in_box]
    merged = np.concatenate([target, placed], axis=0)
    is_obj = np.r_[np.zeros(len(target), bool), np.ones(len(placed), bool)]
    keep = resolve_occlusion(merged, is_obj)

    info = {
        "n_template": int(len(placed)),
        "n_density_dropped": 0,
        "n_box_cleared": int(in_box.sum()),
        "n_target_occluded": int((~keep & ~is_obj).sum()),
        "n_object_occluded": int((~keep & is_obj).sum()),
        "n_visible": int((keep & is_obj).sum())
    }
    return merged[keep], info


def beam_table(n_rings: int, fov=(-30.67, 10.67)) -> np.ndarray:
    """Bảng góc ngước đều cho LiDAR giả định n_rings tia."""
    return np.radians(np.linspace(fov[0], fov[1], n_rings))


def points_vs_distance(name: str, distances: Sequence[float], ring_table: np.ndarray, n_az: int = 1084,
                       sensor_height: float = 1.84, rng=None) -> List[int]:
    """Đo số điểm rơi trên vật thể theo khoảng cách."""
    rng = rng or np.random.default_rng(0)
    m = build_model(name, np.random.default_rng(1))
    out = []
    for r in distances:
        pts = raycast(m, [r, 0, -sensor_height + m["size"][2] / 2], np.radians(45), ring_table, rng, n_az)
        out.append(int(len(pts)))
    return out


if __name__ == "__main__":
    import json
    # Kiểm tra nhanh số điểm theo khoảng cách
    hdl32 = beam_table(32)
    bins = [(3, 7), (7, 12), (12, 20), (20, 30), (30, 45)]
    for name in ("car", "moto_1", "moto_cargo", "ba_gac", "dog"):
        rc = points_vs_distance(name, [np.mean(b) for b in bins], hdl32)
        print(f"Shape {name:12s} points across distances {bins}: {rc}")
