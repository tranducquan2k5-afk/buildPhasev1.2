"""Unit tests cho 2D Photorealistic Sensor Fusion Synthesis & ControlNet Conditioning (task_a/camera_synthesis.py)."""
import numpy as np
import pytest
from pyquaternion import Quaternion

from task_a.camera_synthesis import (
    render_ground_contact_shadow,
    rasterize_dense_surface_zbuffer,
    generate_controlnet_conditions,
    synthesize_camera_view,
    _get_object_primitives,
    PALETTE
)
from utils.camera_proj import lidar_to_cam, render_camera


@pytest.fixture
def mock_cam_setup():
    """Tạo ma trận giả lập K và T_cam_lidar chuẩn nuScenes cho camera trước."""
    w, h = 800, 450
    # Camera nhìn thẳng về phía trước (trục Z là hướng nhìn, X sang phải, Y hướng xuống)
    K = np.array([
        [800.0, 0.0, 400.0],
        [0.0, 800.0, 225.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    # T_cam_lidar: chuyển từ LIDAR (X trước, Y trái, Z trên) sang Camera (X phải, Y xuống, Z trước)
    # LIDAR x -> CAM z, LIDAR y -> CAM -x, LIDAR z -> CAM -y
    R = np.array([
        [0.0, -1.0, 0.0],
        [0.0,  0.0, -1.0],
        [1.0,  0.0, 0.0]
    ], dtype=np.float32)
    t = np.array([0.0, 0.5, -0.8], dtype=np.float32)
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R
    T[:3, 3] = t

    image = np.full((h, w, 3), 180, dtype=np.uint8)  # Mặt đường xám đồng nhất

    # Vật thể xe ô tô chèn ở phía trước 9 mét (thỏa mãn cự ly an toàn >= 9.0m)
    box = {
        "translation": [9.0, 0.0, -0.5],
        "size": [1.9, 4.5, 1.6],  # w, l, h
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "category": "vehicle.car",
        "inserted": True,
        "status": "inserted"
    }
    return image, [box], T, K


def test_ground_contact_shadow(mock_cam_setup):
    """Kiểm tra bóng đổ tiếp xúc mặt đường làm tối vùng đường dưới đáy xe."""
    image, boxes, T, K = mock_cam_setup
    out_img = render_ground_contact_shadow(image, boxes, T, K, weather=None)

    # Ảnh phải giữ nguyên shape và dtype
    assert out_img.shape == image.shape
    assert out_img.dtype == np.uint8

    # Vùng có bóng đổ phải tối hơn màu mặt đường gốc (180)
    shadowed_pixels = out_img[out_img < 180]
    assert len(shadowed_pixels) > 100, "Bóng đổ tiếp xúc mặt đường phải có ít nhất 100 pixel bị làm tối"
    assert out_img.min() < 160, "Độ đậm của bóng đổ phải rõ nét (pixel tối < 160)"


def test_dense_surface_rasterization_and_zbuffer(mock_cam_setup):
    """Kiểm tra kết xuất bề mặt đặc Z-buffer: thay đổi pixel vật thể và lưu depth map hữu hạn."""
    image, boxes, T, K = mock_cam_setup
    out_img, z_buf, mask = rasterize_dense_surface_zbuffer(image, boxes, T, K)

    # 1. Pixel bên trong vùng vật thể được tô màu bề mặt (thay đổi so với ảnh nền 180)
    changed_pixels = np.any(out_img != image, axis=-1)
    assert np.sum(changed_pixels) > 200, "Vùng vật thể phải được phủ kín bởi các đa giác đặc"

    # 2. Z-buffer phải lưu độ sâu hữu hạn tương ứng với khoảng cách camera (~5-9m)
    finite_depths = z_buf[np.isfinite(z_buf)]
    assert len(finite_depths) > 200
    assert 4.0 <= finite_depths.min() <= 10.0, f"Độ sâu min {finite_depths.min()} phải khớp khoảng cách đặt xe (~5-8m)"

    # 3. Mask nhị phân/anti-aliased phải bao phủ toàn bộ các vị trí có độ sâu hữu hạn
    assert np.all(mask[np.isfinite(z_buf)] > 0), "Mọi pixel có độ sâu phải có mask > 0"
    assert np.sum(mask == 255) > 200, "Vùng trung tâm vật thể phải có mask = 255"
    assert np.all(mask[~np.isfinite(z_buf)] == 0), "Vùng không có vật thể phải có mask = 0"


def test_controlnet_conditioning_generator(mock_cam_setup):
    """Kiểm tra sinh điều kiện cho ControlNet: inpaint_mask và depth_map đúng kích thước và chuẩn uint8."""
    image, boxes, T, K = mock_cam_setup
    mask, depth_map = generate_controlnet_conditions(image, boxes, T, K, out_size=(800, 450))

    # Kích thước chuẩn
    assert mask.shape == (450, 800)
    assert depth_map.shape == (450, 800)
    assert mask.dtype == np.uint8
    assert depth_map.dtype == np.uint8

    # Mask có vùng 255 và vùng 0
    assert np.max(mask) == 255
    assert np.min(mask) == 0

    # Depth map chuẩn hóa: gần sáng hơn, xa tối hơn
    assert np.max(depth_map) > 100
    assert np.min(depth_map) == 0


def test_synthesize_camera_view_three_modes(mock_cam_setup):
    """Kiểm tra 3 chế độ hiển thị: photorealistic_box, photorealistic, controlnet_depth."""
    image, boxes, T, K = mock_cam_setup

    # Mode 1: Photorealistic + 3D Box
    res_box = synthesize_camera_view(image, boxes, T, K, mode="photorealistic_box")
    assert res_box.shape == (450, 800, 3)
    assert res_box.dtype == np.uint8

    # Mode 2: Chỉ Photorealistic (không khung dây để train vision BEV)
    res_photo = synthesize_camera_view(image, boxes, T, K, mode="photorealistic")
    assert res_photo.shape == (450, 800, 3)
    assert res_photo.dtype == np.uint8
    # Phải có sự khác biệt giữa có wireframe và không có wireframe
    assert not np.array_equal(res_box, res_photo)

    # Mode 3: ControlNet Depth & Mask View
    res_ctrl = synthesize_camera_view(image, boxes, T, K, mode="controlnet_depth")
    assert res_ctrl.shape == (450, 800, 3)
    assert res_ctrl.dtype == np.uint8


def test_weather_layering(mock_cam_setup):
    """Kiểm tra phân tầng thời tiết: mưa hoặc sương mù phủ lên vật thể mới chèn."""
    image, boxes, T, K = mock_cam_setup

    img_clear = synthesize_camera_view(image, boxes, T, K, weather=None, mode="photorealistic")
    img_rain = synthesize_camera_view(image, boxes, T, K, weather="rain", level="heavy", mode="photorealistic")
    img_fog = synthesize_camera_view(image, boxes, T, K, weather="fog", level="dense", mode="photorealistic")

    # Hiệu ứng thời tiết phải tác động lên ảnh
    assert not np.array_equal(img_clear, img_rain)
    assert not np.array_equal(img_clear, img_fog)
    # Sương mù dày làm tăng độ sáng trung bình (Koschmieder airlight)
    assert img_fog.mean() > img_clear.mean()


def test_vietnamese_specific_primitives():
    """Kiểm tra dựng hình các loại vật thể đặc thù giao thông Việt Nam."""
    size_moto = [0.8, 1.9, 1.2]
    prims_cargo = _get_object_primitives("moto_cargo", size_moto)
    assert any(p[3] == "cargo_cloth" for p in prims_cargo), "moto_cargo phải có thùng hàng cargo_cloth"
    assert any(p[3] == "rider_clothes" for p in prims_cargo), "moto_cargo phải có người lái"

    size_bagac = [1.4, 3.2, 1.6]
    prims_bagac = _get_object_primitives("ba_gac", size_bagac)
    assert any(p[3] == "ba_gac_bed" for p in prims_bagac), "ba_gac phải có thùng xe ba_gac_bed"

    size_cart = [1.2, 1.6, 1.3]
    prims_cart = _get_object_primitives("vendor_cart", size_cart)
    assert any(p[3] == "wood" for p in prims_cart), "vendor_cart phải có khung gỗ"


def test_render_camera_nuscenes_integration(nusc):
    """Kiểm tra tích hợp render_camera trên keyframe thật của nuScenes."""
    sample = nusc.sample[0]
    tok = sample["token"]
    box = {
        "translation": [0.0, 15.0, -0.5],
        "size": [0.8, 1.9, 1.2],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "category": "vehicle.motorcycle",
        "inserted": True,
        "status": "inserted"
    }

    # Test render_camera với mode photorealistic_box
    img_box, n_seen = render_camera(nusc, tok, "CAM_FRONT", [box], mode="photorealistic_box")
    assert img_box.size == (800, 450)
    assert n_seen >= 1, "Vật thể chèn ở cự ly 15m phía trước phải nhìn thấy được trên CAM_FRONT"

    # Test render_camera với mode photorealistic
    img_photo, _ = render_camera(nusc, tok, "CAM_FRONT", [box], mode="photorealistic")
    assert img_photo.size == (800, 450)

    # Test render_camera với mode controlnet_depth
    img_depth, _ = render_camera(nusc, tok, "CAM_FRONT", [box], mode="controlnet_depth")
    assert img_depth.size == (800, 450)


def test_multiview_angle_selection():
    """Kiểm tra logic chọn view_name theo góc quan sát alpha và cos_alpha."""
    from utils.asset_manager import select_view_name, select_view_from_cos

    # 1. Kiểm tra logic cos_alpha theo chuẩn bước 2:
    # cos_alpha > 0.75: rear.png (nhìn thấy đuôi xe)
    assert select_view_from_cos(0.95) == "rear"
    assert select_view_from_cos(0.76) == "rear"

    # cos_alpha < -0.75: front.png (nhìn thấy đầu xe)
    assert select_view_from_cos(-0.95) == "front"
    assert select_view_from_cos(-0.80) == "front"

    # 0.25 < cos_alpha <= 0.75: oblique_rear.png
    assert select_view_from_cos(0.50) == "oblique_rear"
    assert select_view_from_cos(0.30) == "oblique_rear"

    # -0.75 <= cos_alpha < -0.25: oblique_front.png
    assert select_view_from_cos(-0.50) == "oblique_front"
    assert select_view_from_cos(-0.30) == "oblique_front"

    # |cos_alpha| <= 0.25: side.png
    assert select_view_from_cos(0.10) == "side"
    assert select_view_from_cos(-0.15) == "side"
    assert select_view_from_cos(0.0) == "side"

    # 2. Kiểm tra theo góc độ tương thích ngược:
    # Góc chính diện đầu xe (|alpha| <= 35)
    assert select_view_name(0.0) == "front"
    assert select_view_name(30.0) == "front"
    assert select_view_name(-34.5) == "front"

    # Góc chính diện đuôi xe (|alpha| >= 145)
    assert select_view_name(180.0) == "rear"
    assert select_view_name(150.0) == "rear"
    assert select_view_name(-160.0) == "rear"

    # Góc chéo trước (35 < |alpha| < 90)
    assert select_view_name(45.0) == "oblique_front"
    assert select_view_name(-75.0) == "oblique_front"

    # Góc chéo sau (90 <= |alpha| < 145)
    assert select_view_name(90.0) == "oblique_rear"
    assert select_view_name(135.0) == "oblique_rear"
    assert select_view_name(-110.0) == "oblique_rear"


def test_cull_near_and_huge_objects(mock_cam_setup):
    """Kiểm tra loại bỏ không vẽ các vật thể có z_cam < 4.0m hoặc tràn > 65% khung hình."""
    image, _, T, K = mock_cam_setup
    from task_a.camera_synthesis import composite_multiview_sprites

    # Xe ở quá gần (z_cam ~ 2.0m < 4.0m)
    near_box = {
        "translation": [2.8, 0.0, -0.5],
        "size": [1.9, 4.5, 1.6],
        "rotation": [1.0, 0.0, 0.0, 0.0],
        "category": "vehicle.car",
        "inserted": True
    }
    out_near, z_buf_near, mask_near = composite_multiview_sprites(image, [near_box], T, K)
    # Xe quá gần phải bị loại bỏ (không vẽ đè làm che khuất toàn bộ camera)
    assert np.all(out_near == image), "Vật thể có z_cam < 4.0m phải bị loại bỏ để tránh che kín màn hình"
    assert np.sum(mask_near) == 0


def test_sprite_bottom_alignment(mock_cam_setup):
    """Kiểm tra căn chỉnh đáy ảnh PNG chạm đúng đáy khung bao 2D (v_max)."""
    image, boxes, T, K = mock_cam_setup
    from task_a.camera_synthesis import composite_multiview_sprites
    from utils.visualization import to_box

    out_img, z_buf, mask = composite_multiview_sprites(image, boxes, T, K)

    # Tính v_max thực tế của 2D box
    corners_lid = to_box(boxes[0]).corners().T
    corners_cam = corners_lid @ T[:3, :3].T + T[:3, 3]
    uv = (corners_cam @ K.T)[:, :2] / corners_cam[:, 2:3]
    v_max = int(round(np.max(uv[:, 1])))

    # Hàng pixel có mask > 0 ở vị trí thấp nhất (y lớn nhất) phải chạm sát đáy (có clip theo biên ảnh)
    mask_y_indices = np.where(mask > 0)[0]
    assert len(mask_y_indices) > 0
    actual_bottom_y = int(np.max(mask_y_indices))
    expected_bottom = min(v_max, image.shape[0] - 1)
    assert abs(actual_bottom_y - expected_bottom) <= 2, f"Đáy ảnh {actual_bottom_y} phải chạm sát đáy 2D Box {expected_bottom}"

