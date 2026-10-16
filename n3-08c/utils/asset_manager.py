"""Asset Manager cho 2D Multi-view Sprite Compositing.

Quản lý thư mục tài nguyên assets/vn_objects/<category>/, chuẩn hóa tên lớp,
tự động nạp/cache ảnh PNG RGBA tách nền và chọn góc nhìn phù hợp.
"""
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
import cv2
import numpy as np
from PIL import Image

_SPRITE_CACHE: Dict[Tuple[str, str], np.ndarray] = {}


def get_default_assets_dir() -> Path:
    """Đường dẫn thư mục assets/vn_objects."""
    # Tìm theo PROJECT_ROOT
    base = Path(__file__).resolve().parents[2] / "assets" / "vn_objects"
    if not base.exists():
        # Fallback thư mục lân cận
        base_alt = Path(__file__).resolve().parents[1] / "assets" / "vn_objects"
        if base_alt.exists():
            return base_alt
    return base


def normalize_category(category: str) -> str:
    """Chuẩn hóa mọi tên lớp từ nuScenes hoặc kịch bản VN về 16 danh mục chuẩn."""
    cat = category.lower().strip()
    if "ba_gac" in cat or "bagac" in cat:
        return "ba_gac"
    if "cargo" in cat:
        return "moto_cargo"
    if "multi" in cat:
        return "moto_multi"
    if "tow" in cat:
        return "moto_tow"
    if "vendor" in cat or ("cart" in cat and "ox" not in cat):
        return "vendor_cart"
    if "ox" in cat:
        return "ox_cart"
    if "cow" in cat:
        return "cow"
    if "dog" in cat:
        return "dog"
    if "cone" in cat:
        return "traffic_cone"
    if "barrier" in cat:
        return "barrier"
    if "pedestrian" in cat or "human" in cat:
        return "pedestrian"
    if "bicycle" in cat or "bike" in cat:
        return "bicycle"
    if "motorcycle" in cat or "moto" in cat:
        return "motorcycle"
    if "bus" in cat:
        return "bus"
    if "truck" in cat:
        return "truck"
    if "car" in cat or "vehicle" in cat:
        return "car"
    return "car"


def select_view_from_cos(cos_alpha: float) -> str:
    """
    Chọn file ảnh theo cos(alpha) giữa vector hướng xe và vector tia nhìn camera:
    - cos_alpha > 0.75: xe chạy cùng chiều ra xa camera -> nhìn thấy đuôi (rear.png)
    - cos_alpha < -0.75: xe chạy ngược chiều hướng về camera -> nhìn thấy đầu (front.png)
    - 0.25 < cos_alpha <= 0.75: góc chéo sau (oblique_rear.png)
    - -0.75 <= cos_alpha < -0.25: góc chéo trước (oblique_front.png)
    - |cos_alpha| <= 0.25: xe chạy vuông góc cắt ngang -> nhìn ngang hông (side.png)
    """
    if cos_alpha > 0.75:
        return "rear"
    if cos_alpha < -0.75:
        return "front"
    if cos_alpha > 0.25:
        return "oblique_rear"
    if cos_alpha < -0.25:
        return "oblique_front"
    return "side"




def select_view_name(alpha_deg: float) -> str:
    """
    Chọn file ảnh theo góc quan sát alpha (độ) giữa tia nhìn camera và trục dọc vật thể:
    - Nhìn từ phía đầu xe (|alpha| <= 35 độ): front
    - Nhìn từ phía đuôi xe (|alpha| >= 145 độ): rear
    - Nhìn từ góc chéo hoặc ngang hông (35 < |alpha| < 145 độ):
        + Nếu |alpha| < 90 độ: oblique_front (hoặc fallback side)
        + Nếu |alpha| >= 90 độ: oblique_rear (hoặc fallback side)
    """
    abs_a = abs(alpha_deg)
    if abs_a <= 35.0:
        return "front"
    if abs_a >= 145.0:
        return "rear"
    if abs_a < 90.0:
        return "oblique_front"
    return "oblique_rear"


def load_sprite_rgba(category: str, view_name: str, assets_dir: Optional[Path] = None,
                     variant_idx: Optional[int] = None) -> np.ndarray:
    """Nạp ảnh PNG RGBA tách nền từ disk (có cache bộ nhớ). Hỗ trợ Multi-variant (5 biến thể/view)."""
    cat = normalize_category(category)
    cache_key = (cat, view_name, variant_idx)
    if cache_key in _SPRITE_CACHE:
        return _SPRITE_CACHE[cache_key].copy()

    if assets_dir is None:
        assets_dir = get_default_assets_dir()

    cat_dir = assets_dir / cat
    target_fp = None

    # Nếu có variant_idx: ưu tiên nạp biến thể <view_name>_{idx % 5}.png
    if variant_idx is not None and not view_name.endswith(".png"):
        v_cand = cat_dir / f"{view_name}_{variant_idx % 5}.png"
        if v_cand.is_file():
            target_fp = v_cand

    if target_fp is None or not target_fp.is_file():
        target_fp = cat_dir / f"{view_name}.png"

    # Thử fallback hợp lý nếu oblique_front / oblique_rear chưa có
    if not target_fp.is_file():
        if view_name == "oblique_rear":
            target_fp = cat_dir / "rear.png"
        elif view_name == "oblique_front":
            target_fp = cat_dir / "front.png"
        elif "oblique" in view_name:
            target_fp = cat_dir / "side.png"

    # Nếu vẫn chưa có, gọi bộ bootstrap tự tạo tức thì
    if not target_fp.is_file():
        from tools.bootstrap_sprites import generate_sprite
        cat_dir.mkdir(parents=True, exist_ok=True)
        sprite_pil = generate_sprite(cat, view_name if not view_name.endswith(".png") else "front")
        sprite_pil.save(cat_dir / f"{view_name}.png", format="PNG")
        arr = np.asarray(sprite_pil)
    else:
        # Đọc trực tiếp file PNG (hỗ trợ file do người dùng chép đè từ mạng)
        pil_img = Image.open(target_fp).convert("RGBA")
        arr = np.asarray(pil_img).copy()

    _SPRITE_CACHE[cache_key] = arr
    return arr.copy()
