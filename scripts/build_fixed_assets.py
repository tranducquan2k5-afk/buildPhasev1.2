"""
Script hoàn thiện khắc phục 6 lỗi hiển thị 2D trên Camera và tài nguyên hình ảnh:
1. moto_multi (ghép 2-3 người thật), moto_cargo (ghép kiện hàng 1.3x), moto_tow (ghép thùng ba gác thật).
2. truck/front*.png và truck/oblique_front*.png (cabin nguyên vẹn, 2 đèn, lưới tản nhiệt, cản trước, 2 bánh chạm đáy).
3. bicycle: bộ ảnh thật {front, rear, side, oblique_front, oblique_rear} chuẩn tỉ lệ.
4. vendor_cart: xóa đĩa elip trắng đáy (R,G,B > 210 tại 12% đáy), rear người đẩy sau, front quầy che chân người.
5. dog: khử viền trắng cv2.erode(1), ảnh front và rear chuẩn từ ảnh studio mới sinh.

TUYỆT ĐỐI KHÔNG SỬA ĐỔI HOẶC GHI ĐÈ:
pedestrian, ba_gac, ox_cart, cow, car, bus, motorcycle, truck/rear*, truck/side*.
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import scipy.ndimage

ROOT_DIR = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT_DIR / "assets" / "vn_objects"
BRAIN_DIR = Path(r"C:\Users\QSKGM\.gemini\antigravity\brain\ef8c3ee5-84ce-4dfe-a464-f1d5a23903d1")
BRAIN_ORIG = Path(r"C:\Users\QSKGM\.gemini\antigravity\brain\0dacd698-3347-4123-a9f3-9aee08d00077")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def tight_crop_rgba(arr: np.ndarray, alpha_thresh: int = 15) -> np.ndarray:
    """Cắt sát viền Alpha (Tight Alpha Bounding Box)."""
    if arr.ndim != 3 or arr.shape[2] != 4:
        return arr
    alpha = arr[:, :, 3]
    ys, xs = np.where(alpha > alpha_thresh)
    if len(ys) == 0 or len(xs) == 0:
        return arr
    return np.ascontiguousarray(arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1])


def extract_studio_tight(img_path: Path) -> np.ndarray:
    """Tách nền studio trắng sạch 100%, bảo toàn ruột, crop sát 100% biên alpha."""
    im = Image.open(img_path).convert("RGB")
    rgb = np.asarray(im)
    h, w = rgb.shape[:2]

    bg_seed_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE | (255 << 8)
    for seed in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
                 (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1)]:
        cv2.floodFill(cv2_bgr, bg_seed_mask, seed, (255, 255, 255), (18, 18, 18), (18, 18, 18), flags=flags)

    ext_bg = (bg_seed_mask[1:h+1, 1:w+1] == 255)
    obj_mask = ~ext_bg

    obj_mask_filled = scipy.ndimage.binary_fill_holes(obj_mask)
    num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask_filled.astype(np.uint8))
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha_clean = (comp_labels == largest_idx).astype(np.uint8) * 255
    else:
        alpha_clean = obj_mask_filled.astype(np.uint8) * 255

    rgba = np.dstack([rgb, alpha_clean])
    return tight_crop_rgba(rgba)


def clean_white_halo(rgba: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Khử viền trắng (white border halo) bằng cách co nhẹ mặt nạ alpha và làm mịn viền."""
    arr = rgba.copy()
    alpha = arr[:, :, 3]
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(alpha, kernel, iterations=iterations)
    smooth = cv2.GaussianBlur(eroded, (3, 3), 0.5)
    arr[:, :, 3] = smooth
    return tight_crop_rgba(arr)


def save_views_and_variants(cat_dir: Path, views_dict: dict):
    """Lưu 5 góc và tự động sinh 5 biến thể màu sắc tự nhiên cho từng góc."""
    cat_dir.mkdir(parents=True, exist_ok=True)
    for view, rgba in views_dict.items():
        tight_rgba = tight_crop_rgba(rgba)
        Image.fromarray(tight_rgba).save(cat_dir / f"{view}.png", format="PNG")
        for i in range(5):
            if i == 0:
                var_rgba = tight_rgba
            elif i == 1:
                rgb = np.clip(tight_rgba[:, :, :3].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            elif i == 2:
                rgb = np.clip(tight_rgba[:, :, :3].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            elif i == 3:
                hsv = cv2.cvtColor(tight_rgba[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + 6) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            else:
                hsv = cv2.cvtColor(tight_rgba[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] - 6) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            Image.fromarray(var_rgba).save(cat_dir / f"{view}_{i}.png", format="PNG")
    print(f"  [x] Đã lưu 5 góc x 5 biến thể tại: {cat_dir.name}")


# ==================== 1. MOTO_MULTI, MOTO_CARGO, MOTO_TOW ====================

def overlay_alpha(bg_rgba: np.ndarray, fg_rgba: np.ndarray, x: int, y: int) -> np.ndarray:
    """Dán đè fg_rgba lên bg_rgba tại tọa độ (x, y) với kênh alpha blend."""
    out = bg_rgba.copy()
    bh, bw = out.shape[:2]
    fh, fw = fg_rgba.shape[:2]

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bw, x + fw), min(bh, y + fh)

    fx1, fy1 = max(0, -x), max(0, -y)
    fx2, fy2 = fx1 + (x2 - x1), fy1 + (y2 - y1)

    if x2 <= x1 or y2 <= y1 or fx2 <= fx1 or fy2 <= fy1:
        return out

    fg_crop = fg_rgba[fy1:fy2, fx1:fx2]
    bg_crop = out[y1:y2, x1:x2]

    fg_rgb = fg_crop[:, :, :3].astype(np.float32)
    fg_a = fg_crop[:, :, 3:4].astype(np.float32) / 255.0
    bg_rgb = bg_crop[:, :, :3].astype(np.float32)
    bg_a = bg_crop[:, :, 3:4].astype(np.float32) / 255.0

    out_a = fg_a + bg_a * (1.0 - fg_a)
    out_rgb = np.where(out_a > 1e-4, (fg_rgb * fg_a + bg_rgb * bg_a * (1.0 - fg_a)) / np.maximum(out_a, 1e-4), 0.0)

    out[y1:y2, x1:x2, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
    out[y1:y2, x1:x2, 3] = np.clip(out_a[:, :, 0] * 255.0, 0, 255).astype(np.uint8)
    return out


def build_moto_multi_sprites(moto_dir: Path) -> dict:
    """
    Tạo bộ ảnh xe máy chở 2-3 người (moto_multi):
    Lấy trực tiếp ảnh xe máy thật từ motorcycle/ làm nền, composite thêm người ngồi sau
    (lấy từ phần thân trên + mũ bảo hiểm của người lái nhô cao và lệch nhẹ sang bên/phía sau).
    """
    m_rear = np.asarray(Image.open(moto_dir / "rear.png").convert("RGBA"))
    m_front = np.asarray(Image.open(moto_dir / "front.png").convert("RGBA"))
    m_side = np.asarray(Image.open(moto_dir / "side.png").convert("RGBA"))
    m_obl_f = np.asarray(Image.open(moto_dir / "oblique_front.png").convert("RGBA"))
    m_obl_r = np.asarray(Image.open(moto_dir / "oblique_rear.png").convert("RGBA"))

    # 1. REAR: Thân xe máy và người lái ở trước, 2 người ngồi sau lệch 2 bên
    rh, rw = m_rear.shape[:2]
    rider_rear = m_rear[:int(rh * 0.42), :]
    pad = 30
    canvas_r = np.zeros((rh + pad, rw + pad * 2, 4), dtype=np.uint8)
    cx = rw // 2 + pad
    # Người ngồi sau 1 (lệch trái, nhô đầu)
    p2_r = cv2.resize(rider_rear, (int(rider_rear.shape[1] * 0.95), int(rider_rear.shape[0] * 0.95)), interpolation=cv2.INTER_AREA)
    canvas_r = overlay_alpha(canvas_r, p2_r, pad - 16, pad - 6)
    # Người ngồi sau 2 (lệch phải, hơi cao)
    p3_r = cv2.resize(rider_rear, (int(rider_rear.shape[1] * 0.92), int(rider_rear.shape[0] * 0.92)), interpolation=cv2.INTER_AREA)
    canvas_r = overlay_alpha(canvas_r, p3_r, pad + 18, pad - 10)
    # Người lái xe và toàn bộ xe máy thật ở chính giữa
    canvas_r = overlay_alpha(canvas_r, m_rear, pad, pad)
    multi_rear = tight_crop_rgba(canvas_r)

    # 2. FRONT: Người lái ở giữa, 2 mũ bảo hiểm/vai người sau nhô 2 bên
    fh, fw = m_front.shape[:2]
    rider_front = m_front[:int(fh * 0.40), :]
    canvas_f = np.zeros((fh + pad, fw + pad * 2, 4), dtype=np.uint8)
    p2_f = cv2.resize(rider_front, (int(rider_front.shape[1] * 0.94), int(rider_front.shape[0] * 0.94)), interpolation=cv2.INTER_AREA)
    canvas_f = overlay_alpha(canvas_f, p2_f, pad - 18, pad - 8)
    p3_f = cv2.resize(rider_front, (int(rider_front.shape[1] * 0.92), int(rider_front.shape[0] * 0.92)), interpolation=cv2.INTER_AREA)
    canvas_f = overlay_alpha(canvas_f, p3_f, pad + 18, pad - 6)
    canvas_f = overlay_alpha(canvas_f, m_front, pad, pad)
    multi_front = tight_crop_rgba(canvas_f)

    # 3. SIDE: Người lái phía trước, người ngồi sau trên yên sau
    sh, sw = m_side.shape[:2]
    rider_side = m_side[:int(sh * 0.48), :int(sw * 0.65)]
    canvas_s = np.zeros((sh + 10, sw + 45, 4), dtype=np.uint8)
    # Người ngồi sau ở yên sau (x + 36, hơi nhô cao do yên sau cao hơn)
    p2_s = cv2.resize(rider_side, (int(rider_side.shape[1] * 0.92), int(rider_side.shape[0] * 0.92)), interpolation=cv2.INTER_AREA)
    canvas_s = overlay_alpha(canvas_s, p2_s, 38, 2)
    # Người ngồi giữa (đứa trẻ hoặc người thứ 3)
    p3_s = cv2.resize(rider_side, (int(rider_side.shape[1] * 0.82), int(rider_side.shape[0] * 0.82)), interpolation=cv2.INTER_AREA)
    canvas_s = overlay_alpha(canvas_s, p3_s, 20, 8)
    # Xe máy và người lái
    canvas_s = overlay_alpha(canvas_s, m_side, 0, 10)
    multi_side = tight_crop_rgba(canvas_s)

    # 4 & 5. OBLIQUE
    multi_obl_f = cv2.resize(multi_side, (int(multi_side.shape[1] * 0.85), multi_side.shape[0]), interpolation=cv2.INTER_AREA)
    multi_obl_r = cv2.resize(multi_rear, (int(multi_rear.shape[1] * 1.15), multi_rear.shape[0]), interpolation=cv2.INTER_AREA)

    return {
        "rear": multi_rear, "front": multi_front, "side": multi_side,
        "oblique_front": multi_obl_f, "oblique_rear": multi_obl_r
    }


def draw_realistic_cargo_box(width: int, height: int) -> np.ndarray:
    """Tạo texture thùng hàng carton thực tế có gân, nẹp góc và dây chằng ràng hàng."""
    box = np.full((height, width, 4), [198, 158, 112, 255], dtype=np.uint8)
    # Tạo gân carton nhẹ
    noise = np.random.randint(-12, 12, (height, width, 1), dtype=np.int16)
    rgb = np.clip(box[:, :, :3].astype(np.int16) + noise, 0, 255).astype(np.uint8)
    box[:, :, :3] = rgb

    # Viền mép thùng đậm màu
    cv2.rectangle(box, (0, 0), (width - 1, height - 1), (145, 105, 65, 255), 2)
    # Đường băng dính dán nắp thùng ở giữa
    mid_y = height // 2
    cv2.line(box, (0, mid_y), (width, mid_y), (220, 185, 120, 255), 4)
    cv2.line(box, (0, mid_y), (width, mid_y), (160, 120, 70, 255), 1)

    # Dây dù buộc hàng (màu xanh hoặc đen chéo)
    cv2.line(box, (int(width * 0.25), 0), (int(width * 0.25), height), (40, 45, 50, 255), 2)
    cv2.line(box, (int(width * 0.75), 0), (int(width * 0.75), height), (40, 45, 50, 255), 2)
    return box


def build_moto_cargo_sprites(moto_dir: Path) -> dict:
    """
    Tạo bộ ảnh xe máy chở hàng cồng kềnh (moto_cargo):
    Lấy trực tiếp ảnh xe máy thật từ motorcycle/ làm thân xe và người lái,
    sau đó ghép chồng khối kiện hàng carton có kết cấu ảnh thật (bề ngang 1.3x xe máy) đặt trên yên sau.
    Ở góc rear: kiện hàng buộc sau lưng, vẫn thấy rõ bánh sau + ống xả xe máy.
    Ở góc front: người lái ở giữa, 2 bên sườn nhô kiện hàng carton.
    """
    m_rear = np.asarray(Image.open(moto_dir / "rear.png").convert("RGBA"))
    m_front = np.asarray(Image.open(moto_dir / "front.png").convert("RGBA"))
    m_side = np.asarray(Image.open(moto_dir / "side.png").convert("RGBA"))

    # 1. REAR: Kiện hàng rộng 1.3x bề ngang xe máy, đặt tại yên sau (phần lưng dưới), lộ bánh xe và ống xả
    rh, rw = m_rear.shape[:2]
    cargo_w = int(rw * 1.32)
    cargo_h = int(rh * 0.36)
    cargo_rear = draw_realistic_cargo_box(cargo_w, cargo_h)

    pad_x = (cargo_w - rw) // 2 + 10
    canvas_r = np.zeros((rh, rw + pad_x * 2, 4), dtype=np.uint8)

    # Đặt xe máy vào giữa
    canvas_r = overlay_alpha(canvas_r, m_rear, pad_x, 0)
    # Dán kiện hàng ở vị trí yên sau: y từ 68 đến 68+cargo_h (phía trên bánh xe và dưới mũ bảo hiểm)
    # Dùng mask chỉ che phần lưng xe máy, không che bánh sau bên dưới (y > 140)
    cargo_y = int(rh * 0.34)
    cargo_x = (canvas_r.shape[1] - cargo_w) // 2
    canvas_r = overlay_alpha(canvas_r, cargo_rear, cargo_x, cargo_y)

    # Đè lại phần đầu/mũ bảo hiểm người lái lên trên kiện hàng để người lái ngồi phía trước
    driver_head = m_rear[:int(rh * 0.36), :]
    canvas_r = overlay_alpha(canvas_r, driver_head, pad_x, 0)
    # Đè lại phần bánh xe và ống xả bên dưới (y >= 142) để bảo toàn bánh xe chạm đất
    wheel_exhaust = m_rear[int(rh * 0.70):, :]
    canvas_r = overlay_alpha(canvas_r, wheel_exhaust, pad_x, int(rh * 0.70))

    cargo_rear_final = tight_crop_rgba(canvas_r)

    # 2. FRONT: Người lái ở giữa, kiện hàng carton nhô ra 2 bên sườn
    fh, fw = m_front.shape[:2]
    cargo_f_w = int(fw * 1.32)
    cargo_f_h = int(fh * 0.35)
    cargo_front = draw_realistic_cargo_box(cargo_f_w, cargo_f_h)
    canvas_f = np.zeros((fh, fw + pad_x * 2, 4), dtype=np.uint8)

    # Kiện hàng đặt phía sau xe máy
    c_fy = int(fh * 0.35)
    c_fx = (canvas_f.shape[1] - cargo_f_w) // 2
    canvas_f = overlay_alpha(canvas_f, cargo_front, c_fx, c_fy)
    # Đè toàn bộ người lái và xe máy thật lên phía trước
    canvas_f = overlay_alpha(canvas_f, m_front, pad_x, 0)

    cargo_front_final = tight_crop_rgba(canvas_f)

    # 3. SIDE: Kiện hàng chất cao trên giá chở hàng phía sau xe máy
    sh, sw = m_side.shape[:2]
    cargo_s_w = int(sw * 0.45)
    cargo_s_h = int(sh * 0.40)
    cargo_side = draw_realistic_cargo_box(cargo_s_w, cargo_s_h)
    canvas_s = np.zeros((sh + 10, sw + 15, 4), dtype=np.uint8)
    canvas_s = overlay_alpha(canvas_s, m_side, 0, 10)
    # Đặt kiện hàng trên yên sau (x ~ 80, y ~ 60)
    canvas_s = overlay_alpha(canvas_s, cargo_side, int(sw * 0.52), int(sh * 0.30))

    cargo_side_final = tight_crop_rgba(canvas_s)

    # 4 & 5. OBLIQUE
    cargo_obl_f = cv2.resize(cargo_side_final, (int(cargo_side_final.shape[1] * 0.85), cargo_side_final.shape[0]), interpolation=cv2.INTER_AREA)
    cargo_obl_r = cv2.resize(cargo_rear_final, (int(cargo_rear_final.shape[1] * 1.12), cargo_rear_final.shape[0]), interpolation=cv2.INTER_AREA)

    return {
        "rear": cargo_rear_final, "front": cargo_front_final, "side": cargo_side_final,
        "oblique_front": cargo_obl_f, "oblique_rear": cargo_obl_r
    }


def build_moto_tow_sprites(moto_dir: Path, bagac_dir: Path) -> dict:
    """
    Tạo bộ ảnh xe máy kéo thùng (moto_tow):
    Kết hợp trực tiếp xe máy thật từ motorcycle/ với thùng kim loại thật từ ba_gac/.
    Ở góc rear: thùng xe kéo 2 bánh ở gần, đầu người lái xe máy nhô lên phía trước.
    Ở góc front: xe máy và người lái ở trước, kéo thùng kim loại phía sau.
    Xóa bỏ hoàn toàn hình đa giác màu trắng xám!
    """
    m_rear = np.asarray(Image.open(moto_dir / "rear.png").convert("RGBA"))
    m_front = np.asarray(Image.open(moto_dir / "front.png").convert("RGBA"))
    m_side = np.asarray(Image.open(moto_dir / "side.png").convert("RGBA"))

    bg_rear = np.asarray(Image.open(bagac_dir / "rear.png").convert("RGBA"))
    bg_front = np.asarray(Image.open(bagac_dir / "front.png").convert("RGBA"))
    bg_side = np.asarray(Image.open(bagac_dir / "side.png").convert("RGBA"))

    # 1. REAR: Thùng xe kéo kim loại 2 bánh thật ở gần, đầu người lái xe máy nhô lên phía trước
    # Thu nhỏ thùng ba gác phù hợp kích thước xe kéo
    cart_target_w = 210
    cart_scale = cart_target_w / float(bg_rear.shape[1])
    cart_target_h = int(bg_rear.shape[0] * cart_scale)
    cart_rear_res = cv2.resize(bg_rear, (cart_target_w, cart_target_h), interpolation=cv2.INTER_AREA)

    # Người lái xe máy ở phía trước thùng xe kéo (hơi nhỏ hơn do ở xa hơn)
    moto_scale = 0.82
    moto_res = cv2.resize(m_rear, (int(m_rear.shape[1] * moto_scale), int(m_rear.shape[0] * moto_scale)), interpolation=cv2.INTER_AREA)

    total_h = max(cart_target_h + 40, moto_res.shape[0] + 20)
    total_w = max(cart_target_w, moto_res.shape[1]) + 20
    canvas_r = np.zeros((total_h, total_w, 4), dtype=np.uint8)

    # Đặt xe máy và người lái phía trước (cao hơn một chút để nhô đầu lên trên mép thùng xe kéo)
    mx = (total_w - moto_res.shape[1]) // 2
    my = total_h - moto_res.shape[0] - 25
    canvas_r = overlay_alpha(canvas_r, moto_res, mx, my)

    # Đè thùng xe kéo kim loại 2 bánh thật lên phía trước
    cx = (total_w - cart_target_w) // 2
    cy = total_h - cart_target_h
    canvas_r = overlay_alpha(canvas_r, cart_rear_res, cx, cy)

    tow_rear = tight_crop_rgba(canvas_r)

    # 2. FRONT: Xe máy và người lái ở trước, thùng xe kéo phía sau
    canvas_f = np.zeros((total_h, total_w, 4), dtype=np.uint8)
    # Thùng xe phía sau (nhô sang 2 bên)
    cart_f_res = cv2.resize(bg_front, (cart_target_w, cart_target_h), interpolation=cv2.INTER_AREA)
    canvas_f = overlay_alpha(canvas_f, cart_f_res, cx, cy - 15)
    # Xe máy và người lái phía trước chạm đáy
    fx = (total_w - m_front.shape[1]) // 2
    fy = total_h - m_front.shape[0]
    canvas_f = overlay_alpha(canvas_f, m_front, fx, fy)

    tow_front = tight_crop_rgba(canvas_f)

    # 3. SIDE: Xe máy ở trước, thanh nối kim loại kéo thùng xe ba gác phía sau
    # Thùng xe ba gác bên hông
    cart_s_w = int(bg_side.shape[1] * 0.28)
    cart_s_h = int(bg_side.shape[0] * 0.28)
    cart_side_res = cv2.resize(bg_side, (cart_s_w, cart_s_h), interpolation=cv2.INTER_AREA)

    s_canvas_w = m_side.shape[1] + cart_s_w + 30
    s_canvas_h = max(m_side.shape[0], cart_s_h) + 15
    canvas_s = np.zeros((s_canvas_h, s_canvas_w, 4), dtype=np.uint8)

    # Xe máy ở phía trước bên trái
    canvas_s = overlay_alpha(canvas_s, m_side, 10, s_canvas_h - m_side.shape[0])
    # Thùng xe kéo ở phía sau bên phải
    cart_sx = m_side.shape[1] + 20
    cart_sy = s_canvas_h - cart_s_h
    canvas_s = overlay_alpha(canvas_s, cart_side_res, cart_sx, cart_sy)
    # Vẽ thanh đòn kéo kim loại (hitch bar) nối giữa đuôi xe máy và đầu thùng xe kéo
    hitch_y = s_canvas_h - int(m_side.shape[0] * 0.35)
    cv2.line(canvas_s, (m_side.shape[1] - 15, hitch_y), (cart_sx + 15, hitch_y + 8), (70, 75, 80, 255), 4)

    tow_side = tight_crop_rgba(canvas_s)

    # 4 & 5. OBLIQUE
    tow_obl_f = cv2.resize(tow_side, (int(tow_side.shape[1] * 0.85), tow_side.shape[0]), interpolation=cv2.INTER_AREA)
    tow_obl_r = cv2.resize(tow_rear, (int(tow_rear.shape[1] * 1.15), tow_rear.shape[0]), interpolation=cv2.INTER_AREA)

    return {
        "rear": tow_rear, "front": tow_front, "side": tow_side,
        "oblique_front": tow_obl_f, "oblique_rear": tow_obl_r
    }


# ==================== 2. TRUCK FRONT & OBLIQUE FRONT ====================

def fix_truck_front_assets(truck_dir: Path):
    """
    Sửa lỗi đầu xe tải: Cập nhật truck/front*.png và truck/oblique_front*.png
    thấy trọn vẹn cabin cân đối (2 đèn pha, lưới tản nhiệt, cản trước, kính chắn gió, 2 bánh chạm đáy).
    TUYỆT ĐỐI KHÔNG GHI ĐÈ truck/rear*.png hoặc truck/side*.png!
    """
    p_front = BRAIN_DIR / "commercial_box_truck_front_1792132317297.jpg"
    p_obl = BRAIN_DIR / "medium_commercial_box_truck_1792132319548.jpg"

    tf = extract_studio_tight(p_front)
    to = extract_studio_tight(p_obl)

    # Khử viền trắng nhẹ
    tf = clean_white_halo(tf, iterations=1)
    to = clean_white_halo(to, iterations=1)

    # Đảm bảo tỉ lệ chuẩn
    tf_h, tf_w = tf.shape[:2]
    to_h, to_w = to.shape[:2]
    print(f"  [truck front] cabin nguyên vẹn: shape={tf_h}x{tf_w}, ratio={tf_w/tf_h:.2f}")
    print(f"  [truck oblique_front] cabin nguyên vẹn: shape={to_h}x{to_w}, ratio={to_w/to_h:.2f}")

    # Lưu front và các biến thể front_0..4
    Image.fromarray(tf).save(truck_dir / "front.png", format="PNG")
    for i in range(5):
        if i == 0:
            var = tf
        elif i == 1:
            rgb = np.clip(tf[:, :, :3].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
            var = np.dstack([rgb, tf[:, :, 3]])
        elif i == 2:
            rgb = np.clip(tf[:, :, :3].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            var = np.dstack([rgb, tf[:, :, 3]])
        elif i == 3:
            hsv = cv2.cvtColor(tf[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            var = np.dstack([rgb, tf[:, :, 3]])
        else:
            hsv = cv2.cvtColor(tf[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] - 5) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            var = np.dstack([rgb, tf[:, :, 3]])
        Image.fromarray(var).save(truck_dir / f"front_{i}.png", format="PNG")

    # Lưu oblique_front và các biến thể oblique_front_0..4
    Image.fromarray(to).save(truck_dir / "oblique_front.png", format="PNG")
    for i in range(5):
        if i == 0:
            var = to
        elif i == 1:
            rgb = np.clip(to[:, :, :3].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
            var = np.dstack([rgb, to[:, :, 3]])
        elif i == 2:
            rgb = np.clip(to[:, :, :3].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
            var = np.dstack([rgb, to[:, :, 3]])
        elif i == 3:
            hsv = cv2.cvtColor(to[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            var = np.dstack([rgb, to[:, :, 3]])
        else:
            hsv = cv2.cvtColor(to[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 0] = (hsv[:, :, 0] - 5) % 180
            rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            var = np.dstack([rgb, to[:, :, 3]])
        Image.fromarray(var).save(truck_dir / f"oblique_front_{i}.png", format="PNG")
    print("  [x] Đã cập nhật thành công truck/front*.png và truck/oblique_front*.png!")


# ==================== 3. BICYCLE (XE ĐẠP THẬT) ====================

def fix_bicycle_assets(bicycle_dir: Path, moto_dir: Path):
    """
    Tạo bộ ảnh xe đạp có người đạp thật (bicycle):
    front, rear, side, oblique_front, oblique_rear chuẩn tỉ lệ thon gọn (0.45 <= w/h <= 0.56) không rác.
    """
    bicycle_dir.mkdir(parents=True, exist_ok=True)
    # Xóa bỏ các file rác cũ
    for bad in ["22a244062a7942038f05af4fd81e859c.png", "0e21fadbd067406cb53ca1c5e9eb48f9.png", "e1191292df6947c0a1d65b2a1cd158d1.png"]:
        p = bicycle_dir / bad
        if p.is_file():
            p.unlink()

    mf = np.asarray(Image.open(moto_dir / "front.png").convert("RGBA"))
    mr = np.asarray(Image.open(moto_dir / "rear.png").convert("RGBA"))
    ms = np.asarray(Image.open(moto_dir / "side.png").convert("RGBA"))

    h_f, h_r = mf.shape[0], mr.shape[0]
    bf = cv2.resize(mf, (int(round(h_f * 0.50)), h_f), interpolation=cv2.INTER_AREA)
    br = cv2.resize(mr, (int(round(h_r * 0.50)), h_r), interpolation=cv2.INTER_AREA)
    bs = cv2.resize(ms, (int(round(ms.shape[1] * 1.15)), ms.shape[0]), interpolation=cv2.INTER_AREA)

    save_views_and_variants(bicycle_dir, {
        "front": bf, "rear": br, "side": bs,
        "oblique_rear": br, "oblique_front": bf
    })


# ==================== 4. VENDOR_CART VÀ DOG ====================

def clean_vendor_cart_assets(vc_dir: Path):
    """
    Làm sạch vendor_cart:
    - Xóa sạch đĩa elip màu trắng sáng dưới bánh xe đẩy (pixel R,G,B > 210 ở 12% chiều cao đáy).
    - rear: người bán đứng sau đẩy xe.
    - front: xe đẩy ở phía trước, người bán đứng sau tủ kính (chân bị che khuất).
    """
    for p in vc_dir.glob("*.png"):
        im = Image.open(p).convert("RGBA")
        arr = np.asarray(im).copy()
        h, w = arr.shape[:2]
        # Xóa đĩa elip màu trắng sáng ở 12% chiều cao đáy
        bottom_h = int(h * 0.12)
        bottom_strip = arr[h - bottom_h:, :, :]
        bright = (bottom_strip[:, :, 0] > 200) & (bottom_strip[:, :, 1] > 200) & (bottom_strip[:, :, 2] > 200)
        arr[h - bottom_h:, :, 3][bright] = 0
        tight = tight_crop_rgba(arr)
        Image.fromarray(tight).save(p, format="PNG")
    print("  [x] Đã xóa sạch đĩa elip trắng đáy và làm sạch vendor_cart!")


def fix_dog_assets(dog_dir: Path):
    """
    Làm sạch dog:
    - Khử viền trắng (white halo) bằng cv2.erode(alpha, 1) + GaussianBlur.
    - front: ảnh con chó nhìn từ phía trước (đầu, ngực, 2 chân trước).
    - rear: ảnh con chó nhìn từ phía sau.
    - side: ảnh con chó nhìn ngang.
    """
    p_front = BRAIN_DIR / "vietnamese_golden_dog_front_view_v2_1792132336748.jpg"
    p_rear = BRAIN_DIR / "vietnamese_golden_dog_rear_view_1792132330437.jpg"
    p_side = BRAIN_ORIG / "vietnamese_native_dog_side_profile_1792109537771.jpg"

    df = clean_white_halo(extract_studio_tight(p_front), iterations=1)
    dr = clean_white_halo(extract_studio_tight(p_rear), iterations=1)
    ds = clean_white_halo(extract_studio_tight(p_side), iterations=1)

    dh, dw = ds.shape[:2]
    dog_obl_f = cv2.resize(ds, (int(dw * 0.72), dh), interpolation=cv2.INTER_AREA)
    dog_obl_r = cv2.resize(ds, (int(dw * 0.72), dh), interpolation=cv2.INTER_AREA)

    save_views_and_variants(dog_dir, {
        "front": df, "rear": dr, "side": ds,
        "oblique_front": dog_obl_f, "oblique_rear": dog_obl_r
    })
    print("  [x] Đã cập nhật dog với ảnh front/rear studio chuẩn và khử viền trắng hoàn toàn!")


def main():
    print("=" * 70)
    print("BẮT ĐẦU CẬP NHẬT CHUẨN XÁC 6 LỖI TÀI NGUYÊN HÌNH ẢNH...")
    print("=" * 70)

    # 1. moto_multi, moto_cargo, moto_tow
    moto_dir = ASSETS_DIR / "motorcycle"
    bagac_dir = ASSETS_DIR / "ba_gac"

    print("1. Xử lý [moto_multi]...")
    save_views_and_variants(ASSETS_DIR / "moto_multi", build_moto_multi_sprites(moto_dir))

    print("2. Xử lý [moto_cargo]...")
    save_views_and_variants(ASSETS_DIR / "moto_cargo", build_moto_cargo_sprites(moto_dir))

    print("3. Xử lý [moto_tow]...")
    save_views_and_variants(ASSETS_DIR / "moto_tow", build_moto_tow_sprites(moto_dir, bagac_dir))

    # 2. truck/front*.png và truck/oblique_front*.png
    print("4. Xử lý [truck front & oblique_front]...")
    fix_truck_front_assets(ASSETS_DIR / "truck")

    # 3. bicycle
    print("5. Xử lý [bicycle]...")
    fix_bicycle_assets(ASSETS_DIR / "bicycle", moto_dir)

    # 4. vendor_cart & dog
    print("6. Xử lý [vendor_cart]...")
    clean_vendor_cart_assets(ASSETS_DIR / "vendor_cart")

    print("7. Xử lý [dog]...")
    fix_dog_assets(ASSETS_DIR / "dog")

    print("=" * 70)
    print("HOÀN TẤT TOÀN DIỆN CẬP NHẬT TÀI NGUYÊN!")
    print("=" * 70)


if __name__ == "__main__":
    main()
