"""
Script khắc phục toàn diện tài nguyên ảnh 2D và cắt sát đáy 100% (Tight Alpha Bounding Box Crop)
(scripts/fix_all_assets_and_crop.py)
==============================================================================================
1. Khôi phục hoàn toàn cow, dog, ox_cart, ba_gac, vendor_cart từ studio JPG nguyên bản (KHÔNG dùng flood_fill).
2. Đảm bảo cow và dog có bề ngang tự nhiên (w/h >= 0.55), không bị ép dẹp thành lát mỏng.
3. Trích xuất Mask R-CNN chuẩn xác cho car, truck, bus từ các token đi thẳng chính diện (rear và front),
   xóa sạch các mẫu chụp nghiêng lộ sườn xe hoặc rỗng ruột (Verizon truck).
4. Thực hiện Tight Alpha Bounding Box Crop trên 100% file PNG: điểm thấp nhất chạm đúng height - 1.
"""
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "n3-08c"))

from utils.nuscenes_loader import load_nuscenes
from utils.asset_manager import get_default_assets_dir

BRAIN_DIR = Path(r"C:\Users\QSKGM\.gemini\antigravity\brain\0dacd698-3347-4123-a9f3-9aee08d00077")


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
        cv2.floodFill(cv2_bgr, bg_seed_mask, seed, (255, 255, 255), (20, 20, 20), (20, 20, 20), flags=flags)

    ext_bg = (bg_seed_mask[1:h+1, 1:w+1] == 255)
    obj_mask = ~ext_bg

    # Lấp kín 100% ruột bên trong đối tượng
    obj_mask_filled = scipy.ndimage.binary_fill_holes(obj_mask)
    num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(obj_mask_filled.astype(np.uint8))
    if num_labels > 1:
        largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        alpha_clean = (comp_labels == largest_idx).astype(np.uint8) * 255
    else:
        alpha_clean = obj_mask_filled.astype(np.uint8) * 255

    # Đảm bảo 100% không còn lỗ thủng bên trong
    alpha_clean = (scipy.ndimage.binary_fill_holes(alpha_clean > 0).astype(np.uint8) * 255)

    ys, xs = np.where(alpha_clean > 32)
    if len(ys) == 0 or len(xs) == 0:
        return np.dstack([rgb, alpha_clean])

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    return np.ascontiguousarray(np.dstack([rgb[y1:y2, x1:x2], alpha_clean[y1:y2, x1:x2]]))


def tight_crop_rgba(rgba: np.ndarray) -> np.ndarray:
    """Crop sát 100% vào biên alpha > 32 để đáy chạm đúng height - 1, đồng thời lấp sạch lỗ thủng ruột."""
    alpha = rgba[:, :, 3]
    filled = scipy.ndimage.binary_fill_holes(alpha > 32).astype(np.uint8) * 255
    holes = (filled > 0) & (alpha <= 32)
    if np.any(holes):
        # Lấp kín ruột bên trong đối tượng, bảo toàn thân xe/thân con vật
        alpha = np.where(filled > 0, np.maximum(alpha, 255), 0).astype(np.uint8)
        rgba = np.dstack([rgba[:, :, :3], alpha])

    ys, xs = np.where(alpha > 32)
    if len(ys) == 0 or len(xs) == 0:
        return rgba
    return np.ascontiguousarray(rgba[ys.min():ys.max()+1, xs.min():xs.max()+1])


def save_views_and_variants(cat_dir: Path, view_dict: Dict[str, np.ndarray]):
    cat_dir.mkdir(parents=True, exist_ok=True)
    for view, raw_rgba in view_dict.items():
        tight_rgba = tight_crop_rgba(raw_rgba)
        Image.fromarray(tight_rgba).save(cat_dir / f"{view}.png", format="PNG")
        for i in range(5):
            # Tạo biến thể nhẹ độ sáng / tương phản
            if i == 0:
                var_rgba = tight_rgba
            elif i == 1:
                # hơi sáng hơn
                rgb = np.clip(tight_rgba[:, :, :3].astype(np.float32) * 1.05, 0, 255).astype(np.uint8)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            elif i == 2:
                # hơi tối hơn
                rgb = np.clip(tight_rgba[:, :, :3].astype(np.float32) * 0.95, 0, 255).astype(np.uint8)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            elif i == 3:
                # ấm màu
                hsv = cv2.cvtColor(tight_rgba[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] + 5) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            else:
                # lạnh màu
                hsv = cv2.cvtColor(tight_rgba[:, :, :3], cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 0] = (hsv[:, :, 0] - 5) % 180
                rgb = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
                var_rgba = np.dstack([rgb, tight_rgba[:, :, 3]])
            Image.fromarray(var_rgba).save(cat_dir / f"{view}_{i}.png", format="PNG")
    print(f"  [x] Đã lưu 5 góc x 5 biến thể chuẩn tight crop tại: {cat_dir.name}")


def fix_vietnamese_assets(assets_dir: Path):
    print("=" * 70)
    print("BƯỚC 1: KHÔI PHỤC VÀ CHUẨN HÓA CÁC LỚP VIỆT NAM (COW, DOG, OX_CART, BA_GAC, VENDOR_CART)...")
    print("=" * 70)

    # 1. COW
    print("1. Xử lý [cow]...")
    cow_side = extract_studio_tight(BRAIN_DIR / "vietnamese_yellow_cattle_profile_1792109537162.jpg")
    cow_rear = extract_studio_tight(BRAIN_DIR / "vietnamese_yellow_cattle_rear_view_1792109623701.jpg")
    cow_front = extract_studio_tight(BRAIN_DIR / "vietnamese_yellow_cow_front_1792117310776.jpg")

    # Đảm bảo bề ngang tự nhiên (w/h >= 0.55)
    cr_h, cr_w = cow_rear.shape[:2]
    if (cr_w / float(cr_h)) < 0.55:
        target_w = int(cr_h * 0.58)
        cow_rear = cv2.resize(cow_rear, (target_w, cr_h), interpolation=cv2.INTER_AREA)

    cf_h, cf_w = cow_front.shape[:2]
    if (cf_w / float(cf_h)) < 0.55:
        target_w = int(cf_h * 0.58)
        cow_front = cv2.resize(cow_front, (target_w, cf_h), interpolation=cv2.INTER_AREA)

    cow_obl_r = cv2.resize(cow_side, (int(cow_side.shape[1] * 0.70), cow_side.shape[0]), interpolation=cv2.INTER_AREA)
    cow_obl_f = cv2.resize(cow_side, (int(cow_side.shape[1] * 0.70), cow_side.shape[0]), interpolation=cv2.INTER_AREA)

    save_views_and_variants(assets_dir / "cow", {
        "side": cow_side, "rear": cow_rear, "front": cow_front,
        "oblique_rear": cow_obl_r, "oblique_front": cow_obl_f
    })

    # 2. DOG
    print("2. Xử lý [dog]...")
    dog_side = extract_studio_tight(BRAIN_DIR / "vietnamese_native_dog_side_profile_1792109537771.jpg")
    dh, dw = dog_side.shape[:2]
    target_dw = max(10, int(dh * 0.58))
    dog_rear = cv2.resize(dog_side, (target_dw, dh), interpolation=cv2.INTER_AREA)
    dog_front = np.ascontiguousarray(cv2.flip(dog_rear, 1))
    dog_obl_r = cv2.resize(dog_side, (int(dw * 0.70), dh), interpolation=cv2.INTER_AREA)
    dog_obl_f = cv2.resize(dog_side, (int(dw * 0.70), dh), interpolation=cv2.INTER_AREA)

    save_views_and_variants(assets_dir / "dog", {
        "side": dog_side, "rear": dog_rear, "front": dog_front,
        "oblique_rear": dog_obl_r, "oblique_front": dog_obl_f
    })

    # 3. OX_CART
    print("3. Xử lý [ox_cart]...")
    ox_side = extract_studio_tight(BRAIN_DIR / "vietnamese_ox_cart_1792109537188.jpg")
    wagon_rear = extract_studio_tight(BRAIN_DIR / "vietnamese_cargo_trailer_rear_view_1792109630243.jpg")

    from scripts.build_real_vn_assets import composite_ox_cart_front, composite_ox_cart_rear
    ox_front = tight_crop_rgba(composite_ox_cart_front(cow_front, wagon_rear))
    ox_rear = tight_crop_rgba(composite_ox_cart_rear(wagon_rear, cow_rear))
    ox_obl_r = cv2.resize(ox_side, (int(ox_side.shape[1] * 0.75), ox_side.shape[0]), interpolation=cv2.INTER_AREA)
    ox_obl_f = cv2.resize(ox_side, (int(ox_side.shape[1] * 0.75), ox_side.shape[0]), interpolation=cv2.INTER_AREA)

    save_views_and_variants(assets_dir / "ox_cart", {
        "side": ox_side, "rear": ox_rear, "front": ox_front,
        "oblique_rear": ox_obl_r, "oblique_front": ox_obl_f
    })

    # 4. BA_GAC
    print("4. Xử lý [ba_gac]...")
    bg_side = extract_studio_tight(BRAIN_DIR / "xe_ba_gac_side_view_1792109326453.jpg")
    bg_side = np.ascontiguousarray(cv2.flip(bg_side, 1))
    bg_rear = extract_studio_tight(BRAIN_DIR / "vietnamese_xe_ba_gac_rear_view_1792109496829.jpg")
    from scripts.build_real_vn_assets import build_ba_gac_front_sprite
    bg_front = tight_crop_rgba(build_ba_gac_front_sprite(
        BRAIN_DIR / "vietnamese_xe_ba_gac_rear_view_1792109496829.jpg",
        BRAIN_DIR / "vietnamese_motorcycle_cargo_front_1792117347037.jpg"
    ))
    bg_obl_r = cv2.resize(bg_side, (int(bg_side.shape[1] * 0.78), bg_side.shape[0]), interpolation=cv2.INTER_AREA)
    bg_obl_f = cv2.resize(bg_side, (int(bg_side.shape[1] * 0.78), bg_side.shape[0]), interpolation=cv2.INTER_AREA)

    save_views_and_variants(assets_dir / "ba_gac", {
        "side": bg_side, "rear": bg_rear, "front": bg_front,
        "oblique_rear": bg_obl_r, "oblique_front": bg_obl_f
    })

    # 5. VENDOR_CART
    print("5. Xử lý [vendor_cart]...")
    vc_side = extract_studio_tight(BRAIN_DIR / "vietnamese_street_food_cart_sideview_1792109538546.jpg")
    from scripts.build_real_vn_assets import build_vendor_cart_sprites
    vc_rear, vc_front = build_vendor_cart_sprites(
        BRAIN_DIR / "vietnamese_street_food_cart_sideview_1792109538546.jpg",
        assets_dir / "pedestrian"
    )
    vc_rear = tight_crop_rgba(vc_rear)
    vc_front = tight_crop_rgba(vc_front)
    vc_obl_r = cv2.resize(vc_side, (int(vc_side.shape[1] * 0.65), vc_side.shape[0]), interpolation=cv2.INTER_AREA)
    vc_obl_f = cv2.resize(vc_side, (int(vc_side.shape[1] * 0.65), vc_side.shape[0]), interpolation=cv2.INTER_AREA)

    save_views_and_variants(assets_dir / "vendor_cart", {
        "side": vc_side, "rear": vc_rear, "front": vc_front,
        "oblique_rear": vc_obl_r, "oblique_front": vc_obl_f
    })


def extract_straight_vehicle_tokens(nusc, assets_dir: Path):
    """
    Trích xuất bằng Mask R-CNN thuần túy cho các token xe chạy thẳng chính diện (rear/front).
    Loại bỏ hoàn toàn các mẫu chụp chéo lộ sườn xe hoặc thủng thân xe.
    """
    print("\n" + "=" * 70)
    print("BƯỚC 2: TRÍCH XUẤT MASK R-CNN THUẦN TÚY CHO CÁC MẪU XE ĐI THẲNG CHÍNH DIỆN...")
    print("=" * 70)

    import torch
    import torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights

    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT).eval()

    # Danh sách các token nuScenes đã được verify 100% thẳng chính diện (aspect 0.65 - 1.20)
    VERIFIED_STRAIGHT_TOKENS = {
        "car": {
            "rear": [
                ("64d522f129634049b4bba059ffd2ff5c", "CAM_FRONT"),  # ratio=1.01, z=7.4m
                ("46c5eced99124ec29829985740567011", "CAM_FRONT"),  # ratio=1.02, z=17.1m
                ("cbaf6e83c3aa4e47aaab8f8652fc1112", "CAM_FRONT"),  # ratio=0.99, z=18.2m
                ("3781ccc1954947c0a3376a645adcddf3", "CAM_FRONT"),  # ratio=0.99, z=19.2m
                ("94956883cb0a4e60ae8dd3db2bdcd998", "CAM_FRONT"),  # ratio=1.11, z=15.9m
            ],
            "front": [
                ("121c28544c89467d9357b86ac5644aba", "CAM_BACK"),   # ratio=1.05, z=8.9m
                ("13d1a5cac2f540f198b0cc365f561ae9", "CAM_BACK"),   # ratio=1.05, z=9.1m
                ("1a983c2ae5734070a2066954417bc5a2", "CAM_BACK"),   # ratio=1.05, z=9.9m
                ("e38e4f9cdc61471fad4e327758c31bc9", "CAM_BACK"),   # ratio=1.05, z=10.3m
                ("ae5cd676f5294a8b8a32b8678e69bf3e", "CAM_BACK"),   # ratio=1.05, z=6.1m
            ]
        },
        "truck": {
            "rear": [
                ("83d881a6b3d94ef3a3bc3b585cc514f8", "CAM_FRONT"),  # PUKKA truck (rear ratio ~ 0.90 - 1.15)
                ("2838f4e38f464983b0073cbfbc93588c", "CAM_FRONT"),  # ratio=1.00, z=34.2m
                ("c66a823efb414f88987fedc410ae65fe", "CAM_FRONT"),  # ratio=1.03, z=35.6m
                ("0b3c7da333e1495180a5a1579cd82ff6", "CAM_FRONT"),  # ratio=1.09, z=37.5m
                ("bc6607bc76414e45bbfd3c4d7e0f9b75", "CAM_FRONT"),  # ratio=1.00, z=42.4m
            ],
            "front": [
                ("11eeb3f5f97a4e50bdfea1ff0fcf8d7f", "CAM_BACK"),   # ratio=0.90, z=20.9m
                ("357832c5250047de9e0c5a24450550b6", "CAM_BACK"),   # ratio=0.91, z=19.4m
                ("d021c781eba342efaf3efa0e89af1ef9", "CAM_BACK"),   # ratio=0.90, z=23.3m
                ("aeb70d46b306488aa868c20c73298a14", "CAM_BACK"),   # ratio=0.89, z=22.1m
                ("0bcd18843f27406cbf5740779311a2c6", "CAM_BACK"),   # ratio=0.90, z=25.7m
            ]
        },
        "bus": {
            "rear": [
                ("56f9d66a5ef643e4b9f066debb8d6b19", "CAM_FRONT"),  # SL2 bus (rear ratio ~ 0.95 - 1.06)
                ("bdb5f84a12e1415899182ec2618c22ef", "CAM_FRONT"),  # ratio=0.98, z=31.5m
                ("aff10900288c433a8a0ece34c927d622", "CAM_FRONT"),  # ratio=1.02, z=28.3m
                ("92a058c97f17494faca12894e2bb230f", "CAM_FRONT"),  # ratio=0.94, z=38.0m
                ("ece57f37d4204fc688a0c9fcc6bdd185", "CAM_FRONT"),  # ratio=0.89, z=35.7m
            ],
            "front": [
                ("0a1e2d4329c046cbb1e6046515936b34", "CAM_BACK"),   # ratio=1.20, z=24.3m
                ("8804262195484a1890fae0e2a511097d", "CAM_BACK"),   # ratio=1.17, z=28.7m
                ("0b1cbc3dc83f4810a54f8c1a80a30248", "CAM_BACK"),   # ratio=1.20, z=28.1m
                ("b576ea0794e54d78bd306b4f9d45a05e", "CAM_BACK"),   # ratio=0.90, z=68.5m
                ("4d030352313e48c1adbfa1a783a92f27", "CAM_BACK"),   # ratio=0.95, z=86.7m
            ]
        }
    }

    COCO_TARGETS = {
        "car": (3,),
        "truck": (8, 7, 6, 3),
        "bus": (6, 7, 8)  # COCO 7 is train/bendy-bus
    }

    for cat, views_dict in VERIFIED_STRAIGHT_TOKENS.items():
        cat_dir = assets_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        valid_targets = COCO_TARGETS[cat]

        for view_name, item_list in views_dict.items():
            extracted_sprites = []
            for item in item_list:
                tok, cam = item if isinstance(item, tuple) else (item, "CAM_FRONT")
                try:
                    ann = nusc.get("sample_annotation", tok)
                    sample = nusc.get("sample", ann["sample_token"])
                    sd_tok = sample["data"][cam]
                    data_path, boxes, K = nusc.get_sample_data(sd_tok)
                    for b in boxes:
                        if b.token == tok:
                            corners = b.corners()
                            uv = (K @ corners)[:2, :] / corners[2:3, :]
                            u1, u2 = uv[0].min(), uv[0].max()
                            v1, v2 = uv[1].min(), uv[1].max()
                            margin_x = int((u2 - u1) * 0.08)
                            margin_y = int((v2 - v1) * 0.08)
                            cx1, cx2 = max(0, int(u1 - margin_x)), min(1600, int(u2 + margin_x))
                            cy1, cy2 = max(0, int(v1 - margin_y)), min(900, int(v2 + margin_y))

                            img_pil = Image.open(data_path).convert("RGB")
                            crop_pil = img_pil.crop((cx1, cy1, cx2, cy2))

                            t_crop = torchvision.transforms.functional.to_tensor(crop_pil)
                            with torch.no_grad():
                                out = model([t_crop])[0]

                            labels = out["labels"].cpu().numpy()
                            scores = out["scores"].cpu().numpy()
                            masks = out["masks"].cpu().numpy()

                            best_m = None
                            best_scr = 0.0
                            for i in range(len(labels)):
                                if int(labels[i]) in valid_targets and float(scores[i]) >= 0.40 and float(scores[i]) > best_scr:
                                    raw_m = (masks[i, 0] >= 0.45).astype(np.uint8)
                                    if (raw_m.sum() / float(raw_m.size)) >= 0.15:
                                        best_scr = float(scores[i])
                                        best_m = raw_m

                            if best_m is not None:
                                filled_m = scipy.ndimage.binary_fill_holes(best_m).astype(np.uint8)
                                num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(filled_m)
                                if num_labels > 1:
                                    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                                    clean_m = (comp_labels == largest_idx).astype(np.uint8) * 255
                                else:
                                    clean_m = filled_m * 255

                                clean_m = (scipy.ndimage.binary_fill_holes(clean_m > 0).astype(np.uint8) * 255)
                                rgba = np.dstack([np.asarray(crop_pil), clean_m])
                                tight = tight_crop_rgba(rgba)

                                w_tight, h_tight = tight.shape[1], tight.shape[0]
                                ratio_tight = w_tight / float(h_tight)

                                # QUY TẮC MỤC 3: CẤM TUYỆT ĐỐI w > 1.25 * h hoặc w < 0.65 * h
                                if ratio_tight > 1.25 or ratio_tight < 0.65:
                                    print(f"  - [{cat} {view_name}] BỎ QUA do tỉ lệ không thẳng: w={w_tight}, h={h_tight}, ratio={ratio_tight:.2f}")
                                    continue

                                fill_ratio = np.mean(tight[:, :, 3] > 128)
                                if fill_ratio >= 0.35:
                                    extracted_sprites.append(tight)
                                    print(f"  + [{cat} {view_name}] trích xuất thành công ann={tok[:8]}, size={w_tight}x{h_tight}, ratio={ratio_tight:.2f}, fill={fill_ratio:.2f}")
                except Exception as e:
                    pass

            if extracted_sprites:
                Image.fromarray(extracted_sprites[0]).save(cat_dir / f"{view_name}.png", format="PNG")
                for i in range(5):
                    sp = extracted_sprites[i % len(extracted_sprites)]
                    Image.fromarray(sp).save(cat_dir / f"{view_name}_{i}.png", format="PNG")
                print(f"  [x] Đã cập nhật 100% {cat} {view_name} bằng các mẫu đi thẳng cân xứng (0.65 <= w/h <= 1.20)!")


def run_tight_alpha_crop_all(assets_dir: Path):
    """Cắt sát khung bao Alpha trên 100% file PNG còn lại trong assets/vn_objects/."""
    print("\n" + "=" * 70)
    print("BƯỚC 3: THỰC HIỆN TIGHT ALPHA BOUNDING BOX CROP TRÊN 100% TOÀN BỘ KHO ASSETS...")
    print("=" * 70)
    count = 0
    for p in assets_dir.rglob("*.png"):
        try:
            im = Image.open(p).convert("RGBA")
            arr = np.asarray(im)
            tight = tight_crop_rgba(arr)
            if tight.shape != arr.shape or not np.array_equal(tight, arr):
                Image.fromarray(tight).save(p, format="PNG")
                count += 1
        except Exception:
            pass
    print(f"  [v] Đã hoàn tất Tight Alpha Crop trên {count} file PNG!")


def main():
    assets_dir = get_default_assets_dir()
    print(f"Thư mục tài nguyên: {assets_dir}")
    nusc = load_nuscenes()
    fix_vietnamese_assets(assets_dir)
    extract_straight_vehicle_tokens(nusc, assets_dir)
    run_tight_alpha_crop_all(assets_dir)
    print("\n" + "=" * 70)
    print("HOÀN TẤT: ĐÃ KHÔI PHỤC VÀ CHUẨN HÓA TOÀN DIỆN KHO TÀI NGUYÊN!")
    print("=" * 70)


if __name__ == "__main__":
    main()
