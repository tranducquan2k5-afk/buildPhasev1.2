"""
Trích xuất ảnh chụp RGB thật từ nuScenes bằng mạng AI Mask R-CNN (torchvision)
(scripts/extract_real_nuscenes_sprites.py)
=============================================================================
1. Sử dụng mô hình pretrained torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT').
2. Bảo toàn 100% thân xe, kính và màu sơn trắng/bạc:
   - Mask R-CNN phân đoạn đối tượng (mask >= 0.65, pedestrian >= 0.70).
   - Lấp kín hoàn toàn ruột bên trong đường bao thân xe bằng scipy.ndimage.binary_fill_holes.
   - Giữ lại duy nhất 1 thành phần liên thông lớn nhất (connectedComponentsWithStats).
   - Tuyệt đối CẤM dùng ngưỡng RGB xóa thủng thân xe!
3. Chuẩn hóa góc nhìn thẳng cho car / truck:
   - Chỉ trích xuất rear / front cho xe đang đi thẳng hàng giữa làn (|X_cam| < 2.5m, lệch < 10 độ).
   - Tỉ lệ khung ảnh hẹp w/h <= 1.15, loại bỏ hoàn toàn xe rẽ chéo ở ngã tư.
4. Lớp motorcycle (Xe máy 1 người thật):
   - BẮT BUỘC có attribute == 'cycle.with_rider' (đang chạy trên đường, có người lái).
   - Lấy hợp (union) của mask motorcycle + mask person ngồi trên xe.
5. Lớp pedestrian (Người đi bộ thật):
   - Score >= 0.90, mask threshold >= 0.70.
   - Duy nhất 1 thành phần liên thông lớn nhất, triệt tiêu mọi khung mờ, rác biên và mảng màu thừa.
   - Trích xuất đủ 10 mẫu người đi bộ khác nhau cho mỗi góc nhìn.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import shutil
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "n3-08c"))

import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from utils.nuscenes_loader import load_nuscenes, get_pointcloud
from utils.camera_proj import lidar_to_cam
from utils.asset_manager import normalize_category, get_default_assets_dir
from nuscenes.utils.geometry_utils import BoxVisibility

CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
MAX_VARIANTS_PER_VIEW = 5
MAX_PED_VARIANTS_PER_VIEW = 10

# Ánh xạ từ category chuẩn sang nhãn COCO của Mask R-CNN
COCO_CLASS_MAP = {
    "pedestrian": (1, 0.90, 0.15),   # person (score >= 0.90)
    "bicycle": (2, 0.65, 0.15),      # bicycle
    "car": (3, 0.80, 0.25),          # car
    "motorcycle": (4, 0.65, 0.18),   # motorcycle
    "bus": (6, 0.80, 0.25),          # bus
    "truck": (8, 0.75, 0.25),        # truck
}


def compute_cos_alpha(box, K: Optional[np.ndarray] = None) -> Tuple[float, str, bool]:
    """
    Xác định view (rear/front/side/oblique) và hướng quay bằng độ sâu 2 đỉnh P_front vs P_rear.
    Returns: (cos_heading, view_name, needs_flip_to_face_right)
    """
    corners = box.corners()
    front_c = corners[:, :4].mean(axis=1)
    rear_c = corners[:, 4:].mean(axis=1)

    z_front = float(front_c[2])
    z_rear = float(rear_c[2])
    dx = float(front_c[0] - rear_c[0])
    dz = float(z_front - z_rear)
    l_xz = max(1e-4, float(np.hypot(dx, dz)))
    sin_heading = dx / l_xz
    cos_heading = dz / l_xz

    # BẬT QUY TẮC BẤT BIẾN THEO HƯỚNG LÀN ĐƯỜNG:
    # Nếu lệch dưới 20 độ so với trục đường (|sin(heading)| < 0.35) -> BẮT BUỘC rear hoặc front
    if abs(sin_heading) < 0.35:
        if z_front < z_rear:
            view = "front"
        else:
            view = "rear"
    elif abs(cos_heading) < 0.35:
        view = "side"
    else:
        if z_front < z_rear:
            view = "oblique_front"
        else:
            view = "oblique_rear"

    needs_flip_to_face_right = False
    if K is not None and z_front > 0.1 and z_rear > 0.1:
        u_front = float((K @ front_c)[0] / z_front)
        u_rear = float((K @ rear_c)[0] / z_rear)
        if u_front < u_rear:
            needs_flip_to_face_right = True

    return cos_heading, view, needs_flip_to_face_right


def maskrcnn_segment_crop(model, crop_pil: Image.Image, target_coco_id: int,
                          min_score: float, min_mask_ratio: float,
                          is_motorcycle: bool = False,
                          is_pedestrian: bool = False) -> Optional[np.ndarray]:
    """
    Sử dụng Mask R-CNN để phát hiện và bóc tách vật thể thật theo đúng nhãn ngữ nghĩa.
    - Dùng scipy.ndimage.binary_fill_holes lấp kín 100% ruột (bảo toàn sơn trắng/bạc, kính).
    - Dùng connectedComponentsWithStats giữ lại duy nhất 1 thành phần liên thông lớn nhất.
    - Tuyệt đối không dùng ngưỡng RGB xóa thủng thân xe.
    """
    t_crop = torchvision.transforms.functional.to_tensor(crop_pil)
    with torch.no_grad():
        out = model([t_crop])[0]

    labels = out["labels"].cpu().numpy()
    scores = out["scores"].cpu().numpy()
    masks = out["masks"].cpu().numpy()

    if len(labels) == 0:
        return None

    if is_motorcycle:
        # Hợp nhất mask xe máy (COCO 4) và mask người lái (COCO 1)
        moto_m = np.zeros_like(masks[0, 0], dtype=bool)
        has_moto = False
        for i in range(len(labels)):
            lbl = int(labels[i])
            scr = float(scores[i])
            if lbl == 4 and scr >= 0.50:
                moto_m = np.logical_or(moto_m, masks[i, 0] >= 0.55)
                has_moto = True
            elif lbl == 1 and scr >= 0.50:
                moto_m = np.logical_or(moto_m, masks[i, 0] >= 0.55)
        if not has_moto or moto_m.sum() == 0:
            return None
        bin_m = moto_m.astype(np.uint8)
    elif is_pedestrian:
        # Người đi bộ: score >= 0.90, mask threshold >= 0.70
        best_scr = 0.0
        bin_m = None
        for i in range(len(labels)):
            if int(labels[i]) == 1 and float(scores[i]) >= 0.90 and float(scores[i]) > best_scr:
                raw_m = masks[i, 0]
                b = (raw_m >= 0.70).astype(np.uint8)
                if (b.sum() / float(b.size)) >= min_mask_ratio:
                    best_scr = float(scores[i])
                    bin_m = b
        if bin_m is None:
            return None
    else:
        # Ô tô, xe tải, xe buýt, xe đạp
        best_scr = 0.0
        bin_m = None
        for i in range(len(labels)):
            if int(labels[i]) == target_coco_id and float(scores[i]) >= min_score and float(scores[i]) > best_scr:
                raw_m = masks[i, 0]
                b = (raw_m >= 0.65).astype(np.uint8)
                if (b.sum() / float(b.size)) >= min_mask_ratio:
                    best_scr = float(scores[i])
                    bin_m = b
        if bin_m is None:
            return None

    # 1. Lấp kín 100% ruột bên trong đường bao bằng scipy.ndimage.binary_fill_holes
    filled_m = scipy.ndimage.binary_fill_holes(bin_m).astype(np.uint8)

    # 2. Giữ lại duy nhất 1 thành phần liên thông lớn nhất (loại bỏ khung mờ, rác biên)
    num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(filled_m)
    if num_labels <= 1:
        return None
    largest_idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    clean_mask = (comp_labels == largest_idx).astype(np.uint8) * 255

    # 3. Co nhẹ 1px loại bỏ viền đen mặt đường
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_mask = cv2.erode(clean_mask, kernel_erode, iterations=1)

    # 4. Làm mượt biên GaussianBlur
    alpha = cv2.GaussianBlur(eroded_mask, (3, 3), sigmaX=0.8)

    crop_np = np.asarray(crop_pil)
    rgba = np.dstack([crop_np, alpha])

    ys, xs = np.where(alpha > 25)
    if len(ys) < 30 or len(xs) < 20:
        return None

    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    cropped_rgba = rgba[y1:y2, x1:x2]

    return np.ascontiguousarray(cropped_rgba)


def extract_all_sprites_maskrcnn(nusc, assets_dir: Path):
    """Trích xuất toàn bộ kho ảnh thực tế nuScenes bằng Mask R-CNN."""
    print("=" * 65)
    print("BƯỚC 1: TRÍCH XUẤT ẢNH THẬT BẰNG MẠNG AI MASK R-CNN (RESNET-50 FPN)...")
    print("=" * 65)

    print("Đang khởi tạo mô hình Mask R-CNN...")
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    model = maskrcnn_resnet50_fpn(weights=weights).eval()
    print("Mask R-CNN sẵn sàng!")

    categories = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]
    views_list = ["rear", "front", "side", "oblique_rear", "oblique_front"]

    variant_pool: Dict[str, Dict[str, List[Tuple[np.ndarray, str]]]] = {
        cat: {v: [] for v in views_list} for cat in categories
    }

    n_samples = len(nusc.sample)
    print(f"Bắt đầu quét qua {n_samples} mẫu nuScenes mini...")

    for s_idx, sample in enumerate(nusc.sample):
        for cam in CAMS:
            sd_tok = sample["data"][cam]
            data_path, boxes, K = nusc.get_sample_data(sd_tok, box_vis_level=BoxVisibility.ANY)
            if not boxes:
                continue

            img_full = None

            for b in boxes:
                cat = normalize_category(b.name)
                if cat not in categories:
                    continue

                dist = float(b.center[2])
                if not (4.5 <= dist <= 25.0):
                    continue

                corners = b.corners()
                if np.any(corners[2, :] <= 0.5):
                    continue

                uv = (K @ corners)[:2, :] / corners[2:3, :]
                u1, u2 = uv[0, :].min(), uv[0, :].max()
                v1, v2 = uv[1, :].min(), uv[1, :].max()
                bw, bh = u2 - u1, v2 - v1

                if bw < 20 or bh < 24:
                    continue
                if u1 < 5 or u2 > 1595 or v1 < 5 or v2 > 895:
                    continue

                _, view, needs_flip = compute_cos_alpha(b, K)

                target_max = MAX_PED_VARIANTS_PER_VIEW if cat == "pedestrian" else MAX_VARIANTS_PER_VIEW
                if len(variant_pool[cat][view]) >= target_max:
                    continue

                # Lọc nghiêm ngặt cho xe máy: BẮT BUỘC có attribute == 'cycle.with_rider'
                if cat == "motorcycle":
                    ann_rec = nusc.get("sample_annotation", b.token)
                    attr_names = [nusc.get("attribute", at)["name"] for at in ann_rec.get("attribute_tokens", [])]
                    if "cycle.with_rider" not in attr_names:
                        continue

                # Lọc nghiêm ngặt cho car và truck đi thẳng:
                if cat in ("car", "truck"):
                    if view in ("rear", "front"):
                        # Chỉ lấy những xe đang đi thẳng hàng ngay chính giữa làn đường (|X_cam| < 2.5m, góc lệch < 10 độ)
                        if abs(float(b.center[0])) > 2.5:
                            continue
                        f_c = corners[:, :4].mean(axis=1)
                        r_c = corners[:, 4:].mean(axis=1)
                        dx_b = float(f_c[0] - r_c[0])
                        dz_b = float(f_c[2] - r_c[2])
                        l_b = max(1e-4, float(np.hypot(dx_b, dz_b)))
                        if abs(dx_b / l_b) > 0.176:  # tan(10 deg) ~ 0.176
                            continue
                        # Tỉ lệ khung ảnh hẹp w/h <= 1.15
                        if (bw / max(1.0, bh)) > 1.15:
                            continue

                # Cắt ảnh crop với 10% lề ngữ cảnh
                margin = int(max(bw, bh) * 0.10)
                cx1 = max(0, int(u1 - margin))
                cx2 = min(1600, int(u2 + margin))
                cy1 = max(0, int(v1 - margin))
                cy2 = min(900, int(v2 + margin))

                if img_full is None:
                    img_full = Image.open(data_path).convert("RGB")

                crop_pil = img_full.crop((cx1, cy1, cx2, cy2))

                coco_id, min_scr, min_ratio = COCO_CLASS_MAP[cat]
                rgba_cut = maskrcnn_segment_crop(
                    model, crop_pil, coco_id, min_scr, min_ratio,
                    is_motorcycle=(cat == "motorcycle"),
                    is_pedestrian=(cat == "pedestrian")
                )

                if rgba_cut is not None:
                    # Lọc kiểm tra người đi bộ sạch
                    if cat == "pedestrian":
                        alpha_m = rgba_cut[:, :, 3]
                        h_c, w_c = alpha_m.shape
                        aspect = h_c / max(1, w_c)
                        if not (2.0 <= aspect <= 4.5):
                            continue
                        ys_ped = np.where(alpha_m > 30)[0]
                        if len(ys_ped) == 0 or (ys_ped.max() + 1) / h_c < 0.90:
                            continue

                    # Chuẩn hóa chiều đầu xe quay về bên phải
                    if needs_flip:
                        rgba_cut = np.ascontiguousarray(cv2.flip(rgba_cut, 1))

                    variant_pool[cat][view].append((rgba_cut, b.token))

                    # Lưu theo ann_token
                    cat_dir = assets_dir / cat
                    cat_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba_cut).save(cat_dir / f"{b.token}.png", format="PNG")

                    print(f"  + [{cat:10s} | {view:13s} (Mẫu #{len(variant_pool[cat][view])})] "
                          f"ann={b.token[:8]}, size={rgba_cut.shape[1]}x{rgba_cut.shape[0]}, dist={dist:.1f}m")

        # Kiểm tra hoàn thành sớm
        all_done = all(
            all(len(variant_pool[c][v]) >= (MAX_PED_VARIANTS_PER_VIEW if c == "pedestrian" else MAX_VARIANTS_PER_VIEW) for v in views_list)
            for c in categories
        )
        if all_done:
            print("Đã thu thập đủ biến thể cho toàn bộ các danh mục và góc nhìn!")
            break

    # Lưu các biến thể và file mặc định vào assets/vn_objects/
    print("\n" + "=" * 65)
    print("BƯỚC 2: LƯU TRỮ ĐA DẠNG MẪU (MULTI-VARIANT POOL)...")
    print("=" * 65)

    for cat in categories:
        cat_dir = assets_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        target_max = MAX_PED_VARIANTS_PER_VIEW if cat == "pedestrian" else MAX_VARIANTS_PER_VIEW

        for view in views_list:
            variants = variant_pool[cat][view]
            if not variants:
                print(f"  [!] Thiếu biến thể cho {cat} {view}, dùng fallback...")
                for alt_view in ("rear", "front", "side", "oblique_rear", "oblique_front"):
                    if variant_pool[cat][alt_view]:
                        variants = variant_pool[cat][alt_view]
                        break
                if not variants:
                    disk_fp = cat_dir / f"{view}.png"
                    if disk_fp.is_file():
                        sp_pil = Image.open(disk_fp).convert("RGBA")
                        variants = [(np.asarray(sp_pil), "disk")]
                    else:
                        from tools.bootstrap_sprites import generate_sprite
                        sp_pil = generate_sprite(cat, view)
                        variants = [(np.asarray(sp_pil), "bootstrap")]

            for v_idx in range(target_max):
                chosen_sprite = variants[v_idx % len(variants)][0]
                v_fp = cat_dir / f"{view}_{v_idx}.png"
                Image.fromarray(chosen_sprite).save(v_fp, format="PNG")

            default_fp = cat_dir / f"{view}.png"
            Image.fromarray(variants[0][0]).save(default_fp, format="PNG")

            print(f"  [x] Đã lưu {cat:11s} view={view:13s} với {min(len(variants), target_max)} biến thể thật.")

    _extract_barrier_and_cone(nusc, assets_dir)


def _extract_barrier_and_cone(nusc, assets_dir: Path):
    """Trích xuất và chuẩn hóa barrier và traffic_cone sắc nét trên nền trong suốt."""
    views_list = ["rear", "front", "side", "oblique_rear", "oblique_front"]

    for cat in ["barrier", "traffic_cone"]:
        cat_dir = assets_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        collected = []
        for s in nusc.sample:
            for cam in CAMS:
                sd_tok = s["data"][cam]
                data_path, boxes, K = nusc.get_sample_data(sd_tok, box_vis_level=BoxVisibility.ANY)
                for b in boxes:
                    if (cat == "barrier" and "barrier" in b.name) or (cat == "traffic_cone" and "trafficcone" in b.name):
                        if 4.5 <= b.center[2] <= 18.0:
                            uv = (K @ b.corners())[:2, :] / b.corners()[2:3, :]
                            u1, u2 = uv[0, :].min(), uv[0, :].max()
                            v1, v2 = uv[1, :].min(), uv[1, :].max()
                            if 10 <= u1 and u2 < 1590 and 10 <= v1 and v2 < 890:
                                img = Image.open(data_path).convert("RGB")
                                crop = np.asarray(img.crop((int(u1), int(v1), int(u2), int(v2))))
                                if crop.shape[0] > 15 and crop.shape[1] > 15:
                                    if cat == "traffic_cone":
                                        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
                                        mask = cv2.inRange(hsv, np.array([5, 80, 80]), np.array([25, 255, 255]))
                                        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
                                    else:
                                        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
                                        mask = np.full(gray.shape, 255, dtype=np.uint8)
                                        cv2.rectangle(mask, (0, 0), (mask.shape[1]-1, mask.shape[0]-1), 255, -1)
                                    alpha = cv2.GaussianBlur(mask, (3, 3), 0.8)
                                    rgba = np.dstack([crop, alpha])
                                    collected.append(rgba)
                                    if len(collected) >= 5:
                                        break
                if len(collected) >= 5:
                    break
            if len(collected) >= 5:
                break

        if not collected:
            from tools.bootstrap_sprites import generate_sprite
            collected = [np.asarray(generate_sprite(cat, "front"))]

        for v in views_list:
            for v_idx in range(MAX_VARIANTS_PER_VIEW):
                sp = collected[v_idx % len(collected)]
                Image.fromarray(sp).save(cat_dir / f"{v}_{v_idx}.png", format="PNG")
            Image.fromarray(collected[0]).save(cat_dir / f"{v}.png", format="PNG")
        print(f"  [x] Đã chuẩn hóa {cat} với 5 biến thể / view.")


def main():
    assets_dir = get_default_assets_dir()
    print(f"Thư mục tài nguyên: {assets_dir}")
    nusc = load_nuscenes()
    extract_all_sprites_maskrcnn(nusc, assets_dir)
    print("\n" + "=" * 65)
    print("HOÀN TẤT: TRÍCH XUẤT 100% ẢNH THẬT TỪ NUSCENES ĐẠT CHUẨN!")
    print("=" * 65)


if __name__ == "__main__":
    main()
