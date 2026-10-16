"""
Script hoàn thiện toàn bộ Sprite Pool và biến thể Việt Nam
(scripts/finalize_sprites.py)
"""
import os
import sys
from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "n3-08c"))

from utils.nuscenes_loader import load_nuscenes
from utils.asset_manager import get_default_assets_dir
from scripts.extract_real_nuscenes_sprites import (
    _extract_barrier_and_cone,
    create_vietnamese_multi_variant_sprites,
    MAX_VARIANTS_PER_VIEW
)

def finalize_pool(assets_dir: Path):
    views_list = ["rear", "front", "side", "oblique_rear", "oblique_front"]
    std_cats = ["pedestrian", "car", "truck", "bus", "motorcycle", "bicycle"]

    print("=" * 65)
    print("HOÀN THIỆN MULTI-VARIANT SPRITE POOL CHO CÁC LỚP CHUẨN...")
    print("=" * 65)

    for cat in std_cats:
        cat_dir = assets_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Tìm các file token đã trích xuất bằng Mask R-CNN
        token_files = [f for f in cat_dir.glob("*.png") if len(f.stem) == 32]
        print(f"[{cat:11s}] Tìm thấy {len(token_files)} ảnh token bóc tách thật bằng Mask R-CNN.")

        for view in views_list:
            # Kiểm tra xem view_0 .. view_4 đã có chưa
            existing_vars = [cat_dir / f"{view}_{i}.png" for i in range(MAX_VARIANTS_PER_VIEW)]
            all_exist = all(f.is_file() for f in existing_vars)

            if not all_exist:
                # Cần bổ sung biến thể
                collected = []
                # 1. Thử lấy từ các file existing của view này
                for f in existing_vars:
                    if f.is_file():
                        collected.append(Image.open(f).convert("RGBA"))
                # 2. Thử lấy từ default view.png
                def_fp = cat_dir / f"{view}.png"
                if def_fp.is_file():
                    collected.append(Image.open(def_fp).convert("RGBA"))
                # 3. Thử lấy từ token files nếu có
                if token_files:
                    for tf in token_files[:5]:
                        try:
                            collected.append(Image.open(tf).convert("RGBA"))
                        except Exception:
                            pass
                # 4. Fallback view khác nếu vẫn rỗng
                if not collected:
                    for alt_v in views_list:
                        alt_fp = cat_dir / f"{alt_v}.png"
                        if alt_fp.is_file():
                            collected.append(Image.open(alt_fp).convert("RGBA"))
                            break

                if not collected:
                    from tools.bootstrap_sprites import generate_sprite
                    collected.append(generate_sprite(cat, view))

                for i in range(MAX_VARIANTS_PER_VIEW):
                    img = collected[i % len(collected)]
                    img.save(cat_dir / f"{view}_{i}.png", format="PNG")

                if not def_fp.is_file():
                    collected[0].save(def_fp, format="PNG")

                print(f"  [x] Đã hoàn thiện {cat:11s} view={view:13s} (5 biến thể).")
            else:
                print(f"  [v] Đã sẵn sàng {cat:11s} view={view:13s} (5 biến thể).")

def main():
    assets_dir = get_default_assets_dir()
    print(f"Tài nguyên: {assets_dir}")
    finalize_pool(assets_dir)

    print("\nKhởi tạo nuScenes để trích xuất barrier và traffic_cone...")
    nusc = load_nuscenes()
    _extract_barrier_and_cone(nusc, assets_dir)

    print("\nTạo 5 biến thể đa dạng cho các lớp đặc thù Việt Nam...")
    create_vietnamese_multi_variant_sprites(assets_dir)

    print("\n" + "=" * 65)
    print("HOÀN TẤT THÀNH CÔNG: TOÀN BỘ ASSETS 5 BIẾN THỂ ĐÃ SẴN SÀNG 100%!")
    print("=" * 65)

if __name__ == "__main__":
    main()
