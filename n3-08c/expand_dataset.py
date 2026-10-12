"""Script mở rộng dữ liệu (Batch Dataset Expansion) cho nuScenes.

Tạo ra hàng loạt scene nhân tạo bằng cách kết hợp:
1. Chèn vật thể ngẫu nhiên (Task A) với nhiều cấu hình số lượng và vị trí.
2. Áp dụng các điều kiện thời tiết (Task B: mưa, sương mù, mặt đường ướt).

Chạy:
    python expand_dataset.py --samples 20 --objects 3 --weather rain --weather-level moderate
    python expand_dataset.py --samples 50 --all-weather
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from task_a.extract_object import TemplateBank
from task_a.ground_plane import GroundModel, get_map_area, layers_for_category, load_map_for_sample, project_z_to_ground
from task_a.merge import create_annotation, insert_object
from task_a.placement import sample_valid_position
from task_b.fog import simulate_fog
from task_b.rain import simulate_rain
from task_b.wet_road import simulate_puddles
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose, get_lidar_token,
                                   get_pointcloud, load_nuscenes, save_bin)
from utils.transforms import box_sensor_to_global


def expand_dataset(nusc, n_samples: int = 20, min_objects: int = 2, max_objects: int = 5,
                   weather_type: str = "none", weather_level: str = "moderate",
                   output_dir: str = os.path.join(config.OUTPUT_DIR, "expanded_dataset"),
                   seed: int = 42):
    """Mở rộng dữ liệu quy mô lớn."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    # Load hoặc chuẩn bị kho template
    bank = TemplateBank()
    if bank.exists():
        bank.load()
    else:
        print("[Notice] TemplateBank chưa được build. Khởi tạo từ các sample có sẵn...")
        bank.build(nusc, progress=False)

    samples = nusc.sample[:n_samples]
    print(f"Bắt đầu mở rộng dữ liệu: {len(samples)} samples gốc...")
    print(f"Cấu hình: Vật thể/scene: {min_objects}-{max_objects} | Thời tiết: {weather_type} ({weather_level})")
    
    total_generated = 0
    t0 = time.perf_counter()

    for idx, s in enumerate(tqdm(samples, desc="Mở rộng dữ liệu")):
        target_token = s["token"]
        points = get_pointcloud(nusc, target_token)
        existing = get_annotations(nusc, target_token)
        ground = GroundModel.from_points(points, rng=rng)
        nusc_map = load_map_for_sample(nusc, target_token)
        
        # 1. Chọn số lượng và lấy ngẫu nhiên template từ kho
        k_objects = rng.integers(min_objects, max_objects + 1)
        templates = bank.sample(k_objects, rng=rng, balance=True)
        
        new_anns = []
        areas = {}
        for tpl in templates:
            layers = layers_for_category(tpl["category"])
            if layers not in areas:
                areas[layers] = get_map_area(nusc, nusc_map, target_token, layers)
                
            orig_dist = tpl.get("distance", -1)
            d_range = (orig_dist - 5.0, orig_dist + 5.0) if orig_dist > 0 else None
            
            pos = sample_valid_position(areas[layers], existing + new_anns, tpl["size"],
                                        n_attempts=100, rng=rng, distance_range=d_range)
            if pos is None:
                continue
                
            points = insert_object(points, tpl, pos, ground)
            new_anns.append(create_annotation(tpl, pos, project_z_to_ground(pos[0], pos[1], ground)))
            
        # 2. Phủ thời tiết nếu được yêu cầu
        if weather_type == "rain":
            points = simulate_rain(points, intensity_level=weather_level, rng=rng)
            # Nếu mưa to, kích hoạt thêm mặt đường ướt / vũng nước
            if weather_level in ("moderate", "heavy"):
                points = simulate_puddles(points, num_puddles=10, rng=rng)
        elif weather_type == "fog":
            vis_val = config.FOG_VISIBILITY.get(weather_level, 50.0)
            points = simulate_fog(points, visibility=vis_val, rng=rng)
        elif weather_type == "puddles":
            points = simulate_puddles(points, num_puddles=15, rng=rng)

        # 3. Lưu kết quả ra thư mục mở rộng
        file_prefix = f"syn_{idx:04d}_{target_token[:8]}_{weather_type}"
        bin_path = os.path.join(output_dir, f"{file_prefix}.bin")
        json_path = os.path.join(output_dir, f"{file_prefix}.json")
        
        save_bin(bin_path, points)
        
        sd_token = get_lidar_token(nusc, target_token)
        calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)
        
        meta = {
            "synthetic_id": file_prefix,
            "base_sample_token": target_token,
            "weather_type": weather_type,
            "weather_level": weather_level,
            "num_inserted_objects": len(new_anns),
            "annotations_sensor": new_anns,
            "annotations_global": [box_sensor_to_global(a, calib, pose) for a in new_anns],
            "total_lidar_points": len(points)
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=1)
            
        total_generated += 1

    total_time = time.perf_counter() - t0
    print(f"\n[Hoàn thành] Đã tạo thành công {total_generated} scene nhân tạo!")
    print(f"Tổng thời gian: {total_time:.2f}s (~ {total_generated / (total_time / 3600):.1f} scenes/giờ)")
    print(f"Dữ liệu được lưu tại: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mở rộng dữ liệu nuScenes hàng loạt")
    parser.add_argument("--samples", type=int, default=20, help="Số lượng sample gốc muốn dùng để nhân bản")
    parser.add_argument("--min-objects", type=int, default=2, help="Số vật thể tối thiểu chèn vào")
    parser.add_argument("--max-objects", type=int, default=5, help="Số vật thể tối đa chèn vào")
    parser.add_argument("--weather", choices=["none", "rain", "fog", "puddles"], default="none", help="Loại thời tiết")
    parser.add_argument("--weather-level", choices=["light", "moderate", "heavy", "dense"], default="moderate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=os.path.join(config.OUTPUT_DIR, "expanded_dataset"))
    args = parser.parse_args()

    nusc = load_nuscenes()
    expand_dataset(
        nusc=nusc,
        n_samples=args.samples,
        min_objects=args.min_objects,
        max_objects=args.max_objects,
        weather_type=args.weather,
        weather_level=args.weather_level,
        output_dir=args.output,
        seed=args.seed
    )
