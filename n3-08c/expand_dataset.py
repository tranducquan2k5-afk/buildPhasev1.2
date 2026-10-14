"""Script mở rộng dữ liệu (Batch Dataset Expansion) cho nuScenes.

Tạo ra hàng loạt scene nhân tạo bằng cách kết hợp:
1. Chèn vật thể chuẩn quang học (Task A):
   - Template chất lượng cao (Large >= 400 pts & span >= 55%, Car >= 180 pts & span >= 55%).
   - Bám theo hướng làn đường (Road Heading), cách tâm LiDAR >= 8.0m.
   - Buffer an toàn +-1.5m và cách ly lẫn nhau: dist >= 15.0m, |delta theta| >= 40 deg.
   - Khử che khuất Range Image Ray Casting +-15cm và dọn lòng hộp points_in_cuboid.
2. Mô phỏng thời tiết vật lý (Task B: mưa Beer-Lambert, sương mù tán xạ ngược, vũng nước phản xạ gương).

Chạy:
    python expand_dataset.py --samples 20 --min-objects 2 --max-objects 4 --weather rain --weather-level heavy
    python expand_dataset.py --samples 50 --all-weather
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "webapp" / "backend"))

import config
from task_a.ground_plane import (GroundModel, get_map_area, layers_for_category,
                                 load_map_for_sample, project_z_to_ground)
from task_a.merge import create_annotation, insert_object
from task_a.placement import is_mutually_separated, sample_valid_position
from task_b.fog import simulate_fog
from task_b.rain import simulate_rain
from task_b.wet_road import simulate_puddles
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose,
                                   get_lidar_token, get_pointcloud, load_nuscenes,
                                   save_bin)
from utils.transforms import box_sensor_to_global, points_in_cuboid, wrap_angle


def load_filtered_template_bank():
    """Nạp và lọc Template Bank chất lượng cao theo tiêu chuẩn vật lý."""
    bank_path = Path(config.OUTPUT_DIR) / "templates" / "bank.json"
    templates_dir = Path(config.OUTPUT_DIR) / "templates"
    if not bank_path.exists():
        raise FileNotFoundError(f"Template bank not found at {bank_path}")

    with open(bank_path, "r", encoding="utf-8") as f:
        raw_bank = json.load(f)

    filtered = []
    for item in raw_bank:
        cat = item.get("category", "")
        pts = item.get("num_points", 0)
        is_large = any(k in cat for k in ("vehicle.truck", "vehicle.bus", "vehicle.construction", "vehicle.trailer"))
        is_car = (cat == "vehicle.car")

        if is_large:
            if pts < 400:
                continue
            rel = item.get("file", "").replace("\\", "/")
            npz = templates_dir / rel
            if npz.exists():
                try:
                    data = np.load(str(npz))
                    p = data["points"]
                    l = item.get("size", [1.0, 1.0, 1.0])[1]
                    span_ratio = float(p[:, 0].max() - p[:, 0].min()) / (l + 1e-6)
                    if span_ratio >= 0.55:
                        item["span_ratio"] = span_ratio
                        filtered.append(item)
                except Exception:
                    pass
        elif is_car:
            if pts < 180:
                continue
            rel = item.get("file", "").replace("\\", "/")
            npz = templates_dir / rel
            if npz.exists():
                try:
                    data = np.load(str(npz))
                    p = data["points"]
                    l = item.get("size", [1.0, 1.0, 1.0])[1]
                    span_ratio = float(p[:, 0].max() - p[:, 0].min()) / (l + 1e-6)
                    if span_ratio >= 0.55:
                        item["span_ratio"] = span_ratio
                        filtered.append(item)
                except Exception:
                    pass
        elif any(k in cat for k in ("motorcycle", "bicycle", "pedestrian", "barrier", "trafficcone")):
            if pts >= 60:
                filtered.append(item)
        else:
            if pts >= 60:
                filtered.append(item)

    return filtered, templates_dir


def load_template_points(tpl_meta: dict, templates_dir: Path) -> Optional[dict]:
    rel = tpl_meta.get("file", "").replace("\\", "/")
    path = templates_dir / rel
    if not path.exists():
        return None
    try:
        data = np.load(str(path))
        return {
            "points": data["points"],
            "size": tpl_meta["size"],
            "category": tpl_meta["category"],
            "distance": tpl_meta.get("distance", 15.0),
            "ann_token": tpl_meta.get("ann_token", "")
        }
    except Exception:
        return None


def expand_dataset(nusc, n_samples: int = 20, min_objects: int = 2, max_objects: int = 4,
                   weather_type: str = "none", weather_level: str = "moderate",
                   output_dir: str = os.path.join(config.OUTPUT_DIR, "expanded_dataset"),
                   seed: int = 42, all_weather: bool = False):
    """Mở rộng dữ liệu quy mô lớn chuẩn xác theo ngữ cảnh làn đường và cách ly vật lý."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    templates_meta, templates_dir = load_filtered_template_bank()
    samples = nusc.sample[:n_samples]

    print(f"[Start] Expanding dataset: {len(samples)} samples...")
    print(f"Config: Objects/scene: {min_objects}-{max_objects} | Weather: {weather_type} ({weather_level}) | All-Weather: {all_weather}")
    print(f"Filtered dense templates pool: {len(templates_meta)} templates.")

    total_generated = 0
    t0 = time.perf_counter()

    for idx, s in enumerate(tqdm(samples, desc="Mở rộng dữ liệu")):
        target_token = s["token"]
        orig_points = get_pointcloud(nusc, target_token)
        existing = get_annotations(nusc, target_token)
        ground = GroundModel.from_points(orig_points, rng=rng)
        nusc_map = load_map_for_sample(nusc, target_token)

        points_base = orig_points.copy()
        new_anns = []
        areas = {}

        k_objects = int(rng.integers(min_objects, max_objects + 1))
        max_attempts = k_objects * 35
        attempt = 0

        while len(new_anns) < k_objects and attempt < max_attempts:
            attempt += 1
            chosen_target = rng.choice(templates_meta)
            target_cat = chosen_target["category"]
            sub_cands = [t for t in templates_meta if t.get("category") == target_cat]
            if not sub_cands:
                sub_cands = templates_meta

            typical_size = chosen_target.get("size", [1.9, 4.6, 1.6])
            layers = layers_for_category(target_cat)
            if layers not in areas:
                areas[layers] = get_map_area(nusc, nusc_map, target_token, layers)

            # Rejection sampling with r >= 8m, 1.5m buffer, and mutual separation (dist >= 15m, |delta theta| >= 40 deg)
            pos = sample_valid_position(
                areas[layers],
                existing + new_anns,
                typical_size,
                n_attempts=30,
                rng=rng,
                distance_range=(8.0, 45.0),
                points=points_base,
                inserted_boxes=new_anns
            )
            if pos is None:
                continue

            x, y, yaw_road = pos
            r_new = float(np.hypot(x, y))
            theta_new = float(np.arctan2(y, x))
            alpha_target = float(wrap_angle(theta_new - yaw_road))

            # View-Consistent Matching: distance +- 5m, angle diff <= 25 deg
            dist_cands = [t for t in sub_cands if abs(t.get("distance", r_new) - r_new) <= 5.0]
            if not dist_cands:
                dist_cands = [t for t in sub_cands if abs(t.get("distance", r_new) - r_new) <= 10.0]
            if not dist_cands:
                dist_cands = sub_cands

            def angle_diff(t):
                orig_alpha = t.get("view_angle", 0.0)
                return abs(wrap_angle(orig_alpha - alpha_target))

            sorted_by_view = sorted(dist_cands, key=angle_diff)
            view_aligned = [t for t in sorted_by_view if angle_diff(t) <= 0.436]
            pool = view_aligned if view_aligned else sorted_by_view[:3]
            best_meta = rng.choice(pool)

            tpl = load_template_points(best_meta, templates_dir)
            if not tpl or len(tpl["points"]) < 10:
                continue

            points_base = insert_object(points_base, tpl, pos, ground)
            ground_z = project_z_to_ground(x, y, ground)
            ann = create_annotation(tpl, pos, ground_z)
            new_anns.append(ann)

        sd_token = get_lidar_token(nusc, target_token)
        calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)

        # Weather passes
        weathers_to_apply = []
        if all_weather:
            weathers_to_apply = [("rain", "heavy", True), ("fog", "dense", False)]
        else:
            wet = (weather_type == "rain" and weather_level in ("moderate", "heavy")) or (weather_type == "puddles")
            weathers_to_apply = [(weather_type, weather_level, wet)]

        for w_type, w_lvl, w_wet in weathers_to_apply:
            cur_points = points_base.copy()
            if w_type == "rain":
                cur_points = simulate_rain(cur_points, intensity_level=w_lvl, rng=rng)
                if w_wet:
                    cur_points = simulate_puddles(cur_points, num_puddles=15, max_radius=4.0, rng=rng)
            elif w_type == "fog":
                vis_val = config.FOG_VISIBILITY.get(w_lvl, 20.0)
                cur_points = simulate_fog(cur_points, visibility=vis_val, rng=rng)
            elif w_type == "puddles":
                cur_points = simulate_puddles(cur_points, num_puddles=15, max_radius=4.0, rng=rng)

            # Recount exact surviving lidar points inside boxes
            final_anns = []
            for a in new_anns:
                a_copy = dict(a)
                inside = points_in_cuboid(cur_points, a_copy["translation"], a_copy["size"], a_copy["rotation"]).sum()
                a_copy["num_lidar_pts"] = int(inside)
                final_anns.append(a_copy)

            file_prefix = f"syn_{idx:04d}_{target_token[:8]}_{w_type}"
            bin_path = os.path.join(output_dir, f"{file_prefix}.bin")
            json_path = os.path.join(output_dir, f"{file_prefix}.json")

            save_bin(bin_path, cur_points)

            meta = {
                "synthetic_id": file_prefix,
                "base_sample_token": target_token,
                "weather_type": w_type,
                "weather_level": w_lvl,
                "puddles_enabled": bool(w_wet),
                "num_inserted_objects": len(final_anns),
                "annotations_sensor": final_anns,
                "annotations_global": [box_sensor_to_global(a, calib, pose) for a in final_anns],
                "total_lidar_points": int(len(cur_points))
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=1)

            total_generated += 1

    total_time = time.perf_counter() - t0
    print(f"\n[Success] Generated {total_generated} synthetic scenes!")
    print(f"Total time: {total_time:.2f}s (~ {total_generated / (total_time / 3600):.1f} scenes/hour)")
    print(f"Dataset saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mở rộng dữ liệu nuScenes hàng loạt chuẩn vật lý")
    parser.add_argument("--samples", type=int, default=20, help="Số lượng sample gốc muốn dùng để nhân bản")
    parser.add_argument("--min-objects", type=int, default=2, help="Số vật thể tối thiểu chèn vào")
    parser.add_argument("--max-objects", type=int, default=4, help="Số vật thể tối đa chèn vào")
    parser.add_argument("--weather", choices=["none", "rain", "fog", "puddles"], default="none", help="Loại thời tiết")
    parser.add_argument("--weather-level", choices=["light", "moderate", "heavy", "dense"], default="moderate")
    parser.add_argument("--all-weather", action="store_true", help="Sinh đồng thời cả Monsoon Rain + Puddles và Fog Dense")
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
        seed=args.seed,
        all_weather=args.all_weather
    )
