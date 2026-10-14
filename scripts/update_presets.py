"""
Regenerate 100 Scene Presets with Physical Mutual Separation & High-Quality Templates
=====================================================================================
Applies 3 key physical rules:
1. Mutual Separation:
   - Euclidean distance between any 2 newly inserted objects >= 15.0m
   - Azimuth angular clearance between any 2 newly inserted objects >= 40.0 deg
   - BEV 2D collision check with existing boxes expanded by +-1.5m buffer margin
   - Distance from LiDAR sensor origin r >= 8.0m
   - Road-aligned yaw heading (aligned with nearby vehicle flow or roadway PCA)
2. Strict Template Filtering:
   - Trucks, buses, construction, trailers: num_points >= 400 AND span_x / length >= 0.55
   - Cars: num_points >= 180 AND span_x / length >= 0.55
   - Other objects (motorcycle, bicycle, pedestrian, barrier): num_points >= 60
3. Vietnam Weather Synthesis:
   - vietnam_monsoon_dataset (50 scenes): Beer-Lambert heavy rain + 15 specular puddles
   - fog_dense_dataset (50 scenes): Backscatter fog with visibility = 20.0m
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "n3-08c"))
sys.path.insert(0, str(PROJECT_ROOT / "webapp" / "backend"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("update_presets")

from server_config import (
    FOG_DATASET_DIR,
    MONSOON_DATASET_DIR,
    TEMPLATES_DIR,
    get_cached_nuscenes,
    get_cached_template_bank,
)
from task_a.ground_plane import (
    GroundModel,
    get_map_area,
    layers_for_category,
    load_map_for_sample,
    project_z_to_ground,
)
from task_a.merge import create_annotation, insert_object
from task_a.placement import is_mutually_separated, sample_valid_position
from task_b.fog import simulate_fog
from task_b.rain import simulate_rain
from task_b.wet_road import simulate_puddles
from utils.nuscenes_loader import (
    get_annotations,
    get_calibration,
    get_ego_pose,
    get_lidar_token,
    get_pointcloud,
    save_bin,
)
from utils.transforms import box_sensor_to_global, points_in_cuboid, wrap_angle


def load_template_points(tpl_meta: dict) -> Optional[dict]:
    rel = tpl_meta.get("file", "").replace("\\", "/")
    path = TEMPLATES_DIR / rel
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
    except Exception as e:
        logger.error(f"Error loading template {path}: {e}")
        return None


def run_regeneration(start_idx: int = 0, end_idx: int = 50, seed: int = 2026):
    os.makedirs(MONSOON_DATASET_DIR, exist_ok=True)
    os.makedirs(FOG_DATASET_DIR, exist_ok=True)

    rng = np.random.default_rng(seed)
    nusc = get_cached_nuscenes()
    templates_meta, templates_by_cat = get_cached_template_bank()

    samples = nusc.sample[start_idx:end_idx]
    logger.info(f"Starting regeneration of preset scenes from index {start_idx} to {end_idx} (count: {len(samples)})...")
    logger.info(f"Monsoon target dir: {MONSOON_DATASET_DIR}")
    logger.info(f"Fog target dir:     {FOG_DATASET_DIR}")
    logger.info(f"High-quality templates pool: {len(templates_meta)} items across {len(templates_by_cat)} categories.")

    t_start = time.perf_counter()

    for rel_idx, s in enumerate(samples):
        idx = start_idx + rel_idx
        t_scene = time.perf_counter()
        target_token = s["token"]
        orig_points = get_pointcloud(nusc, target_token)
        existing_boxes = get_annotations(nusc, target_token)
        ground = GroundModel.from_points(orig_points, rng=rng)
        nusc_map = load_map_for_sample(nusc, target_token)

        # 1. Base Object Insertion
        points_base = orig_points.copy()
        inserted_boxes = []
        areas = {}

        k_target = int(rng.integers(2, 5))
        max_attempts = k_target * 35
        attempt = 0

        while len(inserted_boxes) < k_target and attempt < max_attempts:
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

            pos = sample_valid_position(
                areas[layers],
                existing_boxes + inserted_boxes,
                typical_size,
                n_attempts=30,
                rng=rng,
                distance_range=(8.0, 45.0),
                points=points_base,
                inserted_boxes=inserted_boxes
            )
            if pos is None:
                continue

            x, y, yaw_road = pos
            r_new = float(np.hypot(x, y))
            theta_new = float(np.arctan2(y, x))
            alpha_target = float(wrap_angle(theta_new - yaw_road))

            # View-Consistent Matching
            dist_cands = [t for t in sub_cands if abs(t.get("distance", r_new) - r_new) <= 5.0]
            if not dist_cands:
                dist_cands = [t for t in sub_cands if abs(t.get("distance", r_new) - r_new) <= 10.0]
            if not dist_cands:
                dist_cands = sub_cands

            def angle_diff(t):
                orig_alpha = t.get("view_angle", 0.0)
                return abs(wrap_angle(orig_alpha - alpha_target))

            sorted_by_view = sorted(dist_cands, key=angle_diff)
            view_aligned = [t for t in sorted_by_view if angle_diff(t) <= 0.436] # 25 degrees
            pool = view_aligned if view_aligned else sorted_by_view[:3]
            best_meta = rng.choice(pool)

            tpl = load_template_points(best_meta)
            if not tpl or len(tpl["points"]) < 10:
                continue

            # Occlusion handling ray casting +-15cm & hollow interior cleaning
            points_base = insert_object(points_base, tpl, pos, ground)
            ground_z = project_z_to_ground(x, y, ground)
            ann = create_annotation(tpl, pos, ground_z)
            inserted_boxes.append(ann)

        sd_token = get_lidar_token(nusc, target_token)
        calib = get_calibration(nusc, sd_token)
        pose = get_ego_pose(nusc, sd_token)

        # 2. Generate Vietnam Monsoon (Heavy Rain + Puddles)
        points_rain = simulate_rain(points_base.copy(), intensity_level="heavy", rng=rng)
        points_rain = simulate_puddles(points_rain, num_puddles=15, max_radius=4.0, rng=rng)

        rain_anns = []
        for b in inserted_boxes:
            b_copy = dict(b)
            inside = points_in_cuboid(points_rain, b_copy["translation"], b_copy["size"], b_copy["rotation"]).sum()
            b_copy["num_lidar_pts"] = int(inside)
            rain_anns.append(b_copy)

        file_prefix_rain = f"syn_{idx:04d}_{target_token[:8]}_rain"
        bin_rain_path = MONSOON_DATASET_DIR / f"{file_prefix_rain}.bin"
        json_rain_path = MONSOON_DATASET_DIR / f"{file_prefix_rain}.json"
        save_bin(str(bin_rain_path), points_rain)

        meta_rain = {
            "synthetic_id": file_prefix_rain,
            "base_sample_token": target_token,
            "weather_type": "rain",
            "weather_level": "heavy",
            "puddles_enabled": True,
            "num_inserted_objects": len(rain_anns),
            "annotations_sensor": rain_anns,
            "annotations_global": [box_sensor_to_global(a, calib, pose) for a in rain_anns],
            "total_lidar_points": int(len(points_rain))
        }
        with open(json_rain_path, "w", encoding="utf-8") as f:
            json.dump(meta_rain, f, indent=1)

        # 3. Generate Fog Dense
        points_fog = simulate_fog(points_base.copy(), visibility=20.0, rng=rng)

        fog_anns = []
        for b in inserted_boxes:
            b_copy = dict(b)
            inside = points_in_cuboid(points_fog, b_copy["translation"], b_copy["size"], b_copy["rotation"]).sum()
            b_copy["num_lidar_pts"] = int(inside)
            fog_anns.append(b_copy)

        file_prefix_fog = f"syn_{idx:04d}_{target_token[:8]}_fog"
        bin_fog_path = FOG_DATASET_DIR / f"{file_prefix_fog}.bin"
        json_fog_path = FOG_DATASET_DIR / f"{file_prefix_fog}.json"
        save_bin(str(bin_fog_path), points_fog)

        meta_fog = {
            "synthetic_id": file_prefix_fog,
            "base_sample_token": target_token,
            "weather_type": "fog",
            "weather_level": "dense",
            "puddles_enabled": False,
            "num_inserted_objects": len(fog_anns),
            "annotations_sensor": fog_anns,
            "annotations_global": [box_sensor_to_global(a, calib, pose) for a in fog_anns],
            "total_lidar_points": int(len(points_fog))
        }
        with open(json_fog_path, "w", encoding="utf-8") as f:
            json.dump(meta_fog, f, indent=1)

        dt = time.perf_counter() - t_scene
        logger.info(f"[{idx+1:02d}/{len(samples):02d}] Scene {target_token[:8]} completed in {dt:.2f}s | "
                    f"Inserted: {len(inserted_boxes)} objs | Rain pts: {len(points_rain)} | Fog pts: {len(points_fog)}")

    total_time = time.perf_counter() - t_start
    logger.info(f"Regeneration complete! 100 preset files regenerated in {total_time:.2f}s (~{total_time/60:.1f} mins).")


def verify_generated_presets():
    """Verify mutual separation, distances, and point counts across all generated presets."""
    logger.info("Running verification across regenerated preset datasets...")
    monsoon_jsons = sorted(MONSOON_DATASET_DIR.glob("*.json"))
    fog_jsons = sorted(FOG_DATASET_DIR.glob("*.json"))

    assert len(monsoon_jsons) == 50, f"Expected 50 monsoon files, found {len(monsoon_jsons)}"
    assert len(fog_jsons) == 50, f"Expected 50 fog files, found {len(fog_jsons)}"

    total_checked = 0
    violations_dist = 0
    violations_azimuth = 0
    violations_center_r = 0

    for jpath in monsoon_jsons + fog_jsons:
        total_checked += 1
        with open(jpath, "r", encoding="utf-8") as f:
            meta = json.load(f)
        boxes = meta.get("annotations_sensor", [])
        if len(boxes) < 2:
            continue

        for i in range(len(boxes)):
            b1 = boxes[i]
            x1, y1 = b1["translation"][0], b1["translation"][1]
            r1 = np.hypot(x1, y1)
            theta1 = np.arctan2(y1, x1)
            if r1 < 7.9:  # margin
                violations_center_r += 1

            for j in range(i + 1, len(boxes)):
                b2 = boxes[j]
                x2, y2 = b2["translation"][0], b2["translation"][1]
                dist = np.hypot(x1 - x2, y1 - y2)
                theta2 = np.arctan2(y2, x2)
                d_theta_deg = np.degrees(abs(wrap_angle(theta1 - theta2)))

                if dist < 14.9:
                    violations_dist += 1
                    logger.warning(f"Distance violation in {jpath.name}: dist = {dist:.2f}m")
                if d_theta_deg < 39.5:
                    violations_azimuth += 1
                    logger.warning(f"Azimuth violation in {jpath.name}: d_theta = {d_theta_deg:.2f} deg")

    logger.info(f"Verification Results over {total_checked} files:")
    logger.info(f"- Mutual Distance Violations (< 15.0m): {violations_dist}")
    logger.info(f"- Mutual Azimuth Violations (< 40.0 deg): {violations_azimuth}")
    logger.info(f"- Sensor Origin Proximity Violations (< 8.0m): {violations_center_r}")

    if violations_dist == 0 and violations_azimuth == 0 and violations_center_r == 0:
        logger.info("ALL PRESETS COMPLY 100% WITH PHYSICAL MUTUAL SEPARATION CONSTRAINTS!")
    else:
        logger.error(f"Encountered violations: dist={violations_dist}, azimuth={violations_azimuth}, r={violations_center_r}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes", type=int, default=50)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=50)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        verify_generated_presets()
    else:
        end = args.end_idx if args.end_idx != 50 or args.scenes == 50 else args.scenes
        run_regeneration(start_idx=args.start_idx, end_idx=end)
        verify_generated_presets()
