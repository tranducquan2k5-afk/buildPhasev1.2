"""
Mode 1: nuScenes Sandbox Engine
================================
Preserves 100% of physical and geometric algorithms:
- Task A: Map-aware placement on drivable_area/walkway, Distance Matching ±5m,
          Occlusion Range Image Ray Casting ±15cm, points_in_cuboid cleaning.
- Task B: Beer-Lambert rain, Fog backscatter, Specular reflection wet road.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure project root and n3-08c are in sys.path
try:
    from server_config import (BANK_JSON_PATH, DATASETS_DIR, MAX_PREVIEW_POINTS,
                               N3_08C_DIR, NUSCENES_DATAROOT, NUSCENES_VERSION,
                               TEMPLATES_DIR, get_cached_nuscenes, get_cached_template_bank)
except (ImportError, ValueError):
    from webapp.backend.server_config import (BANK_JSON_PATH, DATASETS_DIR, MAX_PREVIEW_POINTS,
                                              N3_08C_DIR, NUSCENES_DATAROOT, NUSCENES_VERSION,
                                              TEMPLATES_DIR, get_cached_nuscenes, get_cached_template_bank)

if str(N3_08C_DIR) not in sys.path:
    sys.path.insert(0, str(N3_08C_DIR))

import config as nusc_cfg
from task_a.ground_plane import (GroundModel, get_map_area,
                                 layers_for_category, load_map_for_sample,
                                 project_z_to_ground)
from task_a.merge import create_annotation, insert_object
from task_a.placement import sample_valid_position
from task_b.fog import simulate_fog
from task_b.rain import simulate_rain
from task_b.wet_road import simulate_puddles
from utils.nuscenes_loader import (get_annotations, get_pointcloud,
                                   load_nuscenes)

logger = logging.getLogger("engine_nuscenes")


class NuscenesSandboxEngine:
    def __init__(self):
        # Lazy Singleton RAM Cache
        self.templates_meta, self.templates_by_cat = get_cached_template_bank()

    def get_sample_tokens_list(self) -> List[str]:
        nusc = get_cached_nuscenes()
        return [s["token"] for s in nusc.sample]

    def load_template_points(self, tpl_meta: dict) -> Optional[dict]:
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

    def run(self,
            sample_token: Optional[str] = None,
            enable_insertion: bool = True,
            categories: Optional[List[str]] = None,
            num_objects: int = 3,
            weather_type: str = "none",
            weather_level: str = "moderate",
            enable_wet_road: bool = False,
            seed: int = 42) -> dict:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        nusc = get_cached_nuscenes()

        tokens = [s["token"] for s in nusc.sample]
        if not tokens:
            raise RuntimeError("No samples found in nuScenes dataset.")

        if not sample_token or sample_token not in tokens:
            sample_token = tokens[0]

        # 1. Load original point cloud and annotations
        orig_points = get_pointcloud(nusc, sample_token)
        existing_boxes = get_annotations(nusc, sample_token)
        points = orig_points.copy()
        inserted_boxes = []

        # 2. Vector Map Aware Object Insertion (Task A) with View-Consistent Matching
        if enable_insertion and self.templates_meta and num_objects > 0:
            from utils.transforms import wrap_angle
            ground = GroundModel.from_points(points, rng=rng)
            nusc_map = load_map_for_sample(nusc, sample_token)
            areas = {}

            # Filter candidate templates
            if categories and len(categories) > 0 and "all" not in categories:
                cat_candidates = [
                    t for t in self.templates_meta
                    if any(c.lower() in t.get("category", "").lower() for c in categories)
                ]
            else:
                cat_candidates = self.templates_meta

            if cat_candidates:
                max_total_attempts = num_objects * 35
                attempt = 0
                while len(inserted_boxes) < num_objects and attempt < max_total_attempts:
                    attempt += 1

                    # Pick a target category
                    chosen_target = rng.choice(cat_candidates)
                    target_cat = chosen_target["category"]
                    sub_cands = [t for t in cat_candidates if t.get("category") == target_cat]
                    if not sub_cands:
                        sub_cands = cat_candidates

                    typical_size = chosen_target.get("size", [1.9, 4.6, 1.6])
                    layers = layers_for_category(target_cat)
                    if layers not in areas:
                        areas[layers] = get_map_area(nusc, nusc_map, sample_token, layers)

                    # 1. Sample road-aligned position on vector map (r >= 8.0m, mutual dist >= 15m, azimuth >= 40 deg)
                    pos = sample_valid_position(
                        areas[layers],
                        existing_boxes + inserted_boxes,
                        typical_size,
                        n_attempts=30,
                        rng=rng,
                        distance_range=(8.0, 45.0),
                        points=points,
                        inserted_boxes=inserted_boxes
                    )
                    if pos is None:
                        continue

                    x, y, yaw_road = pos
                    r_new = float(np.hypot(x, y))
                    theta_new = float(np.arctan2(y, x))
                    alpha_target = float(wrap_angle(theta_new - yaw_road))

                    # 2. View-Consistent Template Matching
                    # Distance band: r_new +- 5m (or fallback +- 10m)
                    dist_cands = [t for t in sub_cands if abs(t.get("distance", r_new) - r_new) <= 5.0]
                    if not dist_cands:
                        dist_cands = [t for t in sub_cands if abs(t.get("distance", r_new) - r_new) <= 10.0]
                    if not dist_cands:
                        dist_cands = sub_cands

                    # Angular alignment difference: |alpha_orig - alpha_target|
                    def angle_diff(t):
                        orig_alpha = t.get("view_angle", 0.0)
                        return abs(wrap_angle(orig_alpha - alpha_target))

                    sorted_by_view = sorted(dist_cands, key=angle_diff)
                    # Filter within 25 degrees (0.436 rad) if possible
                    view_aligned = [t for t in sorted_by_view if angle_diff(t) <= 0.436]
                    pool = view_aligned if view_aligned else sorted_by_view[:3]
                    best_meta = rng.choice(pool)

                    tpl = self.load_template_points(best_meta)
                    if not tpl or len(tpl["points"]) < 10:
                        continue

                    # Insert with Range Image Ray Casting +-15cm & points_in_cuboid cleaning
                    points = insert_object(points, tpl, pos, ground)
                    ground_z = project_z_to_ground(x, y, ground)
                    ann = create_annotation(tpl, pos, ground_z)
                    inserted_boxes.append(ann)

        # 3. Task B Weather Simulation
        weather_applied = weather_type.lower()
        if weather_applied == "rain":
            level = weather_level.lower() if weather_level.lower() in ("light", "moderate", "heavy") else "moderate"
            points = simulate_rain(points, intensity_level=level, rng=rng)
        elif weather_applied in ("fog", "sương mù"):
            vis_map = {"light": 200.0, "moderate": 50.0, "heavy": 20.0, "dense": 20.0}
            vis = vis_map.get(weather_level.lower(), 50.0)
            points = simulate_fog(points, visibility=vis, rng=rng)

        # 4. Vietnam Wet Road Specular Reflection
        if enable_wet_road:
            points = simulate_puddles(points, num_puddles=15, max_radius=4.0, rng=rng)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # Giữ nguyên 100% độ phân giải nếu <= 60.000 điểm để bảo toàn độ sắc nét vùng Occlusion ±15cm
        def downsample_flat(pts, max_pts=MAX_PREVIEW_POINTS):
            if len(pts) <= 60000:
                sub = pts
            else:
                idx = rng.choice(len(pts), max_pts, replace=False)
                sub = pts[idx]
            return sub[:, :4].flatten().tolist()

        orig_flat = downsample_flat(orig_points)
        aug_flat = downsample_flat(points)

        return {
            "augmented_points": points,
            "annotations": inserted_boxes,
            "sample_token": sample_token,
            "preview_data": {
                "original_points": orig_flat,
                "augmented_points": aug_flat,
                "num_original": len(orig_points),
                "num_augmented": len(points),
                "boxes": inserted_boxes
            },
            "stats": {
                "mode": "Mode 1: nuScenes Sandbox",
                "initial_point_count": int(len(orig_points)),
                "final_point_count": int(len(points)),
                "point_count_diff": int(len(points) - len(orig_points)),
                "inserted_objects_count": len(inserted_boxes),
                "weather_type": weather_applied,
                "weather_level": weather_level,
                "wet_road_enabled": bool(enable_wet_road),
                "elapsed_ms": round(elapsed_ms, 2)
            }
        }
