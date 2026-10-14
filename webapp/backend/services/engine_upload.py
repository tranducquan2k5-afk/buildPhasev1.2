"""
Mode 2: Custom Upload Engine with Mapless Fallback
==================================================
Handles arbitrary point clouds (.bin, .pcd, .ply, .las, .laz, .xyz, .csv, .npy).
When vector map is absent, automatically activates Mapless Fallback:
- Detects ground plane via RANSAC GroundModel.
- Performs collision-free 2D BEV placement (Shapely IoU = 0.0).
- Applies Distance Matching ±5m and Occlusion Range Image Ray Casting ±15cm.
- Executes physical weather models (Beer-Lambert rain, fog backscatter, wet road).
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from shapely.geometry import Polygon

try:
    from server_config import (BANK_JSON_PATH, MAX_PREVIEW_POINTS, N3_08C_DIR,
                               TEMPLATES_DIR, get_cached_template_bank)
except (ImportError, ValueError):
    from webapp.backend.server_config import (BANK_JSON_PATH, MAX_PREVIEW_POINTS, N3_08C_DIR,
                                              TEMPLATES_DIR, get_cached_template_bank)

if str(N3_08C_DIR) not in sys.path:
    sys.path.insert(0, str(N3_08C_DIR))

from task_a.ground_plane import GroundModel, project_z_to_ground
from task_a.merge import create_annotation, insert_object
from task_a.placement import bev_footprint, check_collision_bev
from task_b.fog import simulate_fog
from task_b.rain import simulate_rain
from task_b.wet_road import simulate_puddles
from utils.io_converter import load_point_cloud, save_point_cloud

logger = logging.getLogger("engine_upload")


class CustomUploadEngine:
    def __init__(self):
        # Lazy Singleton RAM Cache
        self.templates_meta, self.templates_by_cat = get_cached_template_bank()

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

    def sample_mapless_position(self, obj_size: Sequence[float], existing_boxes: List[dict],
                                target_dist: float = 15.0, rng: Optional[np.random.Generator] = None,
                                n_attempts: int = 60,
                                inserted_boxes: Optional[Sequence[dict]] = None) -> Optional[Tuple[float, float, float]]:
        """
        Mapless Fallback: Samples position in front roadway cone with distance matching ±5m
        and Shapely 2D BEV collision check (IoU = 0.0).
        """
        from utils.transforms import wrap_angle
        from task_a.placement import is_mutually_separated
        rng = rng or np.random.default_rng()
        w, l, h = obj_size

        r_min = max(8.0, target_dist - 5.0)
        r_max = min(45.0, target_dist + 5.0)

        for _ in range(n_attempts):
            r = rng.uniform(r_min, r_max)
            theta = rng.uniform(-np.pi * 0.45, np.pi * 0.45) # Frontal roadway sector
            x = float(r * np.cos(theta))
            y = float(r * np.sin(theta))
            
            # Kiểm tra cách ly tối thiểu dist >= 15m và |delta theta| >= 40 deg
            if inserted_boxes and not is_mutually_separated(x, y, inserted_boxes, min_dist=15.0, min_azimuth_deg=40.0):
                continue

            # Hướng xe dọc theo luồng giao thông (tiến theo trục X hoặc đối diện)
            base_yaw = 0.0 if rng.random() > 0.35 else np.pi
            yaw = float(wrap_angle(base_yaw + rng.uniform(-np.radians(6), np.radians(6))))

            cand_box = {
                "translation": [x, y, 0.0],
                "size": [w, l, h],
                "rotation": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            }

            if not check_collision_bev(cand_box, existing_boxes, buffer_margin=1.5):
                return (x, y, yaw)

        return None

    def run(self,
            input_file_or_bytes: Union[str, Path, bytes],
            input_ext: str = ".bin",
            enable_insertion: bool = True,
            categories: Optional[List[str]] = None,
            num_objects: int = 3,
            weather_type: str = "none",
            weather_level: str = "moderate",
            enable_wet_road: bool = False,
            seed: int = 42) -> dict:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # 1. Parse uploaded point cloud via universal IO converter
        orig_points = load_point_cloud(input_file_or_bytes, file_ext=input_ext)
        points = orig_points.copy()
        inserted_boxes = []

        # 2. Mapless Fallback Object Insertion
        if enable_insertion and self.templates_meta and num_objects > 0:
            from utils.transforms import wrap_angle
            ground = GroundModel.from_points(points, rng=rng)

            # Filter candidates
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

                    chosen_target = rng.choice(cat_candidates)
                    target_cat = chosen_target["category"]
                    sub_cands = [t for t in cat_candidates if t.get("category") == target_cat]
                    if not sub_cands:
                        sub_cands = cat_candidates

                    typical_size = chosen_target.get("size", [1.9, 4.6, 1.6])
                    target_dist = chosen_target.get("distance", 18.0)

                    pos = self.sample_mapless_position(typical_size, inserted_boxes,
                                                       target_dist=target_dist, rng=rng,
                                                       inserted_boxes=inserted_boxes)
                    if pos is None:
                        continue

                    x, y, yaw_road = pos
                    r_new = float(np.hypot(x, y))
                    theta_new = float(np.arctan2(y, x))
                    alpha_target = float(wrap_angle(theta_new - yaw_road))

                    # View-Consistent Matching: distance +- 5m & angle diff
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

                    tpl = self.load_template_points(best_meta)
                    if not tpl or len(tpl["points"]) < 10:
                        continue

                    # Insert with Range Image Ray Casting ±15cm & points_in_cuboid cleaning
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
            "preview_data": {
                "original_points": orig_flat,
                "augmented_points": aug_flat,
                "num_original": len(orig_points),
                "num_augmented": len(points),
                "boxes": inserted_boxes
            },
            "stats": {
                "mode": "Mode 2: Custom Upload (Mapless Fallback)",
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
