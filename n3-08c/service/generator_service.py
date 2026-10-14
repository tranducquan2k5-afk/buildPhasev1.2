"""
Synthetic LiDAR Data Generator Engine
=====================================
Unified service powering Web App, SaaS REST API, and batch CLI processing.
Supports arbitrary point clouds from CVAT, LabelCloud, CloudCompare, ROS, KITTI, and nuScenes.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Ensure n3-08c root is in sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from task_a.ground_plane import GroundModel, project_z_to_ground
from task_a.merge import create_annotation, insert_object
from task_a.placement import bev_footprint, check_collision_bev
from task_b.fog import simulate_fog
from task_b.rain import simulate_rain
from task_b.wet_road import simulate_puddles
from utils.io_converter import load_point_cloud, save_point_cloud

logger = logging.getLogger("generator_service")
logging.basicConfig(level=logging.INFO)

WEATHER_PRESETS = {
    "rain": {
        "light": 10.0,      # mm/h
        "moderate": 30.0,   # mm/h
        "heavy": 60.0       # mm/h (Bão nhiệt đới)
    },
    "fog": {
        "light": 50.0,      # visibility meters
        "moderate": 25.0,   # visibility meters
        "heavy": 10.0,      # visibility meters (Sương mù dày đặc)
        "dense": 10.0
    }
}


class SyntheticDataGenerator:
    """Production-grade Synthetic Point Cloud Augmentation Pipeline."""

    def __init__(self, templates_dir: Optional[Union[str, Path]] = None):
        if templates_dir is None:
            # Default to output/templates in workspace
            templates_dir = BASE_DIR.parent / "output" / "templates"
        self.templates_dir = Path(templates_dir)
        self.bank_file = self.templates_dir / "bank.json"
        self.templates_meta = []
        self.templates_by_cat: Dict[str, List[dict]] = {}

        self._load_bank()

    def _load_bank(self):
        """Loads and indexes the 4,111 templates in the Template Bank."""
        if not self.bank_file.exists():
            logger.warning(f"Template bank not found at {self.bank_file}. Object insertion will be disabled.")
            return

        with open(self.bank_file, "r", encoding="utf-8") as f:
            self.templates_meta = json.load(f)

        for item in self.templates_meta:
            cat = item.get("category", "unknown")
            if cat not in self.templates_by_cat:
                self.templates_by_cat[cat] = []
            self.templates_by_cat[cat].append(item)

        logger.info(f"Loaded Template Bank: {len(self.templates_meta)} templates across {len(self.templates_by_cat)} categories.")

    def get_available_categories(self) -> List[str]:
        """Returns list of categories present in the template bank."""
        return sorted(list(self.templates_by_cat.keys()))

    def load_template_points(self, tpl_meta: dict) -> Optional[dict]:
        """Loads the full .npz points for a template meta item."""
        rel_path = tpl_meta.get("file")
        if not rel_path:
            return None
        npz_path = self.templates_dir / rel_path
        if not npz_path.exists():
            # Try normalizing backslashes
            npz_path = self.templates_dir / Path(rel_path.replace("\\", "/"))
        if not npz_path.exists():
            return None

        try:
            data = np.load(str(npz_path))
            points = data["points"]
            return {
                "points": points,
                "size": tpl_meta["size"],
                "category": tpl_meta["category"],
                "distance": tpl_meta.get("distance", 15.0),
                "ann_token": tpl_meta.get("ann_token", "")
            }
        except Exception as e:
            logger.error(f"Failed to load template {npz_path}: {e}")
            return None

    def sample_position_for_cloud(self, points: np.ndarray, ground_model: GroundModel,
                                  obj_size: Sequence[float], existing_boxes: List[dict],
                                  target_distance: float = 15.0,
                                  rng: Optional[np.random.Generator] = None,
                                  n_attempts: int = 50) -> Optional[Tuple[float, float, float]]:
        """
        Samples a valid collision-free position (x, y, yaw) in the front/surrounding field
        with distance matching (target_distance ± 5m).
        """
        rng = rng or np.random.default_rng()
        w, l, h = obj_size

        # Radial bounds based on target distance ± 5m
        r_min = max(3.0, target_distance - 5.0)
        r_max = min(50.0, target_distance + 5.0)

        for _ in range(n_attempts):
            # Sample distance and azimuth (mostly in front sector [-120 deg, 120 deg])
            r = rng.uniform(r_min, r_max)
            theta = rng.uniform(-np.pi * 0.65, np.pi * 0.65) # Frontal semicircle
            x = float(r * np.sin(theta))
            y = float(r * np.cos(theta))

            # Yaw: oriented roughly with road or random
            yaw = float(rng.uniform(-np.pi, np.pi))

            # Check collision with existing boxes
            cand_box = {
                "translation": [x, y, 0.0],
                "size": [w, l, h],
                "rotation": [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]
            }

            if not check_collision_bev(cand_box, existing_boxes):
                return (x, y, yaw)

        return None

    def process(self,
                input_file_or_pts: Union[str, Path, np.ndarray, bytes],
                input_ext: Optional[str] = None,
                enable_insertion: bool = True,
                categories: Optional[List[str]] = None,
                num_objects: int = 3,
                weather_type: str = "none",
                weather_level: str = "moderate",
                enable_wet_road: bool = False,
                seed: Optional[int] = 42) -> dict:
        """
        Executes end-to-end synthetic data generation pipeline.
        Returns dictionary containing:
        - 'augmented_points': (M, 5) float32 numpy array
        - 'annotations': list of 3D box dicts
        - 'preview_data': downsampled point cloud dictionary for Three.js 3D WebGL
        - 'stats': execution metrics & counts
        """
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)

        # 1. Load input point cloud
        if isinstance(input_file_or_pts, np.ndarray):
            points = input_file_or_pts.copy().astype(np.float32)
        else:
            points = load_point_cloud(input_file_or_pts, file_ext=input_ext)

        initial_count = len(points)
        inserted_boxes = []

        # 2. Fit Ground Plane using RANSAC GroundModel
        ground = GroundModel.from_points(points, rng=rng)

        # 3. Object Insertion (Task A with Occlusion & Distance Matching)
        if enable_insertion and self.templates_meta and num_objects > 0:
            # Filter candidate templates
            if categories and len(categories) > 0 and "all" not in categories:
                cand_meta = [m for m in self.templates_meta if m.get("category") in categories]
            else:
                cand_meta = self.templates_meta

            if cand_meta:
                chosen_indices = rng.choice(len(cand_meta), size=min(num_objects * 2, len(cand_meta)), replace=False)
                inserted_count = 0

                for idx in chosen_indices:
                    if inserted_count >= num_objects:
                        break
                    tpl_meta = cand_meta[idx]
                    tpl = self.load_template_points(tpl_meta)
                    if tpl is None or len(tpl["points"]) < 5:
                        continue

                    target_dist = tpl.get("distance", 15.0)
                    pos = self.sample_position_for_cloud(points, ground, tpl["size"], inserted_boxes,
                                                        target_distance=target_dist, rng=rng)
                    if pos is None:
                        continue

                    # Insert with Range Image Ray Casting occlusion
                    points = insert_object(points, tpl, pos, ground)
                    x, y, yaw = pos
                    ground_z = project_z_to_ground(x, y, ground)
                    ann = create_annotation(tpl, pos, ground_z)
                    inserted_boxes.append(ann)
                    inserted_count += 1

        # 4. Weather Simulation (Task B with Physical Models)
        weather_applied = weather_type.lower()
        if weather_applied == "rain":
            level = weather_level.lower()
            if level not in ("light", "moderate", "heavy"):
                level = "moderate"
            points = simulate_rain(points, intensity_level=level, rng=rng)
        elif weather_applied in ("fog", "sương mù"):
            vis_map = {"light": 200.0, "moderate": 50.0, "heavy": 20.0, "dense": 20.0}
            vis = vis_map.get(weather_level.lower(), 50.0)
            points = simulate_fog(points, visibility=vis, rng=rng)

        # 5. Wet Road Puddle Simulation (Specular Reflection - Vietnam Tropical Feature)
        if enable_wet_road:
            points = simulate_puddles(points, num_puddles=15, max_radius=4.0, rng=rng)

        final_count = len(points)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        # 6. Generate 3D WebGL Preview Data (Downsampled for 60 FPS Three.js rendering)
        max_preview_points = 12000
        if len(points) > max_preview_points:
            sample_idx = rng.choice(len(points), max_preview_points, replace=False)
            preview_pts = points[sample_idx]
        else:
            preview_pts = points

        # Convert preview to flat list [x0, y0, z0, i0, x1, y1, z1, i1, ...]
        preview_flat = preview_pts[:, :4].flatten().tolist()

        stats = {
            "initial_point_count": int(initial_count),
            "final_point_count": int(final_count),
            "point_count_diff": int(final_count - initial_count),
            "inserted_objects_count": len(inserted_boxes),
            "weather_type": weather_applied,
            "weather_level": weather_level,
            "wet_road_enabled": bool(enable_wet_road),
            "elapsed_ms": round(elapsed_ms, 2)
        }

        return {
            "augmented_points": points,
            "annotations": inserted_boxes,
            "preview_data": {
                "points": preview_flat,
                "num_points": len(preview_pts),
                "boxes": inserted_boxes
            },
            "stats": stats
        }
