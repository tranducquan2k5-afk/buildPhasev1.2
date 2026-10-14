import json
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

try:
    from server_config import (FOG_DATASET_DIR, MAX_PREVIEW_POINTS,
                               MONSOON_DATASET_DIR, SLIDES_HTML_PATH, VIS_DIR)
except (ImportError, ValueError):
    from webapp.backend.server_config import (FOG_DATASET_DIR, MAX_PREVIEW_POINTS,
                                              MONSOON_DATASET_DIR, SLIDES_HTML_PATH, VIS_DIR)

from utils.io_converter import load_point_cloud

router = APIRouter(tags=["Presets & Visuals"])


@router.get("/api/presets/list")
def list_available_presets():
    """Lists pre-generated scenes from vietnam_monsoon_dataset and fog_dense_dataset."""
    presets = []

    # 1. Monsoon scenes
    if MONSOON_DATASET_DIR.exists():
        for json_file in sorted(MONSOON_DATASET_DIR.glob("*.json"))[:15]:
            bin_file = json_file.with_suffix(".bin")
            if bin_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    presets.append({
                        "dataset": "vietnam_monsoon_dataset",
                        "dataset_name": "Mưa Bão Nhiệt Đới Việt Nam",
                        "filename": bin_file.name,
                        "synthetic_id": meta.get("synthetic_id", bin_file.stem),
                        "weather_type": meta.get("weather_type", "rain"),
                        "weather_level": meta.get("weather_level", "heavy"),
                        "num_objects": meta.get("num_inserted_objects", len(meta.get("annotations_sensor", []))),
                        "total_points": meta.get("total_lidar_points", 0)
                    })
                except Exception:
                    pass

    # 2. Fog scenes
    if FOG_DATASET_DIR.exists():
        for json_file in sorted(FOG_DATASET_DIR.glob("*.json"))[:15]:
            bin_file = json_file.with_suffix(".bin")
            if bin_file.exists():
                try:
                    with open(json_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    presets.append({
                        "dataset": "fog_dense_dataset",
                        "dataset_name": "Sương Mù Dày Đặc (Fog Dense)",
                        "filename": bin_file.name,
                        "synthetic_id": meta.get("synthetic_id", bin_file.stem),
                        "weather_type": meta.get("weather_type", "fog"),
                        "weather_level": meta.get("weather_level", "dense"),
                        "num_objects": meta.get("num_inserted_objects", len(meta.get("annotations_sensor", []))),
                        "total_points": meta.get("total_lidar_points", 0)
                    })
                except Exception:
                    pass

    return {
        "status": "success",
        "total_presets": len(presets),
        "presets": presets
    }


@router.get("/api/presets/load")
def load_preset_scene(dataset: str = Query(...), filename: str = Query(...)):
    """Loads a pre-generated scene into Three.js WebGL format, preserving 100% points <= 60k."""
    base_dir = MONSOON_DATASET_DIR if "monsoon" in dataset else FOG_DATASET_DIR
    bin_path = base_dir / filename
    json_path = bin_path.with_suffix(".json")

    if not bin_path.exists():
        raise HTTPException(status_code=404, detail=f"File {bin_path.name} not found.")

    pts = load_point_cloud(str(bin_path))

    boxes = []
    meta = {}
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            raw_boxes = meta.get("annotations_sensor", [])
            for b in raw_boxes:
                sz = b.get("size", [1.8, 4.5, 1.5])
                w, l, h = float(sz[0]), float(sz[1]), float(sz[2])
                trans = [float(v) for v in b.get("translation", [0.0, 0.0, 0.0])]
                rot = [float(v) for v in b.get("rotation", [1.0, 0.0, 0.0, 0.0])]
                boxes.append({
                    "token": b.get("token", ""),
                    "category": b.get("category", "vehicle.car"),
                    "translation": trans,
                    "size": [w, l, h],
                    "width": w,
                    "length": l,
                    "height": h,
                    "rotation": rot,
                    "num_lidar_pts": b.get("num_lidar_pts", 0),
                    "synthetic": b.get("synthetic", True)
                })
        except Exception:
            pass

    # Giữ nguyên 100% độ phân giải nếu điểm <= 60.000 điểm
    if len(pts) <= 60000:
        sub = pts
    else:
        import numpy as np
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), MAX_PREVIEW_POINTS, replace=False)
        sub = pts[idx]

    flat_pts = sub[:, :4].flatten().tolist()

    return {
        "status": "success",
        "filename": filename,
        "dataset": dataset,
        "meta": meta,
        "preview": {
            "augmented_points": flat_pts,
            "original_points": flat_pts,
            "num_augmented": len(pts),
            "num_original": len(pts),
            "boxes": boxes
        },
        "stats": {
            "mode": f"Preset: {dataset}",
            "initial_point_count": len(pts),
            "final_point_count": len(pts),
            "point_count_diff": 0,
            "inserted_objects_count": len(boxes),
            "weather_type": meta.get("weather_type", "preset"),
            "weather_level": meta.get("weather_level", "default"),
            "wet_road_enabled": "rain" in dataset or "monsoon" in dataset,
            "elapsed_ms": 15.0
        }
    }


@router.get("/api/vis/list")
def list_visual_snapshots():
    """Lists available rendered 3D visual snapshot images in output/vis/."""
    if not VIS_DIR.exists():
        return {"status": "success", "images": []}

    images = []
    for img in sorted(VIS_DIR.glob("*.png")):
        images.append({
            "name": img.name,
            "size_kb": round(img.stat().st_size / 1024, 1),
            "url": f"/vis/{img.name}"
        })
    return {"status": "success", "images": images}


@router.get("/vis/{image_name}")
def get_visual_image(image_name: str):
    """Serves rendered 3D image snapshots."""
    img_path = VIS_DIR / image_name
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(img_path)


@router.get("/slides")
def get_presentation_slides():
    """Serves the interactive HTML slide presentation."""
    if not SLIDES_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="slides.html not found.")
    return FileResponse(SLIDES_HTML_PATH)
