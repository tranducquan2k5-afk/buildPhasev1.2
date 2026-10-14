"""
API Router for 6-Camera Surround View (2D-3D Multi-Modal Fusion)
================================================================
Endpoints:
- GET /api/cameras/surround: Returns 6-camera projections for a preset or sample token.
- POST /api/cameras/project: Projects 3D boxes and LiDAR points onto 6 cameras on demand.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

try:
    from server_config import FOG_DATASET_DIR, MONSOON_DATASET_DIR, get_cached_nuscenes
except (ImportError, ValueError):
    from webapp.backend.server_config import (
        FOG_DATASET_DIR,
        MONSOON_DATASET_DIR,
        get_cached_nuscenes,
    )

from services.camera_projection import CameraProjectionEngine
from utils.io_converter import load_point_cloud

router = APIRouter(prefix="/api/cameras", tags=["Multi-Modal Cameras (2D-3D Fusion)"])
logger = logging.getLogger("router_cameras")

_PROJECTION_ENGINE: Optional[CameraProjectionEngine] = None


def get_projection_engine() -> CameraProjectionEngine:
    global _PROJECTION_ENGINE
    if _PROJECTION_ENGINE is None:
        _PROJECTION_ENGINE = CameraProjectionEngine()
    return _PROJECTION_ENGINE


class CameraProjectRequest(BaseModel):
    sample_token: str
    boxes: List[Dict[str, Any]] = []
    weather_type: str = "none"
    weather_level: str = "moderate"


@router.get("/surround")
def get_surround_view(
    sample_token: Optional[str] = Query(None),
    dataset: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    weather_type: Optional[str] = Query("none"),
    weather_level: Optional[str] = Query("moderate"),
):
    """
    Returns 6 cockpit-ordered surround camera images for a given sample or preset.
    """
    nusc = get_cached_nuscenes()
    engine = get_projection_engine()

    boxes = []
    target_sample_token = sample_token

    # 1. Nếu có filename và dataset, load metadata từ preset
    if dataset and filename:
        base_dir = MONSOON_DATASET_DIR if "monsoon" in dataset else FOG_DATASET_DIR
        json_path = base_dir / Path(filename).with_suffix(".json")
        bin_path = base_dir / filename

        if json_path.exists():
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                target_sample_token = meta.get("base_sample_token", target_sample_token)
                boxes = meta.get("annotations_sensor", [])
                weather_type = meta.get("weather_type", weather_type)
                weather_level = meta.get("weather_level", weather_level)
            except Exception as e:
                logger.error(f"Error loading preset json {json_path}: {e}")

    # Fallback sample token
    if not target_sample_token:
        target_sample_token = nusc.sample[0]["token"]

    try:
        res = engine.generate_surround_view(
            sample_token=target_sample_token,
            boxes=boxes,
            weather_type=weather_type or "none",
            weather_level=weather_level or "moderate"
        )
        return res
    except Exception as e:
        logger.error(f"Error generating surround view: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project")
def project_custom_scene(req: CameraProjectRequest):
    """
    Projects arbitrary 3D bounding boxes from a live augmented scene onto 6 surround cameras.
    """
    engine = get_projection_engine()
    try:
        res = engine.generate_surround_view(
            sample_token=req.sample_token,
            boxes=req.boxes,
            weather_type=req.weather_type,
            weather_level=req.weather_level
        )
        return res
    except Exception as e:
        logger.error(f"Error in camera project endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
