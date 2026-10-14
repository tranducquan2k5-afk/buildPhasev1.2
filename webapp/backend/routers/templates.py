import json
from collections import Counter
from fastapi import APIRouter, HTTPException

try:
    from server_config import get_cached_nuscenes, get_cached_template_bank
except (ImportError, ValueError):
    from webapp.backend.server_config import get_cached_nuscenes, get_cached_template_bank

router = APIRouter(prefix="/api/templates", tags=["Templates"])


@router.get("/stats")
def get_templates_stats():
    """Returns statistics of the 4,111 templates in the Template Bank from Singleton RAM cache."""
    meta, cats = get_cached_template_bank()
    if not meta:
        raise HTTPException(status_code=404, detail="Template bank not found.")

    cat_counts = {k: len(v) for k, v in cats.items()}

    # Get sample tokens from nuScenes Singleton
    try:
        nusc = get_cached_nuscenes()
        sample_tokens = [s["token"] for s in nusc.sample]
    except Exception:
        sample_tokens = []

    return {
        "status": "success",
        "total_templates": len(meta),
        "total_categories": len(cat_counts),
        "categories": dict(sorted(cat_counts.items(), key=lambda x: -x[1])),
        "sample_tokens": sample_tokens[:20],
        "supported_upload_formats": [".bin", ".pcd", ".ply", ".las", ".laz", ".xyz", ".csv", ".txt", ".npy"],
        "supported_export_formats": [".pcd", ".bin", ".ply", ".las", ".xyz", ".npy"]
    }
