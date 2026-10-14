import json
import uuid
import zipfile
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from server_config import TEMP_DIR
    from schemas.payload import NuscenesAugmentRequest
    from services.engine_nuscenes import NuscenesSandboxEngine
    from services.engine_upload import CustomUploadEngine
except (ImportError, ValueError):
    from webapp.backend.server_config import TEMP_DIR
    from webapp.backend.schemas.payload import NuscenesAugmentRequest
    from webapp.backend.services.engine_nuscenes import NuscenesSandboxEngine
    from webapp.backend.services.engine_upload import CustomUploadEngine
from utils.io_converter import save_point_cloud

router = APIRouter(prefix="/api/augment", tags=["Augmentation Engine"])

engine_mode1 = NuscenesSandboxEngine()
engine_mode2 = CustomUploadEngine()


@router.post("/nuscenes")
def augment_nuscenes(req: NuscenesAugmentRequest):
    """Mode 1: nuScenes Sandbox Mode (Vector Map-aware object insertion + physical weather)."""
    try:
        task_id = uuid.uuid4().hex[:12]
        task_dir = TEMP_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        res = engine_mode1.run(
            sample_token=req.sample_token,
            enable_insertion=req.enable_insertion,
            categories=req.categories,
            num_objects=req.num_objects,
            weather_type=req.weather_type,
            weather_level=req.weather_level,
            enable_wet_road=req.enable_wet_road
        )

        out_fmt = req.output_format.lower()
        if not out_fmt.startswith("."):
            out_fmt = "." + out_fmt

        sample_name = res.get("sample_token", "nuscenes")[:8]
        out_cloud_name = f"augmented_{sample_name}{out_fmt}"
        out_cloud_path = task_dir / out_cloud_name
        save_point_cloud(res["augmented_points"], out_cloud_path, format=out_fmt)

        ann_path = task_dir / "annotations_cvat_nuscenes.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "mode": "Mode 1: nuScenes Sandbox",
                "sample_token": res.get("sample_token"),
                "weather": {
                    "type": req.weather_type,
                    "level": req.weather_level,
                    "wet_road": req.enable_wet_road
                },
                "total_points": len(res["augmented_points"]),
                "num_annotations": len(res["annotations"]),
                "annotations": res["annotations"]
            }, f, indent=2)

        zip_path = task_dir / f"augmented_bundle_{task_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_cloud_path, arcname=out_cloud_name)
            z.write(ann_path, arcname="annotations.json")

        res["stats"]["task_id"] = task_id
        res["stats"]["output_file"] = out_cloud_name
        res["stats"]["input_file"] = f"nuScenes sample [{sample_name}]"

        return {
            "status": "success",
            "task_id": task_id,
            "stats": res["stats"],
            "preview": res["preview_data"],
            "download_urls": {
                "pointcloud": f"/api/download/{task_id}/pointcloud",
                "annotations": f"/api/download/{task_id}/annotations",
                "zip": f"/api/download/{task_id}/zip"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def augment_upload(
    file: Optional[UploadFile] = File(None),
    use_sample: bool = Form(False),
    enable_insertion: bool = Form(True),
    categories: Optional[str] = Form(None),
    num_objects: int = Form(3),
    weather_type: str = Form("none"),
    weather_level: str = Form("moderate"),
    enable_wet_road: bool = Form(False),
    output_format: str = Form(".pcd")
):
    """Mode 2: Custom Upload Mode (Mapless Fallback + RANSAC Ground + Task B Weather)."""
    try:
        task_id = uuid.uuid4().hex[:12]
        task_dir = TEMP_DIR / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        if use_sample or not file:
            # Pick a sample .bin from dataset
            sample_dir = Path("output/vietnam_monsoon_dataset")
            bins = list(sample_dir.glob("*.bin")) if sample_dir.exists() else []
            if not bins:
                bins = list(Path("output/augmented").glob("*.bin"))
            if not bins:
                raise HTTPException(status_code=404, detail="No sample file available on server.")
            input_bytes = bins[0].read_bytes()
            input_filename = bins[0].name
            input_ext = ".bin"
        else:
            input_bytes = await file.read()
            input_filename = file.filename or "uploaded.bin"
            input_ext = Path(input_filename).suffix.lower()

        cat_list = None
        if categories:
            try:
                cat_list = json.loads(categories)
            except Exception:
                cat_list = [c.strip() for c in categories.split(",") if c.strip()]

        res = engine_mode2.run(
            input_file_or_bytes=input_bytes,
            input_ext=input_ext,
            enable_insertion=enable_insertion,
            categories=cat_list,
            num_objects=num_objects,
            weather_type=weather_type,
            weather_level=weather_level,
            enable_wet_road=enable_wet_road
        )

        out_fmt = output_format.lower()
        if not out_fmt.startswith("."):
            out_fmt = "." + out_fmt

        stem = Path(input_filename).stem
        out_cloud_name = f"augmented_{stem}{out_fmt}"
        out_cloud_path = task_dir / out_cloud_name
        save_point_cloud(res["augmented_points"], out_cloud_path, format=out_fmt)

        ann_path = task_dir / "annotations_cvat_nuscenes.json"
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump({
                "task_id": task_id,
                "mode": "Mode 2: Custom Upload (Mapless Fallback)",
                "input_file": input_filename,
                "weather": {
                    "type": weather_type,
                    "level": weather_level,
                    "wet_road": enable_wet_road
                },
                "total_points": len(res["augmented_points"]),
                "num_annotations": len(res["annotations"]),
                "annotations": res["annotations"]
            }, f, indent=2)

        zip_path = task_dir / f"augmented_bundle_{task_id}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            z.write(out_cloud_path, arcname=out_cloud_name)
            z.write(ann_path, arcname="annotations.json")

        res["stats"]["task_id"] = task_id
        res["stats"]["output_file"] = out_cloud_name
        res["stats"]["input_file"] = input_filename

        return {
            "status": "success",
            "task_id": task_id,
            "stats": res["stats"],
            "preview": res["preview_data"],
            "download_urls": {
                "pointcloud": f"/api/download/{task_id}/pointcloud",
                "annotations": f"/api/download/{task_id}/annotations",
                "zip": f"/api/download/{task_id}/zip"
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{task_id}/{file_type}")
def download_result(task_id: str, file_type: str):
    """File download endpoint for generated point clouds and annotations."""
    task_dir = TEMP_DIR / task_id
    if not task_dir.exists():
        raise HTTPException(status_code=404, detail="Task not found or expired.")

    if file_type == "pointcloud":
        clouds = [f for f in task_dir.iterdir() if f.suffix not in (".json", ".zip")]
        if not clouds:
            raise HTTPException(status_code=404, detail="Point cloud file not found.")
        return FileResponse(clouds[0], filename=clouds[0].name)

    elif file_type == "annotations":
        ann = task_dir / "annotations_cvat_nuscenes.json"
        if not ann.exists():
            raise HTTPException(status_code=404, detail="Annotation file not found.")
        return FileResponse(ann, filename="annotations.json")

    elif file_type == "zip":
        zips = list(task_dir.glob("*.zip"))
        if not zips:
            raise HTTPException(status_code=404, detail="ZIP bundle not found.")
        return FileResponse(zips[0], filename=zips[0].name)

    else:
        raise HTTPException(status_code=400, detail="Invalid file type requested.")
