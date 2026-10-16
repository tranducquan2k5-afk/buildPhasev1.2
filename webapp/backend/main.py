"""Web app demo N3-08C (FastAPI): sinh kịch bản VN trên nuScenes, xem 3D + 6 camera, tải kết quả, xem số đo thật.

Giao diện + kiến trúc web lấy ý tưởng từ repo nhóm tranducquan2k5-afk/buildPhasev1.2 (FastAPI + Three.js + 6 camera +
Docker), nối với pipeline của repo này. Mọi số liệu hiển thị đọc từ output/ (đo thật), không có số gõ tay.

Chạy: python run_app.py   (hoặc docker compose up — xem Dockerfile)
"""
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import OUTPUT, ROOT, TASKS, WEATHER_LEVELS, Engine

app = FastAPI(title="N3-08C — Edge case LiDAR giao thông Việt Nam", version="1.0")
engine = Engine()
FRONT = ROOT / "webapp" / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONT / "static")), name="static")
(OUTPUT / "sequences").mkdir(parents=True, exist_ok=True)  # tools/sequence_export.py ghi vào đây
for url, path in (("/report", OUTPUT / "report"), ("/cases", OUTPUT / "cases"), ("/videos", OUTPUT / "videos"),
                  ("/sequences", OUTPUT / "sequences"), ("/demo", ROOT / "docs" / "demo")):
    if path.is_dir():
        app.mount(url, StaticFiles(directory=str(path), html=True), name=url.strip("/"))


class CustomGroup(BaseModel):
    type: str
    count: int = Field(1, ge=0, le=150)
    placement: str = "traffic"


class GenerateRequest(BaseModel):
    sample_token: str
    scenario: str = "moto_swarm"
    seed: int = Field(42, ge=0)
    weather: str = "none"
    level: Optional[str] = None
    puddles: int = Field(0, ge=0, le=30)
    sweeps: bool = False
    custom: Optional[List[CustomGroup]] = None  # dùng khi scenario = "custom"


@app.get("/")
def index():
    return FileResponse(FRONT / "index.html")


@app.get("/slides")
def slides():
    p = ROOT / "docs" / "demo" / "slides.html"
    if not p.is_file():
        raise HTTPException(404, "chưa có docs/demo/slides.html (chạy tools/build_demo.py)")
    return FileResponse(p)


@app.get("/api/meta")
def meta():
    return engine.meta()


@app.post("/api/generate")
def generate(req: GenerateRequest):
    if req.weather not in WEATHER_LEVELS:
        raise HTTPException(400, f"weather phải thuộc {list(WEATHER_LEVELS)}")
    try:
        return engine.generate(req.sample_token, req.scenario, req.seed, req.weather, req.level, req.puddles,
                               req.sweeps, [g.model_dump() for g in req.custom] if req.custom else None)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/api/cameras/{task_id}")
def cameras(task_id: str, mode: str = "photorealistic_box"):
    if not (TASKS / task_id / "state.json").is_file():
        raise HTTPException(404, "không có task")
    return engine.cameras(task_id, mode=mode)


@app.post("/api/detect/{task_id}")
def detect(task_id: str):
    if not (TASKS / task_id / "state.json").is_file():
        raise HTTPException(404, "không có task")
    return engine.detect(task_id)


@app.get("/api/results")
def results():
    return engine.results()


@app.get("/api/sequences")
def sequences():
    """Chuỗi khung có video (tools/sequence_export.py): {tên: {label, video, frames: [...], totals}}."""
    p = OUTPUT / "sequences" / "index.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}


@app.get("/api/case/{name}")
def case(name: str):
    try:
        return engine.case(name)
    except FileNotFoundError:
        raise HTTPException(404, "không có trường hợp")


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload point cloud tùy chỉnh (.pcd, .bin, .ply, .las, .laz, .xyz, .csv, .npy)."""
    try:
        content = await file.read()
        return engine.upload_point_cloud_data(content, file.filename)
    except Exception as e:
        raise HTTPException(400, f"Lỗi đọc mây điểm tải lên: {e}")


@app.get("/api/download/{task_id}/{kind}")
def download(task_id: str, kind: str):
    files = {
        "pcd": "augmented.pcd",
        "bin": "augmented.bin",
        "ply": "augmented.ply",
        "xyz": "augmented.xyz",
        "labels": "labels.json",
        "zip": "bundle.zip"
    }
    if kind not in files or "/" in task_id or "\\" in task_id or ".." in task_id:
        raise HTTPException(400, "yêu cầu không hợp lệ")
    p = TASKS / task_id / files[kind]
    if not p.is_file():
        raise HTTPException(404, "không có file")
    return FileResponse(p, filename=f"{task_id}_{files[kind]}")


@app.get("/health")
def health():
    return {"status": "ok"}
