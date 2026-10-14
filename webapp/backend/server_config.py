import os
import sys
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("server_config")

# ==============================================================================
# 1. IMPORT AN TOÀN TỪ n3-08c (XỬ LÝ DẤU GẠCH NGANG)
# ==============================================================================
BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent
N3_08C_DIR = PROJECT_ROOT / "n3-08c"

# Đảm bảo n3-08c và PROJECT_ROOT luôn đứng đầu sys.path
for p in (str(N3_08C_DIR), str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# ==============================================================================
# CẤU HÌNH ĐƯỜNG DẪN THƯ MỤC
# ==============================================================================
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMPLATES_DIR = OUTPUT_DIR / "templates"
BANK_JSON_PATH = TEMPLATES_DIR / "bank.json"
EVAL_REPORT_PATH = OUTPUT_DIR / "eval_report.json"
DATASETS_DIR = PROJECT_ROOT / "datasets"
TEMP_DIR = WEBAPP_DIR / "temp_outputs"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Thư mục dữ liệu đã sinh sẵn & Vis & Slides
MONSOON_DATASET_DIR = OUTPUT_DIR / "vietnam_monsoon_dataset"
FOG_DATASET_DIR = OUTPUT_DIR / "fog_dense_dataset"
VIS_DIR = OUTPUT_DIR / "vis"

# Đường dẫn slides.html (kiểm tra cả Powerpoint/ và root)
POWERPOINT_SLIDES = PROJECT_ROOT.parent / "Powerpoint" / "slides.html"
ROOT_SLIDES = PROJECT_ROOT / "slides.html"
SLIDES_HTML_PATH = POWERPOINT_SLIDES if POWERPOINT_SLIDES.exists() else ROOT_SLIDES

# nuScenes settings
NUSCENES_VERSION = "v1.0-mini"
NUSCENES_DATAROOT = os.environ.get("NUSCENES_DATAROOT", str(DATASETS_DIR / "nuscenes"))

# ==============================================================================
# 3. GIỮ NGUYÊN 100% ĐỘ PHÂN GIẢI ĐIỂM TRÊN THREE.JS (NGƯỠNG 60.000 ĐIỂM)
# ==============================================================================
# Một frame Velodyne VLP-32C chỉ có ~34.000 điểm.
# Thiết lập ngưỡng 60.000 để bảo toàn 100% điểm gốc, giữ sắc nét vùng Occlusion ±15cm.
MAX_PREVIEW_POINTS = 60000

# ==============================================================================
# 2. CACHE SINGLETON CHO bank.json VÀ NuScenes (NẠP 1 LẦN DUY NHẤT VÀO RAM)
# ==============================================================================
_GLOBAL_BANK_CACHE: Optional[List[dict]] = None
_GLOBAL_CATS_CACHE: Optional[Dict[str, List[dict]]] = None
_GLOBAL_NUSC_INSTANCE = None


def get_cached_template_bank() -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Lazy Singleton Cache: nạp bank.json đúng 1 lần vào RAM khi server chạy."""
    global _GLOBAL_BANK_CACHE, _GLOBAL_CATS_CACHE
    if _GLOBAL_BANK_CACHE is None:
        if not BANK_JSON_PATH.exists():
            logger.warning(f"Template bank not found at {BANK_JSON_PATH}")
            _GLOBAL_BANK_CACHE = []
            _GLOBAL_CATS_CACHE = {}
        else:
            logger.info(f"Loading Template Bank (4,111 templates) into RAM from {BANK_JSON_PATH}...")
            with open(BANK_JSON_PATH, "r", encoding="utf-8") as f:
                raw_bank = json.load(f)

            import numpy as np
            # Lọc chất lượng cao theo yêu cầu:
            # - Xe tải/buýt/công trình lớn: num_points >= 400 VÀ điểm trải dài >= 55% chiều dài hộp (Spatial Span Check)
            # - Ô tô con (car): num_points >= 180 VÀ điểm trải dài >= 55% chiều dài hộp
            # - Xe nhỏ / Người / Vật thể (motorcycle, bicycle, pedestrian, barrier, trafficcone): num_points >= 60
            _GLOBAL_BANK_CACHE = []
            for item in raw_bank:
                cat = item.get("category", "")
                pts = item.get("num_points", 0)
                is_large = any(k in cat for k in ("vehicle.truck", "vehicle.bus", "vehicle.construction", "vehicle.trailer"))
                is_car = (cat == "vehicle.car")

                if is_large:
                    if pts < 400:
                        continue
                    # Spatial Span Check: trải dài >= 55% chiều dài length
                    rel = item.get("file", "").replace("\\", "/")
                    npz_path = TEMPLATES_DIR / rel
                    if npz_path.exists():
                        try:
                            data = np.load(str(npz_path))
                            p = data["points"]
                            l = item.get("size", [1.0, 1.0, 1.0])[1]
                            span_ratio = float(p[:, 0].max() - p[:, 0].min()) / (l + 1e-6)
                            if span_ratio >= 0.55:
                                item["span_ratio"] = span_ratio
                                _GLOBAL_BANK_CACHE.append(item)
                        except Exception:
                            pass
                elif is_car:
                    if pts < 180:
                        continue
                    rel = item.get("file", "").replace("\\", "/")
                    npz_path = TEMPLATES_DIR / rel
                    if npz_path.exists():
                        try:
                            data = np.load(str(npz_path))
                            p = data["points"]
                            l = item.get("size", [1.0, 1.0, 1.0])[1]
                            span_ratio = float(p[:, 0].max() - p[:, 0].min()) / (l + 1e-6)
                            if span_ratio >= 0.55:
                                item["span_ratio"] = span_ratio
                                _GLOBAL_BANK_CACHE.append(item)
                        except Exception:
                            pass
                elif any(k in cat for k in ("motorcycle", "bicycle", "pedestrian", "barrier", "trafficcone")):
                    if pts >= 60:
                        _GLOBAL_BANK_CACHE.append(item)
                else:
                    if pts >= 60:
                        _GLOBAL_BANK_CACHE.append(item)

            _GLOBAL_CATS_CACHE = {}
            for item in _GLOBAL_BANK_CACHE:
                cat = item.get("category", "unknown")
                if cat not in _GLOBAL_CATS_CACHE:
                    _GLOBAL_CATS_CACHE[cat] = []
                _GLOBAL_CATS_CACHE[cat].append(item)
            logger.info(f"Singleton Bank Loaded & Filtered: {len(_GLOBAL_BANK_CACHE)} high-quality templates (large >= 400pts & span >= 55%, cars >= 180pts), {len(_GLOBAL_CATS_CACHE)} categories.")
    return _GLOBAL_BANK_CACHE, _GLOBAL_CATS_CACHE


def get_cached_nuscenes():
    """Lazy Singleton Cache: nạp NuScenes instance đúng 1 lần vào RAM."""
    global _GLOBAL_NUSC_INSTANCE
    if _GLOBAL_NUSC_INSTANCE is None:
        from utils.nuscenes_loader import load_nuscenes
        logger.info(f"Initializing NuScenes Singleton from {NUSCENES_DATAROOT}...")
        _GLOBAL_NUSC_INSTANCE = load_nuscenes(dataroot=NUSCENES_DATAROOT, version=NUSCENES_VERSION, verbose=False)
        logger.info("NuScenes Singleton initialized successfully.")
    return _GLOBAL_NUSC_INSTANCE
