"""Cấu hình chung cho N3-08C — Synthetic Point Cloud Augmentation.

DATAROOT có thể override bằng biến môi trường NUSCENES_DATAROOT.
"""
import os

# Thư mục gốc repo (cha của n3-08c\), không phụ thuộc vị trí clone
PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _find_dataroot() -> str:
    # 1. Biến môi trường người dùng đặt
    env_root = os.environ.get("NUSCENES_DATAROOT")
    if env_root and os.path.isfile(os.path.join(env_root, "v1.0-mini", "sample.json")):
        return env_root
    # 2. Thư mục cục bộ trong repo
    local_root = os.path.join(PROJECT_ROOT, "datasets", "nuscenes")
    if os.path.isfile(os.path.join(local_root, "v1.0-mini", "sample.json")):
        return local_root
    # 3. Các vị trí quen thuộc trong môi trường dev
    candidates = [
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "nuscenes-project", "datasets", "nuscenes")),
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "nuscenes-project", "datasets", "nuscenes")),
        r"D:\VinAI\Build phase\nuscenes-project\datasets\nuscenes",
        r"C:\VinAI\Build phase\nuscenes-project\datasets\nuscenes",
    ]
    for cand in candidates:
        if os.path.isfile(os.path.join(cand, "v1.0-mini", "sample.json")):
            return cand
    return local_root

# ===== Dataset =====
DATAROOT: str = _find_dataroot()

VERSION: str = "v1.0-mini"
LIDAR_CHANNEL: str = "LIDAR_TOP"

# ===== Output paths =====
OUTPUT_DIR: str = os.environ.get("N308C_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
TEMPLATE_DIR: str = os.path.join(OUTPUT_DIR, "templates")    # object templates (object-local frame)
AUGMENTED_DIR: str = os.path.join(OUTPUT_DIR, "augmented")   # Task A: scene sau khi chèn object
WEATHER_DIR: str = os.path.join(OUTPUT_DIR, "weather")       # Task B: scene sau khi mô phỏng thời tiết
VIS_DIR: str = os.path.join(OUTPUT_DIR, "vis")               # ảnh BEV

SEED: int = 42

# ===== Task A: thresholds =====
MIN_DISTANCE: float = 9.0           # khoảng cách tối thiểu từ sensor tới object chèn (m)
MAX_DISTANCE: float = 70.0          # khoảng cách tối đa (m)
COLLISION_IOU_THRESH: float = 0.0   # BEV IoU > ngưỡng này = va chạm
MIN_TEMPLATE_POINTS: int = 20       # template ít điểm hơn bị bỏ
TEMPLATE_GROUND_MARGIN: float = 0.1  # bỏ điểm trong dải này phía trên đáy box (mặt đường lẫn vào); 0 = tắt
TEMPLATE_CATEGORIES: tuple = ("vehicle", "human.pedestrian", "movable_object")  # prefix category cho bank
# Template phải có attribute này mới được chèn (prefix category -> attribute nuScenes).
# Xe 2 bánh: phần lớn box trong nuScenes là xe đỗ không người lái; kịch bản giao thông cần xe đang chạy.
TEMPLATE_REQUIRED_ATTRIBUTES: dict = {"vehicle.motorcycle": "cycle.with_rider", "vehicle.bicycle": "cycle.with_rider"}
MAX_OBSTACLE_POINTS: int = 5        # footprint chứa nhiều điểm cao > 0.3 m hơn ngưỡng này = vật cản không nhãn
MAX_DISTANCE_SCALE: float = 1.5    # object được dời ra xa tối đa 1.5 lần khoảng cách lúc cắt template
MIN_VISIBLE_POINTS: int = 5         # object còn ít điểm hơn sau che khuất thì bỏ (nuScenes cũng có box 1-5 điểm)

# Range image LIDAR_TOP (Velodyne HDL-32E, 20 Hz): 32 ring x ~1084 điểm mỗi vòng (bước azimuth ~0.33°)
LIDAR_RINGS: int = 32
LIDAR_AZIMUTH_BINS: int = 1084
OCCLUSION_EPS: float = 0.2          # (m) điểm xa hơn điểm gần nhất cùng tia quá eps thì bị che
OCCLUSION_NEAR_RANGE: float = 10.0  # (m) object gần hơn: che cả ring ±1 (ring ước lượng có thể lệch 1)
N_SWEEPS: int = 10                  # keyframe + 9 sweep trước (~0.45 s), như detector nuScenes đọc

# Ground model (tham số chọn theo sai số đáy box xe thật, xem task_a/ground_plane.py)
GROUND: dict = {
    "z_threshold": -1.5,   # RANSAC chỉ xét điểm thấp hơn (LIDAR_TOP cao ~1.84 m)
    "band": 0.6,           # ground candidate: |khoảng cách tới plane| < band (m)
    "cell": 1.0,           # lưới BEV (m) để lấy điểm thấp nhất mỗi cell
    "cell_tol": 0.2,       # giữ điểm trong cell có z <= z_min_cell + cell_tol
    "radius": 3.0,         # bán kính lấy median z cục bộ (m)
    "min_support": 5,      # ít điểm hơn -> fallback về plane, coi là vùng không thấy mặt đất
}

# Map layer hợp lệ để đặt object, theo prefix category (prefix dài nhất khớp được dùng)
PLACEMENT_LAYERS: dict = {
    "vehicle": ("drivable_area",),
    "movable_object": ("drivable_area",),
    "human.pedestrian": ("walkway", "ped_crossing"),
}
DEFAULT_LAYERS: tuple = ("drivable_area",)

# ===== Task B: weather levels =====
RAIN_LEVELS: dict = {
    "light": {"dropout": 0.1, "noise_ratio": 0.03},
    "moderate": {"dropout": 0.2, "noise_ratio": 0.07},
    "heavy": {"dropout": 0.3, "noise_ratio": 0.12},
}
FOG_VISIBILITY: dict = {"light": 200.0, "moderate": 50.0, "dense": 20.0}  # (m)
