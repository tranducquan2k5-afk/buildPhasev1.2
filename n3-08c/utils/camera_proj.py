"""Chiếu LiDAR (LIDAR_TOP sensor frame của keyframe) lên 6 camera nuScenes + vẽ box 3D / điểm object chèn.

Dùng chung cho ảnh demo tĩnh (tools/case_gallery.py) và web app (webapp/backend).
Ảnh camera là ảnh thật của nuScenes: object chèn KHÔNG có trong ảnh (chỉ chèn vào LiDAR), chỉ hiện qua box chiếu +
điểm LiDAR chiếu lên. Thời tiết ảnh: task_b/camera_weather.py.
"""
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

CAMS = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT")
SURROUND = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT")  # 2 x 3
BOX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
STATUS_RGB = {"ok": (26, 158, 58), "missed": (214, 39, 40), "merged": (214, 39, 40), "confused": (255, 140, 0),
              "inserted": (245, 158, 11), "real": (148, 163, 184)}
OBJ_RGB = (177, 76, 255)


def lidar_to_cam(nusc, sample_token: str, cam: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """(T 4x4 LIDAR_TOP keyframe -> camera, intrinsic K 3x3, đường dẫn ảnh)."""
    from utils.transforms import transform_matrix
    sample = nusc.get("sample", sample_token)
    lid = nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
    cam_sd = nusc.get("sample_data", sample["data"][cam])
    l_cal = nusc.get("calibrated_sensor", lid["calibrated_sensor_token"])
    l_pose = nusc.get("ego_pose", lid["ego_pose_token"])
    c_cal = nusc.get("calibrated_sensor", cam_sd["calibrated_sensor_token"])
    c_pose = nusc.get("ego_pose", cam_sd["ego_pose_token"])
    T = (np.linalg.inv(transform_matrix(c_cal["translation"], c_cal["rotation"]))
         @ np.linalg.inv(transform_matrix(c_pose["translation"], c_pose["rotation"]))
         @ transform_matrix(l_pose["translation"], l_pose["rotation"])
         @ transform_matrix(l_cal["translation"], l_cal["rotation"]))
    return T, np.array(c_cal["camera_intrinsic"]), nusc.get_sample_data_path(cam_sd["token"])


def project(T: np.ndarray, K: np.ndarray, xyz: np.ndarray, w: int = 1600, h: int = 900):
    """Điểm sensor LiDAR -> pixel (u, v); mask trong ảnh (và trước camera); mask trước camera."""
    c = xyz @ T[:3, :3].T + T[:3, 3]
    front = c[:, 2] > 0.5
    uv = (c @ K.T)[:, :2] / np.maximum(c[:, 2:3], 1e-6)
    ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv, ok, front


def render_camera(nusc, sample_token: str, cam: str, boxes: Sequence[dict], obj_points: Optional[np.ndarray] = None,
                  size=(800, 450), weather: Optional[str] = None, level: Optional[str] = None, seed: int = 0,
                  mode: str = "photorealistic_box"):
    """Ảnh camera (PIL RGB) + kết xuất 2D quang học hoặc box chiếu lên.
    mode: 'photorealistic_box' | 'photorealistic' | 'controlnet_depth' | 'wireframe_only'.
    Returns (ảnh PIL RGB, số object chèn thấy được)."""
    from PIL import Image, ImageDraw
    from utils.visualization import to_box

    T, K, img_path = lidar_to_cam(nusc, sample_token, cam)
    img = Image.open(img_path).convert("RGB").resize(size)
    sx, sy = size[0] / 1600, size[1] / 900

    n_seen = 0
    for b in boxes:
        if not b.get("inserted"):
            continue
        cuv, cok, cfront = project(T, K, to_box(b).corners().T)
        if cfront.all() and cok.any():
            n_seen += 1

    if mode in ("photorealistic_box", "photorealistic", "photorealistic_only", "controlnet_depth"):
        from task_a.camera_synthesis import synthesize_camera_view
        rendered_arr = synthesize_camera_view(
            np.asarray(img), boxes, T, K,
            weather=weather, level=level, mode=mode, seed=seed
        )
        img = Image.fromarray(rendered_arr)
    else:
        if weather:
            from task_b.camera_weather import apply_camera_weather
            img = Image.fromarray(apply_camera_weather(np.asarray(img), weather, level, np.random.default_rng(seed)))
        dr = ImageDraw.Draw(img, "RGBA")
        for b in sorted(boxes, key=lambda b: bool(b.get("inserted"))):
            cuv, cok, cfront = project(T, K, to_box(b).corners().T)
            if not cfront.all() or not cok.any():
                continue
            ins = bool(b.get("inserted"))
            rgb = STATUS_RGB.get(b.get("status", "inserted" if ins else "real"), STATUS_RGB["inserted"])
            for a, c in BOX_EDGES:
                dr.line([(cuv[a, 0] * sx, cuv[a, 1] * sy), (cuv[c, 0] * sx, cuv[c, 1] * sy)],
                        fill=rgb + ((255,) if ins else (110,)), width=2 if ins else 1)
        if obj_points is not None and len(obj_points):
            uv, ok, _ = project(T, K, obj_points[:, :3])
            for u, v in uv[ok]:
                dr.ellipse([u * sx - 1.6, v * sy - 1.6, u * sx + 1.6, v * sy + 1.6], fill=OBJ_RGB + (255,))

    return img, int(n_seen)


def surround(nusc, sample_token: str, boxes: Sequence[dict], obj_points=None, mode: str = "photorealistic_box", **kw) -> Dict[str, tuple]:
    """{camera: (ảnh PIL, số object chèn thấy được)} cho 6 camera."""
    return {cam: render_camera(nusc, sample_token, cam, boxes, obj_points, seed=i, mode=mode, **kw) for i, cam in enumerate(CAMS)}
