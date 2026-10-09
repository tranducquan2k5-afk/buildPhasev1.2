"""Pipeline Task A end-to-end: source_token -> target_token -> augmented scene.

Output (config.AUGMENTED_DIR):
    <target_sample_token>.bin   point cloud (N+M, 5) float32, sensor frame (cùng format nuScenes)
    <target_sample_token>.json  annotations mới (sensor + global frame) + thống kê cho evaluate.py

Chạy:
    python task_a/run_insertion.py                                   # demo: sample[0] -> sample[10]
    python task_a/run_insertion.py --source-ann <ann_token> --target <sample_token>
    python task_a/run_insertion.py --source-sample <tok> --target <tok> --num-objects 3 --vis
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.extract_object import extract_all_templates, extract_object_template  # noqa: E402
from task_a.ground_plane import (GroundModel, get_map_area, layers_for_category,  # noqa: E402
                                 load_map_for_sample, project_z_to_ground)
from task_a.merge import create_annotation, insert_object  # noqa: E402
from task_a.placement import sample_valid_position  # noqa: E402
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose,  # noqa: E402
                                   get_lidar_token, get_pointcloud, load_nuscenes, save_bin)
from utils.transforms import box_sensor_to_global  # noqa: E402


def collect_templates(nusc, source_sample: Optional[str], source_anns: Optional[List[str]],
                      num_objects: int, rng: Optional[np.random.Generator] = None) -> List[dict]:
    """Template từ annotation chỉ định, hoặc random num_objects từ source_sample với mọi category."""
    if source_anns:
        return [extract_object_template(nusc, nusc.get("sample_annotation", a)["sample_token"], a)
                for a in source_anns]
    templates = extract_all_templates(nusc, source_sample, categories=config.TEMPLATE_CATEGORIES)
    
    if not templates:
        return []
        
    rng = rng or np.random.default_rng(config.SEED)
    # Lựa chọn ngẫu nhiên để "chèn mọi thứ" thay vì chỉ chọn vật thể to nhất/nhiều điểm nhất
    rng.shuffle(templates)
    return templates[:num_objects]


def run_insertion(nusc, target_token: str, templates: List[dict], n_attempts: int = 100,
                  rng: Optional[np.random.Generator] = None) -> dict:
    """Chèn lần lượt các template vào target sample. Trả về dict points + annotations + thống kê."""
    rng = rng or np.random.default_rng(config.SEED)
    t0 = time.perf_counter()

    points = get_pointcloud(nusc, target_token)
    existing = get_annotations(nusc, target_token)
    ground = GroundModel.from_points(points, rng=rng)
    nusc_map = load_map_for_sample(nusc, target_token)
    areas = {}  # cache vùng đặt object theo bộ layer

    new_anns = []
    for tpl in templates:
        layers = layers_for_category(tpl["category"])
        if layers not in areas:
            areas[layers] = get_map_area(nusc, nusc_map, target_token, layers)
        
        # Khớp khoảng cách: chèn vật thể ở khoảng cách tương đương với lúc trích xuất (± 5m)
        orig_dist = tpl.get("distance", -1)
        d_range = (orig_dist - 5.0, orig_dist + 5.0) if orig_dist > 0 else None
            
        pos = sample_valid_position(areas[layers], existing + new_anns, tpl["size"], n_attempts, rng, distance_range=d_range)
        if pos is None:
            continue
        points = insert_object(points, tpl, pos, ground)
        new_anns.append(create_annotation(tpl, pos, project_z_to_ground(pos[0], pos[1], ground)))
    drivable = areas.get(config.DEFAULT_LAYERS) or get_map_area(nusc, nusc_map, target_token)

    sd_token = get_lidar_token(nusc, target_token)
    calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)
    return {
        "points": points,
        "target_sample_token": target_token,
        "source_ann_tokens": [t.get("ann_token") for t in templates],
        "n_requested": len(templates),
        "n_inserted": len(new_anns),
        "elapsed_s": time.perf_counter() - t0,
        "annotations_sensor": new_anns,
        "annotations_global": [box_sensor_to_global(a, calib, pose) for a in new_anns],
        "existing_annotations": existing,
        "drivable": drivable,
    }


def save_result(result: dict, out_dir: str = config.AUGMENTED_DIR) -> str:
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, result["target_sample_token"])
    save_bin(stem + ".bin", result["points"])
    meta = {k: v for k, v in result.items() if k not in ("points", "drivable", "existing_annotations")}
    with open(stem + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    return stem


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--source-sample", help="sample lấy object (mặc định sample[0])")
    ap.add_argument("--source-ann", nargs="*", help="annotation token cụ thể cần chèn")
    ap.add_argument("--target", help="sample token đích (mặc định sample[10])")
    ap.add_argument("--num-objects", type=int, default=3)
    ap.add_argument("--n-attempts", type=int, default=100)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--vis", action="store_true", help="mở Open3D sau khi chèn")
    args = ap.parse_args()

    nusc = load_nuscenes()
    source = args.source_sample or nusc.sample[0]["token"]
    target = args.target or nusc.sample[10]["token"]

    rng = np.random.default_rng(args.seed)
    templates = collect_templates(nusc, source, args.source_ann, args.num_objects, rng=rng)
    result = run_insertion(nusc, target, templates, args.n_attempts, rng)
    stem = save_result(result)

    print(f"target {target}: chèn {result['n_inserted']}/{result['n_requested']} objects "
          f"trong {result['elapsed_s']:.2f}s")
    for a in result["annotations_sensor"]:
        print(f"  {a['category']:25s} at ({a['translation'][0]:.1f}, {a['translation'][1]:.1f}) "
              f"{a['num_lidar_pts']} pts")
    print("saved", stem + ".bin/.json")

    from utils.visualization import plot_bev, show_open3d
    png = os.path.join(config.VIS_DIR, f"insertion_{target}.png")
    plot_bev(result["points"], result["existing_annotations"], result["annotations_sensor"],
             region=result["drivable"], title=f"insertion {target}", save_path=png)
    print("saved", png)
    if args.vis:
        show_open3d(result["points"], result["existing_annotations"], result["annotations_sensor"])
