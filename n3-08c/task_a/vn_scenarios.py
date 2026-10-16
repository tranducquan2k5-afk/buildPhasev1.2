"""Kịch bản VN với vật thể không có trong nuScenes (docs/danh_muc_kich_ban.md).

Mỗi kịch bản = danh sách nhóm vật thể, mỗi nhóm:
    nguồn "bank"  cắt mẫu nuScenes (xe máy có người lái, người đi bộ) qua TemplateMatcher như scenarios.py
    nguồn "rc"    ray-cast hình dựng sẵn (task_a/raycast.py): xe máy chở 2-3 người / chở hàng, xe ba gác, xe kéo,
                  xe hàng rong, chó, bò, xe bò; mỗi object dựng mới (kích thước hàng hoá / số người ngẫu nhiên)
Tư thế lấy mẫu trong EGO frame như scenarios.py, kiểm tra vị trí bằng placement.check_position.
Object ray-cast còn 0 điểm sau che khuất bị bỏ và đếm vào reject "invisible" (giới hạn cảm biến, vd chó ở xa).
Xe bò: bò + xe gỗ là 2 box riêng (như truck + trailer của nuScenes).

rc_control_car / rc_control_moto: ô tô / xe máy RAY-CAST ở tư thế bình thường, so với control_car / control_2w
(cắt mẫu thật) để kiểm chứng detector "thấy" vật thể ray-cast như vật thể thật.

Chạy: python task_a/vn_scenarios.py --scenario ba_gac_slow   (sample[10], in số object / số điểm)
"""
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.extract_object import TemplateBank  # noqa: E402
from task_a.ground_plane import GroundModel, get_map_area, load_map_for_sample  # noqa: E402
from task_a.merge import create_annotation, insert_object, points_in_new_box  # noqa: E402
from task_a.placement import check_position  # noqa: E402
from task_a.raycast import build_model, insert_raycast  # noqa: E402
from task_a.scenarios import EgoFrame, TemplateMatcher, _swarm  # noqa: E402
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose, get_lidar_token,  # noqa: E402
                                   get_pointcloud)
from utils.transforms import box_sensor_to_global, quaternion_to_yaw, wrap_angle  # noqa: E402

D = np.radians


def _s(rng):
    return 1 if rng.random() < 0.5 else -1


def _traffic(rng):      # cùng chiều, trước / sau, làn ego / làn bên
    return _s(rng) * rng.uniform(9.0, 30.0), rng.uniform(-4, 4), D(rng.uniform(-10, 10))


def _ahead(rng):        # phía trước trong / cạnh làn ego
    return rng.uniform(9.0, 25.0), rng.uniform(-3, 3), D(rng.uniform(-8, 8))


def _curb_slow(rng):    # sát lề, phía trước
    return rng.uniform(9.0, 30.0), _s(rng) * rng.uniform(1.5, 3.5), D(rng.uniform(-5, 5))


def _curb(rng):         # lấn từ lề xuống lòng đường, đứng / đi chậm 2 chiều
    return rng.uniform(9.0, 25.0), _s(rng) * rng.uniform(2.5, 5.5), (0 if rng.random() < 0.5 else np.pi) + D(
        rng.uniform(-10, 10))


def _vendor_right_curb(rng):
    """Đặt xe đẩy hàng rong sát lề đường bên phải (y < 0), cự ly an toàn x >= 9.0m, tránh tim đường và nắp capo xe chủ."""
    return rng.uniform(9.0, 26.0), -rng.uniform(2.8, 4.2), D(rng.uniform(-6, 6))


def _dart(rng):         # lao ngang trước xe
    return rng.uniform(9.0, 20.0), rng.uniform(-3, 3), _s(rng) * D(rng.uniform(70, 110))


def _wander(rng):       # đứng / đi chậm hướng bất kỳ trên lòng đường
    return rng.uniform(9.0, 30.0), rng.uniform(-3, 3), rng.uniform(-np.pi, np.pi)


def _ped_road(rng):     # người đi bộ dưới lòng đường cạnh xe hàng rong
    return rng.uniform(9.0, 20.0), _s(rng) * rng.uniform(2, 5), rng.uniform(-np.pi, np.pi)


def _group_cross(rng):  # tâm nhóm người băng ngang trước xe, hướng vuông góc làn
    return rng.uniform(9.0, 20.0), rng.uniform(-3, 3), _s(rng) * D(rng.uniform(75, 105))


def _swarm_lane(rng):
    """Bố trí bầy quanh xe (15 m): bám song song trục làn đường,
    làn phải (y <= 0) đi thuận chiều (yaw ~ 0 +- 8 deg),
    làn trái (y > 0) đi ngược chiều (yaw ~ pi +- 8 deg). Tuyệt đối không xoay ngang 90 độ."""
    while True:
        x, y = rng.uniform(-15, 15), rng.uniform(-6, 6)
        if np.hypot(x, y) >= 9.0 and np.hypot(x - 1.3, y) <= 15:
            if y > 0.0:
                yaw = np.pi + D(rng.uniform(-8, 8))
            else:
                yaw = D(rng.uniform(-8, 8))
            return x, y, yaw


def _crowd(rng):        # đám đông đi lại quanh xe trong 15 m, hướng bất kỳ (chợ, tan trường)
    while True:
        x, y = rng.uniform(-15, 15), rng.uniform(-8, 8)
        if np.hypot(x, y) >= 9.0 and np.hypot(x - 1.3, y) <= 15:
            return x, y, rng.uniform(-np.pi, np.pi)


@dataclass
class Group:
    source: str                      # "bank" | "rc"
    what: Tuple[str, ...]            # bank: prefix category; rc: tên hình (raycast.SHAPES)
    count: Tuple[int, int]
    pose: Callable
    speed: Tuple[float, float]
    trailer: Optional[str] = None    # rc: hình kéo phía sau (box riêng), vd xe bò
    cluster: Optional[Tuple[float, float]] = None  # (nửa dài, nửa rộng) m: cả nhóm quanh 1 tâm, cùng hướng ±15°
    label: Optional[str] = None      # tên hiển thị trong group_stats (kịch bản tự chọn: loại trong CUSTOM_TYPES)


@dataclass
class VNScenario:
    name: str
    description: str
    groups: List[Group] = field(default_factory=list)


MOTO = ("vehicle.motorcycle",)  # kịch bản VN mới chỉ dùng xe máy (xe đạp giữ ở T6, đo tách lớp)
PEDS = ("human.pedestrian.adult", "human.pedestrian.child")
VN_SCENARIOS: Dict[str, VNScenario] = {s.name: s for s in [
    VNScenario("rc_control_car", "đối chứng ray-cast: ô tô đi cùng chiều",
               [Group("rc", ("car",), (1, 3), _traffic, (5, 12))]),
    VNScenario("rc_control_moto", "đối chứng ray-cast: xe máy 1 người đi cùng chiều",
               [Group("rc", ("moto_1",), (1, 3), _traffic, (4, 9))]),
    VNScenario("multi_rider", "xe máy chở 2-3 người trong dòng xe", [Group("rc", ("moto_multi",), (1, 3), _traffic, (5, 9))]),
    VNScenario("cargo_moto", "xe máy chở hàng cồng kềnh", [Group("rc", ("moto_cargo",), (1, 2), _ahead, (3, 7))]),
    VNScenario("ba_gac_slow", "xe ba gác chạy chậm sát lề", [Group("rc", ("ba_gac",), (1, 1), _curb_slow, (1.5, 4))]),
    VNScenario("tow_moto", "xe máy kéo thùng hàng", [Group("rc", ("moto_tow",), (1, 1), _ahead, (3, 6))]),
    VNScenario("vendor_curb", "xe đẩy hàng rong sát lề đường bên phải",
               [Group("rc", ("vendor_cart",), (1, 2), _vendor_right_curb, (0, 0.8))]),
    VNScenario("dog_dart", "chó lao ngang trước xe", [Group("rc", ("dog",), (1, 1), _dart, (3, 6))]),
    VNScenario("cow_road", "trâu / bò trên lòng đường", [Group("rc", ("cow",), (1, 1), _wander, (0, 0.8))]),
    VNScenario("ox_cart_road", "xe bò đi chậm trong làn",
               [Group("rc", ("cow",), (1, 1), _curb_slow, (0.8, 1.5), trailer="ox_wagon")]),
    VNScenario("rush_hour", "giờ cao điểm: bầy xe máy 1 người / 2-3 người / chở hàng + xe ba gác",
               [Group("bank", MOTO, (6, 10), _swarm_lane, (4, 9)), Group("rc", ("moto_multi",), (2, 4), _swarm_lane, (4, 9)),
                Group("rc", ("moto_cargo",), (1, 2), _swarm_lane, (3, 7)), Group("rc", ("ba_gac",), (1, 1), _curb_slow, (2, 4))]),
    VNScenario("market_street", "phố chợ: xe hàng rong + người đi bộ dưới lòng đường",
               [Group("rc", ("vendor_cart",), (1, 2), _vendor_right_curb, (0, 0.8)),
                Group("bank", ("human.pedestrian.adult",), (2, 4), _ped_road, (0.5, 1.5))]),
    VNScenario("ped_group_cross", "nhóm 3-8 người băng ngang cùng lúc",
               [Group("bank", PEDS, (3, 8), _group_cross, (0.8, 1.6), cluster=(1.5, 2.5))]),
    VNScenario("ped_crowd", "đám đông 10-20 người dưới lòng đường quanh xe",
               [Group("bank", PEDS, (10, 20), _crowd, (0.3, 1.5))]),
    VNScenario("animal_in_traffic", "chó lao ra giữa bầy xe máy",
               [Group("bank", MOTO, (6, 10), _swarm_lane, (4, 9)), Group("rc", ("dog",), (1, 1), _dart, (3, 6))]),
]}


# ----- Kịch bản tự chọn (web app): loại vật thể x số lượng x kiểu bố trí -----
# loại -> (nguồn, what, tốc độ m/s, trailer); nguồn bank = cắt mẫu nuScenes, rc = ray-cast
CUSTOM_TYPES: Dict[str, tuple] = {
    "car": ("bank", ("vehicle.car",), (5, 12), None),
    "truck": ("bank", ("vehicle.truck",), (4, 10), None),
    "bus": ("bank", ("vehicle.bus",), (4, 9), None),
    "moto": ("bank", MOTO, (4, 9), None),
    "bicycle": ("bank", ("vehicle.bicycle",), (2, 5), None),
    "pedestrian": ("bank", PEDS, (0.6, 1.6), None),
    "moto_multi": ("rc", ("moto_multi",), (4, 9), None),
    "moto_cargo": ("rc", ("moto_cargo",), (3, 7), None),
    "ba_gac": ("rc", ("ba_gac",), (1.5, 4), None),
    "moto_tow": ("rc", ("moto_tow",), (3, 6), None),
    "vendor_cart": ("rc", ("vendor_cart",), (0, 1), None),
    "dog": ("rc", ("dog",), (3, 6), None),
    "cow": ("rc", ("cow",), (0, 0.8), None),
    "ox_cart": ("rc", ("cow",), (0.8, 1.5), "ox_wagon"),
    "car_rc": ("rc", ("car",), (5, 12), None),
}
# kiểu bố trí -> (hàm tư thế ego frame, cụm (nửa dài, nửa rộng) hoặc None)
PLACEMENTS: Dict[str, tuple] = {
    "traffic": (_traffic, None),        # cùng chiều, trước / sau, làn ego / làn bên
    "ahead": (_ahead, None),            # phía trước trong / cạnh làn ego
    "swarm": (_swarm_lane, None),       # quanh xe trong 15 m, bám song song trục làn đường
    "curb": (_curb_slow, None),         # sát lề phía trước
    "roadside": (_curb, None),          # lấn từ lề xuống lòng đường, 2 chiều
    "crossing": (_dart, None),          # băng ngang trước xe
    "group_crossing": (_group_cross, (1.5, 2.5)),  # cả nhóm băng ngang cùng lúc
    "crowd": (_crowd, None),            # đi lại hướng bất kỳ quanh xe
    "wander": (_wander, None),          # đứng / đi chậm hướng bất kỳ trên lòng đường
}
MAX_CUSTOM_OBJECTS = 150


def build_custom(groups: Sequence[dict]) -> VNScenario:
    """groups: [{"type": loại trong CUSTOM_TYPES, "count": số lượng, "placement": kiểu trong PLACEMENTS}]."""
    out, total = [], 0
    for g in groups:
        if g["type"] not in CUSTOM_TYPES or g.get("placement", "traffic") not in PLACEMENTS:
            raise ValueError(f"loại / bố trí không hỗ trợ: {g}")
        n = int(g["count"])
        if n < 1:
            continue
        total += n
        src, what, speed, trailer = CUSTOM_TYPES[g["type"]]
        pose, cluster = PLACEMENTS[g.get("placement", "traffic")]
        if g["type"] == "vendor_cart":
            pose = _vendor_right_curb
        out.append(Group(src, what, (n, n), pose, speed, trailer=trailer, cluster=cluster, label=g["type"]))
    if not out or total > MAX_CUSTOM_OBJECTS:
        raise ValueError(f"cần 1-{MAX_CUSTOM_OBJECTS} object (đang {total})")
    desc = ", ".join(f"{g['count']} {g['type']} ({g.get('placement', 'traffic')})" for g in groups if int(g["count"]) > 0)
    return VNScenario("custom", "tự chọn: " + desc, out)


def run_vn_scenario(nusc, target_token: str, sc: VNScenario, bank: TemplateBank,
                    rng: Optional[np.random.Generator] = None, n_attempts: int = 300,
                    allow_template=None) -> dict:
    """Cùng schema output với scenarios.run_scenario (lưu bằng scenarios.save_scenario, kể cả sweep)."""
    rng = rng or np.random.default_rng(config.SEED)
    t0 = time.perf_counter()
    points = get_pointcloud(nusc, target_token)
    existing = get_annotations(nusc, target_token)
    sd_token = get_lidar_token(nusc, target_token)
    calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)
    ego = EgoFrame(calib)
    ground = GroundModel.from_points(points, rng=rng)
    area = get_map_area(nusc, load_map_for_sample(nusc, target_token), target_token, ("drivable_area",))
    scene = nusc.get("sample", target_token)["scene_token"]
    blockers = existing + [ego.ego_box()]
    new_anns, used, rejects, n_req, group_stats = [], {}, {}, 0, []

    def rej(k):
        rejects[k] = rejects.get(k, 0) + 1

    for g in sc.groups:
        n_target = int(rng.integers(g.count[0], g.count[1] + 1))
        n_req += n_target
        matcher = (TemplateMatcher(bank, g.what, 20, exclude_scenes=[scene], allow=allow_template)
                   if g.source == "bank" else None)
        placed = 0
        center = g.pose(rng) if g.cluster else None
        for _ in range(n_attempts):
            if placed >= n_target:
                break
            if center is not None:  # thành viên nhóm: quanh tâm nhóm, cùng hướng ±15°
                cx, cy, cyaw = center
                u, v = rng.uniform(-g.cluster[0], g.cluster[0]), rng.uniform(-g.cluster[1], g.cluster[1])
                ex, ey = cx + u * np.cos(cyaw) - v * np.sin(cyaw), cy + u * np.sin(cyaw) + v * np.cos(cyaw)
                eyaw = cyaw + D(rng.uniform(-15, 15))
            else:
                ex, ey, eyaw = g.pose(rng)
            x, y, yaw_req = ego.to_sensor(ex, ey, eyaw)
            if g.source == "bank":
                tpl = matcher.match(x, y, yaw_req, rng)
                if tpl is None:
                    rej("no_template")
                    continue
                # Giữ nguyên hướng yaw_req song song trục làn đường (tuyệt đối không xoay ngang 90 độ)
                yaw = float(wrap_angle(yaw_req))
                insert = insert_object
                min_pts = config.MIN_VISIBLE_POINTS
            else:
                tpl, yaw, insert, min_pts = build_model(g.what[0], rng), yaw_req, insert_raycast, 1
            parts = [(tpl, x, y)]
            if g.trailer:  # xe kéo phía sau, box riêng
                tr = build_model(g.trailer, rng)
                back = tpl["size"][1] / 2 + 0.4 + tr["size"][1] / 2
                parts.append((tr, x - back * np.cos(yaw), y - back * np.sin(yaw)))
            reason = None
            for p, px, py in parts:
                reason = reason or check_position(area, blockers + new_anns, p["size"], px, py, yaw, ground, points, min_ego_dist=9.0)
            if reason:
                rej(reason)
                continue
            aug, anns = points, []
            for p, px, py in parts:
                aug, info = insert(aug, p, (px, py, yaw), ground, rng)
                if info["n_visible"] < min_pts:
                    break
                z, _ = ground.height_at(px, py)
                anns.append((p, create_annotation(p, (px, py, yaw), float(z[0]), info["n_visible"])))
            if len(anns) < len(parts):
                rej("invisible" if g.source == "rc" else "occluded")
                continue
            points = aug
            speed = float(rng.uniform(*g.speed))
            for p, ann in anns:
                ann["velocity_sensor"] = [speed * np.cos(yaw), speed * np.sin(yaw)]
                new_anns.append(ann)
                used[p["ann_token"]] = p
            placed += 1
        group_stats.append({"what": g.label or g.what[0], "requested": n_target, "placed": placed})
    for ann in new_anns:  # object chèn sau có thể che object chèn trước
        ann["num_lidar_pts"] = int(points_in_new_box(points, ann).sum())

    anns_global = []
    for ann in new_anns:
        g = box_sensor_to_global(ann, calib, pose)
        dyaw = quaternion_to_yaw(g["rotation"]) - quaternion_to_yaw(ann["rotation"])
        vx, vy = ann["velocity_sensor"]
        g["velocity"] = [vx * np.cos(dyaw) - vy * np.sin(dyaw), vx * np.sin(dyaw) + vy * np.cos(dyaw)]
        anns_global.append(g)
    return {"points": points, "target_sample_token": target_token, "physical": True, "scenario": sc.name,
            "scenario_description": sc.description, "source_ann_tokens": [a["source_ann_token"] for a in new_anns],
            "n_requested": n_req, "n_inserted": len(new_anns), "elapsed_s": time.perf_counter() - t0,
            "reject_stats": rejects, "group_stats": group_stats, "annotations_sensor": new_anns,
            "annotations_global": anns_global, "quality": [], "existing_annotations": existing, "drivable": area,
            "templates": used}


if __name__ == "__main__":
    import argparse

    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="ba_gac_slow", choices=[*VN_SCENARIOS, "all"])
    ap.add_argument("--sample", type=int, default=10)
    args = ap.parse_args()
    nusc = load_nuscenes()
    bank = TemplateBank().load_or_build(nusc)
    tok = nusc.sample[args.sample]["token"]
    for name in (VN_SCENARIOS if args.scenario == "all" else [args.scenario]):
        r = run_vn_scenario(nusc, tok, VN_SCENARIOS[name], bank, np.random.default_rng(config.SEED))
        print(f"{name:18s} {r['n_inserted']}/{r['n_requested']} object, điểm "
              f"{[(a['category'][3:] if a['category'].startswith('vn.') else 'moto/ped', a['num_lidar_pts']) for a in r['annotations_sensor']][:6]}"
              f" reject {r['reject_stats']} ({r['elapsed_s']:.1f}s)")
