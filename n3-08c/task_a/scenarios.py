"""T2: kịch bản giao thông Việt Nam (từng khung) bằng object insertion có chủ đích.

Mỗi kịch bản sinh tư thế object trong EGO frame (x hướng đầu xe ego, y sang trái), lấy hướng xe ego làm
hướng làn (xe ego đi đúng làn). Không cần lane centerline: các kịch bản đều đo so với xe ego.

    cut_in      1 xe 2 bánh có người lái cắt đầu: phía trước 3-8 m, lệch ngang 0.5-2.5 m, hướng chéo 20-70°
                vào làn ego
    wrong_way   1 xe (ô tô / xe máy) ngược chiều trong làn ego, phía trước 8-30 m
    moto_swarm  10-20 xe 2 bánh có người lái quanh ego (bán kính 15 m), cùng chiều ±15°
    jaywalk     1 người đi bộ băng ngang trên lòng đường, phía trước 5-20 m, ngoài vạch sang đường
    lane_block  1 ô tô / xe tải đỗ lấn làn ego, phía trước 8-30 m
    control_*   đối chứng cho T6: cùng loại object, tư thế giao thông bình thường (cùng chiều / trên vỉa hè)

Vật lý giữ như task_a/placement.sample_position_for_template: object chỉ được đặt sao cho LiDAR thấy cùng mặt
như lúc cắt template. Với tư thế mong muốn (r, phi, yaw), chọn template trong bank có
    distance trong [r / MAX_DISTANCE_SCALE, r / (1 - R_JITTER)]   (mật độ: chỉ dời ra xa)
    |view_angle - (phi - yaw)| < VIEW_TOL                         (mặt nhìn thấy)
rồi chỉnh yaw = phi - view_angle (lệch < VIEW_TOL so với tư thế kịch bản).
Hợp lệ: placement.check_position (trong vùng map, thấy mặt đất, không va chạm kể cả xe ego, không vật cản),
còn >= MIN_VISIBLE_POINTS điểm sau che khuất.

Object có vận tốc (global, m/s) theo kịch bản; dùng cho T4 (sweep) và ghi vào annotation như nuScenes.

Chạy:
    python task_a/scenarios.py --scenario cut_in                      # sample[10]
    python task_a/scenarios.py --scenario moto_swarm --split mini_val --sweeps --vis
    python task_a/scenarios.py --scenario all --split mini_val --sweeps     # tập kịch bản cho T6
Output: <OUTPUT_DIR>/scenarios/<scenario>/<sample_token>.bin/.json (+ sweeps/ nếu --sweeps),
        ảnh BEV <OUTPUT_DIR>/vis/scenario_<scenario>_<token>.png nếu --vis.
"""
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from shapely.geometry.base import BaseGeometry

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.extract_object import TemplateBank  # noqa: E402
from task_a.ground_plane import GroundModel, get_map_area, load_map_for_sample  # noqa: E402
from task_a.merge import create_annotation, insert_object, points_in_new_box  # noqa: E402
from task_a.placement import check_position  # noqa: E402
from task_a.run_insertion import facing_cos, save_result  # noqa: E402
from utils.nuscenes_loader import (get_annotations, get_calibration, get_ego_pose, get_lidar_token,  # noqa: E402
                                   get_pointcloud)
from utils.transforms import box_sensor_to_global, quaternion_to_yaw, wrap_angle, yaw_to_quaternion  # noqa: E402

VIEW_TOL = np.radians(15.0)  # lệch tối đa giữa góc nhìn của template và góc nhìn kịch bản yêu cầu
R_JITTER = 0.05              # như placement: cho kéo lại gần tối đa 5%
EGO_SIZE = (1.73, 4.08, 1.56)  # Renault Zoe của nuScenes (w, l, h)
EGO_CENTER_X = 1.3             # tâm xe ego trong ego frame (gốc ego = trục sau)
TWO_WHEEL = ("vehicle.motorcycle", "vehicle.bicycle")

EgoPose = Tuple[float, float, float]  # (x, y, yaw) trong ego frame


@dataclass
class Scenario:
    name: str
    description: str
    categories: Tuple[str, ...]                 # prefix category template
    count: Tuple[int, int]                      # số object [min, max]
    sample_pose: Callable[[np.random.Generator], EgoPose]
    speed: Tuple[float, float]                  # m/s theo hướng yaw của object
    layers: Tuple[str, ...] = ("drivable_area",)
    exclude_layers: Tuple[str, ...] = ()        # footprint không được chạm các layer này
    min_points: int = 20                        # số điểm tối thiểu của template


def _side(rng) -> int:
    return 1 if rng.random() < 0.5 else -1


def _cut_in(rng):
    s = _side(rng)  # bên trái (y > 0) thì chéo sang phải (yaw < 0) và ngược lại
    return rng.uniform(3, 8), s * rng.uniform(0.5, 2.5), -s * np.radians(rng.uniform(20, 70))


def _wrong_way(rng):
    return rng.uniform(8, 30), rng.uniform(-1.5, 1.5), np.pi + np.radians(rng.uniform(-15, 15))


def _swarm(rng):
    while True:
        x, y = rng.uniform(-10, 15), rng.uniform(-6, 6)
        if np.hypot(x - EGO_CENTER_X, y) <= 15:
            return x, y, np.radians(rng.uniform(-15, 15))


def _jaywalk(rng):
    return rng.uniform(5, 20), rng.uniform(-4, 4), _side(rng) * np.radians(rng.uniform(60, 120))


def _lane_block(rng):
    return rng.uniform(8, 30), _side(rng) * rng.uniform(1.0, 2.2), np.radians(rng.uniform(-10, 10))


def _normal_traffic(rng):
    """Đối chứng: xe đi cùng chiều ở làn ego / làn bên, trước hoặc sau 4-35 m (gồm cả cự ly gần như cut_in)."""
    return _side(rng) * rng.uniform(4, 35), rng.uniform(-5, 5), np.radians(rng.uniform(-10, 10))


def _normal_walk(rng):
    """Đối chứng: người đi dọc vỉa hè, hướng song song đường."""
    yaw = (0.0 if rng.random() < 0.5 else np.pi) + np.radians(rng.uniform(-15, 15))
    return _side(rng) * rng.uniform(5, 25), _side(rng) * rng.uniform(3, 10), yaw


SCENARIOS: Dict[str, Scenario] = {s.name: s for s in [
    Scenario("cut_in", "xe máy cắt đầu 3-8 m", TWO_WHEEL, (1, 1), _cut_in, (5, 10)),
    Scenario("wrong_way", "xe ngược chiều trong làn ego", ("vehicle.car", "vehicle.motorcycle"), (1, 1),
             _wrong_way, (5, 12)),
    Scenario("moto_swarm", "bầy 10-20 xe 2 bánh quanh ego", TWO_WHEEL, (10, 20), _swarm, (4, 9)),
    Scenario("jaywalk", "người đi bộ băng ngang lòng đường", ("human.pedestrian.adult", "human.pedestrian.child"),
             (1, 1), _jaywalk, (0.8, 1.8), exclude_layers=("ped_crossing",)),
    Scenario("lane_block", "ô tô / xe tải đỗ lấn làn", ("vehicle.car", "vehicle.truck"), (1, 1),
             _lane_block, (0, 0)),
    # Đối chứng cho T6 (nhóm B): cùng template, cùng pipeline chèn, nhưng tư thế giao thông bình thường.
    # Recall đối chứng ~ recall object thật -> object chèn không "giả"; kịch bản VN thấp hơn đối chứng -> do tình huống.
    Scenario("control_2w", "đối chứng: xe 2 bánh đi cùng chiều", TWO_WHEEL, (1, 3), _normal_traffic, (4, 9)),
    Scenario("control_car", "đối chứng: ô tô / xe tải đi cùng chiều", ("vehicle.car", "vehicle.truck"), (1, 3),
             _normal_traffic, (5, 12)),
    Scenario("control_ped", "đối chứng: người đi bộ trên vỉa hè",
             ("human.pedestrian.adult", "human.pedestrian.child"), (1, 3), _normal_walk, (0.8, 1.8),
             layers=("walkway",)),
]}


class EgoFrame:
    """Chuyển tư thế ego frame <-> LIDAR_TOP sensor frame của 1 sample_data (calibration sensor -> ego)."""

    def __init__(self, calibration: dict):
        from pyquaternion import Quaternion
        q = Quaternion(calibration["rotation"])
        self.R = q.rotation_matrix[:2, :2]          # sensor -> ego (xoay quanh z, LIDAR_TOP nằm ngang)
        self.t = np.asarray(calibration["translation"][:2])
        self.yaw = float(np.arctan2(self.R[1, 0], self.R[0, 0]))

    def to_sensor(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        xs, ys = self.R.T @ (np.array([x, y]) - self.t)
        return float(xs), float(ys), float(wrap_angle(yaw - self.yaw))

    def ego_box(self) -> dict:
        x, y, yaw = self.to_sensor(EGO_CENTER_X, 0.0, 0.0)
        return {"translation": [x, y, 0.0], "size": list(EGO_SIZE), "rotation": yaw_to_quaternion(yaw),
                "category": "ego"}


class TemplateMatcher:
    """Tìm template trong bank có khoảng cách + góc nhìn tương thích với tư thế yêu cầu (vector hoá)."""

    def __init__(self, bank: TemplateBank, categories: Sequence[str], min_points: int,
                 exclude_scenes: Sequence[str] = (), allow: Optional[Callable[[dict], bool]] = None):
        self.bank = bank
        self.categories = list(categories)
        self.is_bicycle = (len(self.categories) == 1 and "bicycle" in self.categories[0])
        if self.is_bicycle:
            # MỤC 3: Đối với class bicycle (và Xe đạp thật): chấp nhận num_lidar_pts >= 10
            min_points = min(min_points, 10)
            req_attrs = {k: v for k, v in config.TEMPLATE_REQUIRED_ATTRIBUTES.items() if "bicycle" not in k}
            self.meta = bank.filter(categories=list(categories), min_points=min_points, exclude_scenes=exclude_scenes,
                                    required_attributes=req_attrs)
            if len(self.meta) < 3:
                # Nới lỏng scene exclusion nếu số mẫu trong bank ít
                self.meta = bank.filter(categories=list(categories), min_points=min_points)
        else:
            self.meta = bank.filter(categories=list(categories), min_points=min_points, exclude_scenes=exclude_scenes,
                                    required_attributes=config.TEMPLATE_REQUIRED_ATTRIBUTES)
        if allow is not None:  # vd T7: tách template train / test theo instance
            self.meta = [m for m in self.meta if allow(m)]
        self.dist = np.array([m["distance"] for m in self.meta])
        self.view = np.array([m["view_angle"] for m in self.meta])

    def __len__(self) -> int:
        return len(self.meta)

    def match(self, x: float, y: float, yaw: float, rng: np.random.Generator) -> Optional[dict]:
        if not len(self.meta):
            return None
        r, phi = np.hypot(x, y), np.arctan2(y, x)
        if self.is_bicycle:
            # MỤC 3: Nới lỏng khoảng cách khớp template (d_tol = 25.0m hoặc cho phép lấy bất kỳ mẫu bicycle nào trong bank.json)
            dist_diff = np.abs(self.dist - r)
            ok = (dist_diff <= 25.0)
            idx = np.flatnonzero(ok)
            if len(idx) == 0:
                idx = np.arange(len(self.meta))
            return self.bank.get(self.meta[rng.choice(idx)])
        else:
            ok = ((self.dist >= r / config.MAX_DISTANCE_SCALE) & (self.dist <= r / (1 - R_JITTER))
                  & (np.abs(wrap_angle(self.view - (phi - yaw))) < VIEW_TOL))
            idx = np.flatnonzero(ok)
            return self.bank.get(self.meta[rng.choice(idx)]) if len(idx) else None


def scenario_area(nusc, nusc_map, sample_token: str, sc: Scenario) -> BaseGeometry:
    area = get_map_area(nusc, nusc_map, sample_token, sc.layers)
    if sc.exclude_layers:
        area = area.difference(get_map_area(nusc, nusc_map, sample_token, sc.exclude_layers))
    return area


def run_scenario(nusc, target_token: str, sc: Scenario, bank: TemplateBank,
                 rng: Optional[np.random.Generator] = None, n_attempts: int = 300,
                 allow_template: Optional[Callable[[dict], bool]] = None) -> dict:
    """Chèn object theo kịch bản sc vào target sample. Cùng schema output với run_insertion.run_insertion,
    thêm "scenario", "templates" (để chèn sweep) và "velocity" trong annotation.

    allow_template: lọc thêm template theo metadata bank (None = mọi template hợp lệ).
    """
    rng = rng or np.random.default_rng(config.SEED)
    t0 = time.perf_counter()
    points = get_pointcloud(nusc, target_token)
    existing = get_annotations(nusc, target_token)
    sd_token = get_lidar_token(nusc, target_token)
    calib, pose = get_calibration(nusc, sd_token), get_ego_pose(nusc, sd_token)
    ego = EgoFrame(calib)
    ground = GroundModel.from_points(points, rng=rng)
    nusc_map = load_map_for_sample(nusc, target_token)
    area = scenario_area(nusc, nusc_map, target_token, sc)
    matcher = TemplateMatcher(bank, sc.categories, sc.min_points,
                              exclude_scenes=[nusc.get("sample", target_token)["scene_token"]], allow=allow_template)

    n_target = int(rng.integers(sc.count[0], sc.count[1] + 1))
    blockers = existing + [ego.ego_box()]
    new_anns, quality, used, rejects = [], [], {}, {}
    for _ in range(n_attempts):
        if len(new_anns) >= n_target:
            break
        ex, ey, eyaw = sc.sample_pose(rng)
        x, y, yaw_req = ego.to_sensor(ex, ey, eyaw)
        tpl = matcher.match(x, y, yaw_req, rng)
        if tpl is None:
            rejects["no_template"] = rejects.get("no_template", 0) + 1
            continue
        yaw = float(wrap_angle(np.arctan2(y, x) - tpl["view_angle"]))
        reason = check_position(area, blockers + new_anns, tpl["size"], x, y, yaw, ground, points)
        if reason:
            rejects[reason] = rejects.get(reason, 0) + 1
            continue
        aug, info = insert_object(points, tpl, (x, y, yaw), ground, rng)
        if info["n_visible"] < config.MIN_VISIBLE_POINTS:
            rejects["occluded"] = rejects.get("occluded", 0) + 1
            continue
        points = aug
        z, support = ground.height_at(x, y)
        ann = create_annotation(tpl, (x, y, yaw), float(z[0]), info["n_visible"])
        speed = float(rng.uniform(*sc.speed))
        ann["velocity_sensor"] = [speed * np.cos(yaw), speed * np.sin(yaw)]
        new_anns.append(ann)
        used[tpl["ann_token"]] = tpl
        quality.append({"source_distance": tpl["distance"], "distance": float(np.hypot(x, y)),
                        "ground_support": int(support[0]),
                        "facing_cos": facing_cos(points[len(points) - info["n_visible"]:], ann),
                        "yaw_adjust_deg": float(np.degrees(wrap_angle(yaw - yaw_req))),
                        "ego_pose": [ex, ey, float(eyaw)], **info})
    for ann, q in zip(new_anns, quality):  # object chèn sau có thể che object chèn trước
        ann["num_lidar_pts"] = q["n_visible_final"] = int(points_in_new_box(points, ann).sum())

    anns_global = []
    for ann in new_anns:
        g = box_sensor_to_global(ann, calib, pose)
        # vận tốc: xoay theo hướng sensor -> global (chỉ quanh z)
        dyaw = quaternion_to_yaw(g["rotation"]) - quaternion_to_yaw(ann["rotation"])
        vx, vy = ann["velocity_sensor"]
        g["velocity"] = [vx * np.cos(dyaw) - vy * np.sin(dyaw), vx * np.sin(dyaw) + vy * np.cos(dyaw)]
        anns_global.append(g)
    return {
        "points": points, "target_sample_token": target_token, "physical": True,
        "scenario": sc.name, "scenario_description": sc.description,
        "source_ann_tokens": [a["source_ann_token"] for a in new_anns],
        "n_requested": n_target, "n_inserted": len(new_anns), "elapsed_s": time.perf_counter() - t0,
        "reject_stats": rejects, "n_matchable_templates": len(matcher),
        "annotations_sensor": new_anns, "annotations_global": anns_global, "quality": quality,
        "existing_annotations": existing, "drivable": area, "templates": used,
    }


def split_samples(nusc, split: str) -> List[str]:
    """Sample token của split: all | mini_train | mini_val (scene theo nuscenes.utils.splits)."""
    if split == "all":
        return [s["token"] for s in nusc.sample]
    from nuscenes.utils import splits
    names = set(getattr(splits, split))
    scenes = {s["token"] for s in nusc.scene if s["name"] in names}
    return [s["token"] for s in nusc.sample if s["scene_token"] in scenes]


def save_scenario(nusc, result: dict, out_dir: str, sweeps: bool = False,
                  rng: Optional[np.random.Generator] = None) -> str:
    """Lưu keyframe (+ sweep nếu sweeps=True, T4) theo format run_insertion.save_result."""
    templates = result.pop("templates")
    if sweeps and result["n_inserted"]:
        from task_a.sweeps import insert_into_sweeps
        result["sweeps"] = insert_into_sweeps(nusc, result["target_sample_token"], result["annotations_global"],
                                              templates, out_dir, rng=rng)
    return save_result(result, out_dir)


if __name__ == "__main__":
    import argparse

    from tqdm import tqdm

    from utils.nuscenes_loader import load_nuscenes

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="cut_in", choices=[*SCENARIOS, "all"])
    ap.add_argument("--target", help="sample token (mặc định sample[10]); bỏ qua nếu có --split")
    ap.add_argument("--split", choices=["all", "mini_train", "mini_val"], help="chạy trên mọi sample của split")
    ap.add_argument("--sweeps", action="store_true", help="chèn cả 9 sweep trước keyframe (T4)")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--out-dir", default=os.path.join(config.OUTPUT_DIR, "scenarios"))
    ap.add_argument("--vis", action="store_true", help="lưu ảnh BEV")
    args = ap.parse_args()

    nusc = load_nuscenes()
    bank = TemplateBank().load_or_build(nusc)
    rng = np.random.default_rng(args.seed)
    targets = split_samples(nusc, args.split) if args.split else [args.target or nusc.sample[10]["token"]]
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]

    for name in names:
        sc, out_dir, summary = SCENARIOS[name], os.path.join(args.out_dir, name), []
        for tok in tqdm(targets, desc=name, disable=len(targets) == 1):
            r = run_scenario(nusc, tok, sc, bank, rng)
            summary.append((r["n_inserted"], r["n_requested"], r["reject_stats"]))
            if args.vis:
                from utils.visualization import plot_bev
                plot_bev(r["points"], r["existing_annotations"], r["annotations_sensor"], region=r["drivable"],
                         xlim=35, title=f"{name}: {sc.description} ({r['n_inserted']}/{r['n_requested']})",
                         save_path=os.path.join(config.VIS_DIR, f"scenario_{name}_{tok}.png"))
            if r["n_inserted"]:
                save_scenario(nusc, r, out_dir, args.sweeps, rng)
        ok = sum(n > 0 for n, _, _ in summary)
        rej: Dict[str, int] = {}
        for _, _, s in summary:
            for k, v in s.items():
                rej[k] = rej.get(k, 0) + v
        print(f"{name:11s} {ok}/{len(summary)} sample có object; "
              f"object {sum(n for n, _, _ in summary)}/{sum(m for _, m, _ in summary)}; reject {rej} -> {out_dir}")
