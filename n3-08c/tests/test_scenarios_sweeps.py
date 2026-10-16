"""Kiểm tra T2 (task_a/scenarios.py) và T4 (task_a/sweeps.py) trên data thật.

Chạy: python tests/test_scenarios_sweeps.py   (hoặc pytest tests/)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from task_a.extract_object import TemplateBank  # noqa: E402
from task_a.merge import points_in_new_box  # noqa: E402
from task_a.placement import bev_iou, box_footprint  # noqa: E402
from task_a.scenarios import SCENARIOS, EgoFrame, TemplateMatcher, VIEW_TOL, run_scenario  # noqa: E402
from task_a.sweeps import insert_into_sweeps, load_multisweep, sweep_tokens  # noqa: E402
from utils.nuscenes_loader import get_calibration, get_lidar_token  # noqa: E402
from utils.transforms import wrap_angle  # noqa: E402


def test_ego_frame_matches_calibration(nusc):
    """Điểm ego (x, 0) phải ra đúng vị trí khi biến đổi bằng calibration đầy đủ."""
    from utils.transforms import transform_matrix
    calib = get_calibration(nusc, get_lidar_token(nusc, nusc.sample[10]["token"]))
    ego = EgoFrame(calib)
    x, y, _ = ego.to_sensor(10.0, 2.0, 0.0)
    T = np.linalg.inv(transform_matrix(calib["translation"], calib["rotation"]))
    expect = T @ np.array([10.0, 2.0, 0.0, 1.0])
    assert np.allclose([x, y], expect[:2], atol=0.05)


def test_matcher_respects_distance_and_view(nusc):
    bank = TemplateBank().load_or_build(nusc)
    m = TemplateMatcher(bank, ("vehicle.car",), 20)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x, y, yaw = rng.uniform(-30, 30), rng.uniform(-30, 30), rng.uniform(-np.pi, np.pi)
        t = m.match(x, y, yaw, rng)
        if t is None:
            continue
        r = np.hypot(x, y)
        assert r / config.MAX_DISTANCE_SCALE <= t["distance"] <= r / 0.95
        assert abs(wrap_angle(t["view_angle"] - (np.arctan2(y, x) - yaw))) < VIEW_TOL


def test_scenarios_follow_spec(nusc):
    """Mọi kịch bản chèn được trên sample[10]; tư thế ego nằm trong miền kịch bản; không đè xe ego."""
    bank = TemplateBank().load_or_build(nusc)
    tok = nusc.sample[10]["token"]
    ego = EgoFrame(get_calibration(nusc, get_lidar_token(nusc, tok))).ego_box()
    for name, sc in SCENARIOS.items():
        r = run_scenario(nusc, tok, sc, bank, np.random.default_rng(config.SEED))
        assert r["n_inserted"] >= 1, name
        for a, q in zip(r["annotations_sensor"], r["quality"]):
            assert any(a["category"].startswith(c) for c in sc.categories)
            assert abs(q["yaw_adjust_deg"]) < np.degrees(VIEW_TOL) + 1e-6
            # người đi bộ thân hẹp: tâm cụm điểm ~ tâm box nên facing_cos nhiễu; mặt nhìn thấy đã do yaw_adjust đảm bảo
            if not a["category"].startswith("human."):
                assert q["facing_cos"] > -0.5, (name, q["facing_cos"])
            assert bev_iou(box_footprint(a), box_footprint(ego)) == 0
            assert a["num_lidar_pts"] >= config.MIN_VISIBLE_POINTS or name == "moto_swarm"  # che nhau trong bầy
            if a["category"].startswith(("vehicle.motorcycle", "vehicle.bicycle")):
                assert "cycle.with_rider" in r["templates"][a["source_ann_token"]]["attributes"]
        ex, ey, _ = r["quality"][0]["ego_pose"]
        if name == "cut_in":
            assert 3 <= ex <= 8 and 0.5 <= abs(ey) <= 2.5
        if name == "wrong_way":
            assert 8 <= ex <= 30 and abs(ey) <= 1.5


def _static_and_moving(nusc, tok, out_dir):
    from task_a.run_insertion import collect_templates, run_insertion, save_result
    rng = np.random.default_rng(config.SEED)
    tpls = collect_templates(nusc, nusc.sample[0]["token"], None, 2, rng=rng)
    r = run_insertion(nusc, tok, tpls, rng=rng)
    save_result(r, out_dir)
    anns = [dict(a) for a in r["annotations_global"]]
    anns[-1]["velocity"] = [5.0, 0.0]  # object cuối chạy 5 m/s theo trục x global
    sweeps = insert_into_sweeps(nusc, tok, anns, {t["ann_token"]: t for t in tpls}, out_dir, rng=rng)
    return r, anns, sweeps


def test_sweeps_insert_static_object_every_sweep(nusc):
    tok = nusc.sample[10]["token"]
    out_dir = tempfile.mkdtemp(prefix="test_sweeps_")
    r, _, sweeps = _static_and_moving(nusc, tok, out_dir)
    assert len(sweeps) == len(sweep_tokens(nusc, tok)) - 1 == config.N_SWEEPS - 1
    for s in sweeps:
        assert s["n_visible"][0] >= config.MIN_VISIBLE_POINTS  # object đứng yên có điểm ở mọi sweep
    ms = load_multisweep(nusc, tok, out_dir)
    assert ms.shape[1] == 5 and 0.4 < ms[:, 4].max() < 0.6
    # Object đứng yên: điểm của cả 10 sweep (đã đưa về keyframe frame) nằm trong box keyframe
    box = r["annotations_sensor"][0]
    in_box = points_in_new_box(ms, box)
    assert in_box.sum() >= 0.9 * (box["num_lidar_pts"] + sum(s["n_visible"][0] for s in sweeps))
    assert len(np.unique(ms[in_box, 4])) == config.N_SWEEPS


def test_sweeps_moving_object_trails(nusc):
    """Object chạy 5 m/s: ở sweep cách keyframe dt, tâm điểm object lùi 5 * dt m theo hướng chạy (global)."""
    from utils.nuscenes_loader import get_ego_pose, load_bin
    from utils.transforms import box_global_to_sensor, sensor_to_global
    tok = nusc.sample[10]["token"]
    out_dir = tempfile.mkdtemp(prefix="test_sweeps_")
    _, anns, sweeps = _static_and_moving(nusc, tok, out_dir)
    a = anns[-1]
    s = sweeps[-1]
    sd = s["sample_data_token"]
    calib, pose = get_calibration(nusc, sd), get_ego_pose(nusc, sd)
    moved = dict(a, translation=[a["translation"][0] - 5.0 * s["time_lag"], *a["translation"][1:]])
    pts = load_bin(f"{out_dir}/{s['file']}")
    m = points_in_new_box(pts, box_global_to_sensor(moved, calib, pose))
    assert m.sum() >= config.MIN_VISIBLE_POINTS
    center = sensor_to_global(pts[m, :3].astype(float), calib, pose).mean(0)
    assert np.linalg.norm(center[:2] - np.array(moved["translation"][:2])) < max(a["size"][:2])


def test_bicycle_insertion_scene_0103(nusc):
    """MỤC 3: Kiểm tra khi gọi sinh 10 bicycle trên scene-0103 phải chèn thành công >= 8 xe đạp."""
    from task_a.vn_scenarios import build_custom, run_vn_scenario
    scene_token = [s["token"] for s in nusc.scene if s["name"] == "scene-0103"][0]
    sample_token = nusc.get("scene", scene_token)["first_sample_token"]
    bank = TemplateBank().load()
    sc = build_custom([{"type": "bicycle", "count": 10, "placement": "traffic"}])
    res = run_vn_scenario(nusc, sample_token, sc, bank, rng=np.random.default_rng(42), n_attempts=1000)
    assert res["n_inserted"] >= 8, f"Mong đợi >= 8 xe đạp chèn thành công, nhưng chỉ chèn được {res['n_inserted']}/10"


if __name__ == "__main__":
    from utils.nuscenes_loader import load_nuscenes

    _nusc = load_nuscenes()
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(_nusc)
            print("PASS", name)
