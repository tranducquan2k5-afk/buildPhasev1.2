"""Engine của web app: chạy pipeline n3-08c (kịch bản VN, thời tiết, camera) cho 1 sample nuScenes.

Dùng nguyên code của repo (task_a/scenarios.py, task_a/vn_scenarios.py, task_b/*, utils/camera_proj.py), không
viết lại thuật toán. Detector chạy trực tiếp chỉ khi môi trường có torch + OpenPCDet + GPU (vd WSL env pcdet);
Docker / Windows không có -> dùng kết quả detector đã đo sẵn (output/cases, output/report).
"""
import base64
import io
import json
import os
import sys
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
N308C = ROOT / "n3-08c"
sys.path.insert(0, str(N308C))

try:  # torch phải load trước shapely (xem tools/detect.py); không có torch thì bỏ qua
    import torch  # noqa: F401
except ImportError:
    pass

import config  # noqa: E402

OUTPUT = Path(config.OUTPUT_DIR)
TASKS = ROOT / "webapp" / "temp_outputs"
TASKS.mkdir(parents=True, exist_ok=True)
PREVIEW_RADIUS = 60.0
MAX_PREVIEW = 60000
WEATHER_LEVELS = {"none": [], "rain": ["light", "moderate", "heavy"], "fog": ["light", "moderate", "dense"],
                  "storm": ["storm"]}


class Engine:
    def __init__(self):
        self._lock = threading.Lock()
        self._nusc = self._bank = self._det = None
        self.detector_error = None

    # ---------- tài nguyên dùng chung (nạp 1 lần) ----------
    @property
    def nusc(self):
        if self._nusc is None:
            from utils.nuscenes_loader import load_nuscenes
            self._nusc = load_nuscenes()
        return self._nusc

    @property
    def bank(self):
        if self._bank is None:
            from task_a.extract_object import TemplateBank
            self._bank = TemplateBank().load_or_build(self.nusc)
        return self._bank

    def detector(self):
        """Detector OpenPCDet nếu chạy được, ngược lại None (lý do ở self.detector_error)."""
        if self._det is None and self.detector_error is None:
            try:
                import torch
                if not torch.cuda.is_available():
                    raise RuntimeError("không có GPU CUDA")
                from tools.detect import Detector
                self._det = Detector()
            except Exception as e:  # Docker / Windows: không có torch / OpenPCDet / GPU
                self.detector_error = f"{type(e).__name__}: {e}"
        return self._det

    # ---------- danh mục ----------
    def meta(self) -> dict:
        from task_a.scenarios import SCENARIOS
        from task_a.vn_scenarios import VN_SCENARIOS
        from nuscenes.utils import splits
        val = set(splits.mini_val)
        samples = []
        for scene in self.nusc.scene:
            tok, k = scene["first_sample_token"], 0
            while tok:
                samples.append({"token": tok, "scene": scene["name"], "index": k,
                                "split": "mini_val" if scene["name"] in val else "mini_train",
                                "description": scene["description"]})
                tok, k = self.nusc.get("sample", tok)["next"], k + 1
        scen = [{"name": n, "group": "Kịch bản giao thông (xe cắt mẫu)", "description": s.description,
                 "count": list(s.count)} for n, s in SCENARIOS.items()]
        scen += [{"name": n, "group": "Vật thể Việt Nam (ray-cast) / kết hợp", "description": s.description,
                  "count": [sum(g.count[0] for g in s.groups), sum(g.count[1] for g in s.groups)]}
                 for n, s in VN_SCENARIOS.items()]
        from task_a.vn_scenarios import CUSTOM_TYPES, MAX_CUSTOM_OBJECTS, PLACEMENTS
        return {"samples": samples, "scenarios": scen, "weather": WEATHER_LEVELS,
                "custom_types": {k: {"source": v[0]} for k, v in CUSTOM_TYPES.items()},
                "placements": list(PLACEMENTS), "max_custom_objects": MAX_CUSTOM_OBJECTS,
                "detector": self.detector() is not None, "detector_error": self.detector_error}

    # ---------- sinh ----------
    def generate(self, sample_token: str, scenario: str, seed: int = 42, weather: str = "none",
                 level: Optional[str] = None, puddles: int = 0, sweeps: bool = False,
                 custom: Optional[List[dict]] = None) -> dict:
        """scenario = "custom": dùng custom = [{type, count, placement}] (task_a/vn_scenarios.build_custom)."""
        from task_a.scenarios import SCENARIOS, run_scenario, save_scenario
        from task_a.vn_scenarios import VN_SCENARIOS, build_custom, run_vn_scenario
        from tools.detect import detection_class
        from utils.nuscenes_loader import get_pointcloud
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        with self._lock:
            if scenario == "custom":
                r = run_vn_scenario(self.nusc, sample_token, build_custom(custom or []), self.bank, rng, 2000)
            elif scenario in VN_SCENARIOS:
                r = run_vn_scenario(self.nusc, sample_token, VN_SCENARIOS[scenario], self.bank, rng, 1000)
            elif scenario in SCENARIOS:
                r = run_scenario(self.nusc, sample_token, SCENARIOS[scenario], self.bank, rng, 1000)
            else:
                raise ValueError(f"không có kịch bản {scenario}")
        task_id = uuid.uuid4().hex[:12]
        tdir = TASKS / task_id
        tdir.mkdir(parents=True)
        orig = get_pointcloud(self.nusc, sample_token)
        aug = r["points"]
        if sweeps and r["n_inserted"]:  # 9 sweep trước (cần cho detector), lưu theo format tools/t6_eval
            save_scenario(self.nusc, dict(r), str(tdir), sweeps=True, rng=rng)
        else:
            r_save = {k: v for k, v in r.items() if k != "templates"}
            from task_a.run_insertion import save_result
            save_result(r_save, str(tdir))
        if weather != "none":
            from task_b.run_weather import apply_weather
            lvl = level or WEATHER_LEVELS[weather][-1]
            vehicles = [a for a in r["existing_annotations"] + r["annotations_sensor"]
                        if a["category"].startswith("vehicle") or a["category"].startswith("vn.")]
            aug = apply_weather(aug, weather, lvl, rng, vehicles)[0] if weather == "storm" else \
                apply_weather(aug, weather, lvl, rng)[0]
            if weather == "storm" and puddles:
                from task_b.storm import puddle_mask
                aug = aug[puddle_mask(aug, rng, puddles)]
        # file tải về đa định dạng (chuẩn hóa qua utils.io_converter)
        from utils.io_converter import save_point_cloud
        save_point_cloud(aug, str(tdir / "augmented.bin"), format=".bin")
        save_point_cloud(aug, str(tdir / "augmented.pcd"), format=".pcd")
        save_point_cloud(aug, str(tdir / "augmented.ply"), format=".ply")
        save_point_cloud(aug, str(tdir / "augmented.xyz"), format=".xyz")
        labels = [{"category": a["category"], "detection_class": detection_class(a["category"]),
                   "translation": a["translation"], "size": a["size"], "rotation": a["rotation"],
                   "velocity_sensor": a.get("velocity_sensor"), "num_lidar_pts": a["num_lidar_pts"],
                   "difficult": a["num_lidar_pts"] < 5, "synthetic": True} for a in r["annotations_sensor"]]
        info = {"task_id": task_id, "sample_token": sample_token, "scenario": scenario, "seed": seed,
                "weather": weather, "level": level, "puddles": puddles, "frame": "LIDAR_TOP sensor (nuScenes)",
                "point_format": "float32 [x, y, z, intensity, ring]", "labels": labels,
                "reject_stats": r["reject_stats"]}
        (tdir / "labels.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        with zipfile.ZipFile(tdir / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("augmented.bin", "augmented.pcd", "augmented.ply", "augmented.xyz", "labels.json"):
                if (tdir / f).is_file():
                    z.write(tdir / f, f)
        state = {"sample_token": sample_token, "scenario": scenario, "weather": weather, "level": level,
                 "annotations": r["annotations_sensor"], "existing": r["existing_annotations"], "sweeps": sweeps}
        (tdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        return {"task_id": task_id, "preview": self._preview(orig, aug, r["annotations_sensor"], r["existing_annotations"]),
                "stats": {"scenario": scenario, "description": r.get("scenario_description", ""),
                          "requested": r["n_requested"], "inserted": r["n_inserted"],
                          "difficult": int(sum(l["difficult"] for l in labels)), "reject_stats": r["reject_stats"],
                          "points_before": int(len(orig)), "points_after": int(len(aug)),
                          "elapsed_ms": round(1000 * (time.perf_counter() - t0)), "weather": weather, "level": level,
                          "group_stats": r.get("group_stats", [])},
                "downloads": {k: f"/api/download/{task_id}/{k}" for k in ("pcd", "bin", "ply", "xyz", "labels", "zip")}}

    def _preview(self, orig, aug, inserted, existing, statuses=None, preds=None) -> dict:
        from task_a.merge import points_in_new_box
        rng = np.random.default_rng(0)

        def pack(p, mark_obj):
            p = p[np.hypot(p[:, 0], p[:, 1]) < PREVIEW_RADIUS]
            obj = np.zeros(len(p), bool)
            if mark_obj:
                for a in inserted:
                    obj |= points_in_new_box(p, a)
            if len(p) > MAX_PREVIEW:
                keep = np.r_[np.flatnonzero(obj), rng.choice(np.flatnonzero(~obj), MAX_PREVIEW - obj.sum(), False)]
                p, obj = p[keep], obj[keep]
            return np.round(p[:, :4], 2).ravel().tolist(), obj.astype(int).tolist()
        o_pts, _ = pack(orig, False)
        a_pts, a_obj = pack(aug, True)
        statuses = statuses or {}
        boxes = [{"translation": a["translation"], "size": a["size"], "rotation": a["rotation"],
                  "category": a["category"], "num_lidar_pts": a["num_lidar_pts"], "inserted": True,
                  "status": statuses.get(i, "inserted")} for i, a in enumerate(inserted)]
        boxes += [{"translation": a["translation"], "size": a["size"], "rotation": a["rotation"],
                   "category": a["category"], "inserted": False, "status": "real"} for a in existing
                  if np.hypot(*a["translation"][:2]) < PREVIEW_RADIUS]
        return {"original": o_pts, "augmented": a_pts, "obj": a_obj, "boxes": boxes, "preds": preds or []}

    # ---------- camera ----------
    def cameras(self, task_id: str, statuses: Optional[Dict[int, str]] = None, mode: str = "photorealistic_box") -> dict:
        from task_a.merge import points_in_new_box
        from utils.camera_proj import SURROUND, surround
        from utils.nuscenes_loader import load_bin
        st = json.loads((TASKS / task_id / "state.json").read_text(encoding="utf-8"))
        aug = load_bin(str(TASKS / task_id / f"{st['sample_token']}.bin"))
        obj = np.zeros(len(aug), bool)
        for a in st["annotations"]:
            obj |= points_in_new_box(aug, a)
        statuses = statuses or st.get("statuses", {})
        boxes = [dict(a, inserted=True, status=statuses.get(str(i), statuses.get(i, "inserted")))
                 for i, a in enumerate(st["annotations"])]
        boxes += [dict(a, inserted=False, status="real") for a in st["existing"]]
        lvl = st["level"] or (WEATHER_LEVELS.get(st["weather"]) or [None])[-1]
        res = surround(self.nusc, st["sample_token"], boxes, aug[obj],
                       weather=None if st["weather"] == "none" else st["weather"], level=lvl, mode=mode)
        out = []
        for cam in SURROUND:
            img, n = res[cam]
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=82)
            out.append({"channel": cam, "objects": n,
                        "image": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()})
        return {"cameras": out, "weather": st["weather"], "mode": mode}

    # ---------- detector trực tiếp ----------
    def detect(self, task_id: str) -> dict:
        det = self.detector()
        if det is None:
            return {"available": False, "reason": self.detector_error}
        from task_a.sweeps import load_multisweep
        from tools.t6_eval import inserted_gts, object_records, real_gts
        tdir = TASKS / task_id
        st = json.loads((tdir / "state.json").read_text(encoding="utf-8"))
        if not st["sweeps"]:
            return {"available": False, "reason": "cần sinh kèm 9 sweep (bật 'Sinh 10 sweep') để chạy detector"}
        tok = st["sample_token"]
        with open(tdir / f"{tok}.json", encoding="utf-8") as f:
            result = json.load(f)
        real = real_gts(self.nusc, tok)
        gts = real + inserted_gts(result)
        with self._lock:
            pred = det.predict(load_multisweep(self.nusc, tok, str(tdir)))
        recs = object_records(gts, pred, "web", tok, None)[len(real):]
        # map record -> chỉ số annotation chèn (inserted_gts bỏ object 0 điểm / ngoài 50 m)
        ins_ids = [i for i, a in enumerate(result["annotations_sensor"])
                   if a["num_lidar_pts"] > 0 and np.hypot(*a["translation"][:2]) < 50]
        statuses = {i: ("ok" if r["detected"] else r["miss"].split(":")[0]) for i, r in zip(ins_ids, recs)}
        confused_as = {i: r["miss"].split(":")[1] for i, r in zip(ins_ids, recs) if r["miss"] and ":" in r["miss"]}
        st["statuses"] = statuses
        (tdir / "state.json").write_text(json.dumps(st), encoding="utf-8")
        keep = pred["scores"] >= 0.3
        preds = [{"c": [float(v) for v in b[:3]], "s": [float(b[4]), float(b[3]), float(b[5])], "yaw": float(b[6]),
                  "label": str(l), "score": float(s)} for b, l, s in
                 zip(pred["boxes"][keep], pred["labels"][keep], pred["scores"][keep])]
        n_ok = sum(v == "ok" for v in statuses.values())
        return {"available": True, "statuses": statuses, "confused_as": confused_as, "preds": preds,
                "summary": {"detected": n_ok, "total": len(statuses),
                            "confused": sum(v == "confused" for v in statuses.values()),
                            "missed": sum(v in ("missed", "merged") for v in statuses.values())}}

    # ---------- kết quả đo sẵn ----------
    def results(self) -> dict:
        def load(p):
            return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
        cases = load(OUTPUT / "cases" / "3d" / "cases.json") or {}
        return {"summary": load(OUTPUT / "report" / "summary.json") or {},
                "weather": load(OUTPUT / "weather_eval" / "report.json"),
                "weather_swarm": load(OUTPUT / "weather_eval" / "report_swarm.json"),
                "cases": cases,
                "charts": sorted(p.name for p in (OUTPUT / "report").glob("*.png")) if (OUTPUT / "report").is_dir() else [],
                "videos": sorted(p.name for p in (OUTPUT / "videos").glob("*.mp4")) if (OUTPUT / "videos").is_dir() else []}

    def case(self, name: str) -> dict:
        p = OUTPUT / "cases" / "3d" / f"{name}.json"
        if not p.is_file():
            raise FileNotFoundError(name)
        return json.loads(p.read_text(encoding="utf-8"))

    # ---------- upload point cloud tùy chỉnh ----------
    def upload_point_cloud_data(self, file_bytes: bytes, filename: str) -> dict:
        from utils.io_converter import load_point_cloud, save_point_cloud
        ext = Path(filename).suffix.lower()
        pts = load_point_cloud(file_bytes, file_ext=ext)
        task_id = uuid.uuid4().hex[:12]
        tdir = TASKS / task_id
        tdir.mkdir(parents=True, exist_ok=True)
        save_point_cloud(pts, str(tdir / "augmented.bin"), format=".bin")
        save_point_cloud(pts, str(tdir / "augmented.pcd"), format=".pcd")
        save_point_cloud(pts, str(tdir / "augmented.ply"), format=".ply")
        save_point_cloud(pts, str(tdir / "augmented.xyz"), format=".xyz")
        preview_pts = pts[np.hypot(pts[:, 0], pts[:, 1]) < PREVIEW_RADIUS]
        if len(preview_pts) > MAX_PREVIEW:
            rng = np.random.default_rng(0)
            idx = rng.choice(len(preview_pts), MAX_PREVIEW, replace=False)
            preview_pts = preview_pts[idx]
        p_flat = np.round(preview_pts[:, :4], 2).ravel().tolist()
        state = {"sample_token": "custom_upload", "filename": filename, "points": len(pts),
                 "annotations": [], "existing": []}
        (tdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
        with zipfile.ZipFile(tdir / "bundle.zip", "w", zipfile.ZIP_DEFLATED) as z:
            for f in ("augmented.bin", "augmented.pcd", "augmented.ply", "augmented.xyz"):
                if (tdir / f).is_file():
                    z.write(tdir / f, f)
        return {
            "task_id": task_id,
            "preview": {"original": p_flat, "augmented": p_flat, "obj": [0] * len(preview_pts), "boxes": [], "preds": []},
            "stats": {"scenario": "custom_upload", "description": f"Tệp tải lên: {filename}",
                      "requested": 0, "inserted": 0, "difficult": 0, "reject_stats": {},
                      "points_before": len(pts), "points_after": len(pts), "elapsed_ms": 15,
                      "weather": "none", "level": None, "group_stats": []},
            "downloads": {k: f"/api/download/{task_id}/{k}" for k in ("pcd", "bin", "ply", "xyz", "zip")}
        }

