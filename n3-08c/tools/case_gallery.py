"""Ảnh kết quả từng trường hợp: mỗi kịch bản 1 khung, detector pretrained chạy trên dữ liệu đã chèn.

Box GT: xanh lá = phát hiện đúng, đỏ = bỏ sót, cam = nhận nhầm lớp (ghi lớp bị nhận thành); nét đậm = object chèn,
nét mảnh = object thật; nét đứt xanh dương = prediction (score >= 0.3); điểm tím = điểm LiDAR của object chèn.
Khung chọn cố định (không chọn khung đẹp): khung có nhiều object chèn nhất trong 20 m, seed 0.
Tiêu đề phụ ghi recall của CẢ kịch bản (mọi khung, 2-3 seed) để khung minh hoạ không bị hiểu thành kết quả chung.

Thời tiết: cùng 1 khung bầy xe máy dưới trời trong / mưa to / sương dày / giông.

Chạy trong WSL env pcdet (GPU):  ~/miniconda3/envs/pcdet/bin/python tools/case_gallery.py
Output: output/cases/<nhóm>_<kịch bản>.png (từng ảnh) + output/cases/grid_*.png (ghép cho slide).
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tools.detect  # noqa: E402,F401  (load torch trước shapely)
import config  # noqa: E402
from tools.demo_video import STATUS_COLOR, draw_panel  # noqa: E402
from tools.t6_eval import inserted_gts, object_records, real_gts  # noqa: E402

OUT = os.path.join(config.OUTPUT_DIR, "cases")
CASES = [  # (nhóm, thư mục kết quả, kịch bản, tên hiển thị)
    ("t6", "t6", "control_2w", "Đối chứng: xe máy đi cùng chiều"),
    ("t6", "t6", "moto_swarm", "Bầy 10-20 xe máy quanh xe"),
    ("t6", "t6", "cut_in", "Xe máy cắt đầu 3-8 m"),
    ("t6", "t6", "wrong_way", "Xe ngược chiều trong làn"),
    ("t6", "t6", "jaywalk", "Người băng ngang ngoài vạch"),
    ("t6", "t6", "lane_block", "Ô tô / xe tải đỗ lấn làn"),
    ("vn", "vn", "rc_control_moto", "Đối chứng: xe máy ray-cast"),
    ("vn", "vn", "multi_rider", "Xe máy chở 2-3 người"),
    ("vn", "vn", "cargo_moto", "Xe máy chở hàng cồng kềnh"),
    ("vn", "vn", "ba_gac_slow", "Xe ba gác chạy chậm sát lề"),
    ("vn", "vn", "tow_moto", "Xe máy kéo thùng hàng"),
    ("vn", "vn", "vendor_curb", "Xe hàng rong lấn lòng đường"),
    ("vn", "vn", "dog_dart", "Chó lao ngang trước xe"),
    ("vn", "vn", "cow_road", "Trâu / bò trên lòng đường"),
    ("vn", "vn", "ox_cart_road", "Xe bò đi chậm trong làn"),
    ("vn", "vn", "rush_hour", "Giờ cao điểm (mix)"),
    ("vn", "vn", "market_street", "Phố chợ (mix)"),
    ("vn", "vn", "animal_in_traffic", "Chó lao ra giữa bầy xe (mix)"),
    ("vn", "vn", "ped_group_cross", "Nhóm 3-8 người băng ngang"),
    ("vn", "vn", "ped_crowd", "Đám đông 10-20 người dưới lòng đường"),
]
WEATHER = [("clear", None, None, "Trời trong"), ("rain_heavy", "rain", "heavy", "Mưa to"),
           ("fog_dense", "fog", "dense", "Sương dày"), ("storm", "storm", "storm", "Giông / bão")]


def pick_frame(aug_dir):
    """Khung có nhiều object chèn nhất trong 20 m (cố định, không chọn theo kết quả detector)."""
    best, best_n = None, -1
    for f in sorted(os.listdir(aug_dir)):
        if not f.endswith(".json"):
            continue
        with open(os.path.join(aug_dir, f), encoding="utf-8") as fh:
            r = json.load(fh)
        n = sum(np.hypot(*a["translation"][:2]) < 20 for a in r["annotations_sensor"])
        if n > best_n:
            best, best_n = f[:-5], n
    return best


def scenario_recall(records_path):
    """{kịch bản: (recall object chèn, n)} trên mọi khung / seed; gộp cả records_*.json chạy bổ sung."""
    import glob
    by = defaultdict(list)
    for path in [records_path] + sorted(glob.glob(records_path.replace("records.json", "records_*.json"))):
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            recs = json.load(f)
        groups = {r["group"] for r in recs if r["group"] != "real"}
        for g in groups:  # file sau thay nhóm của file trước
            by.pop(g, None)
        for r in recs:
            if r["group"] != "real":
                by[r["group"]].append(r["detected"])
    return {g: (float(np.mean(v)), len(v)) for g, v in by.items()}


def view(gts, n_real, min_half=9.0):
    """Khung nhìn phóng vào object chèn, giữ cả xe ego (0, 0): (tâm, nửa cạnh)."""
    xy = np.array([[0.0, 0.0]] + [g["center"][:2] for g in gts[n_real:]])
    lo, hi = xy.min(0) - 4, xy.max(0) + 4
    return tuple((lo + hi) / 2), max(min_half, float((hi - lo).max()) / 2)


def export_3d(path, pts, gts, recs, pred, n_real, max_points=22000, radius=35.0, seed=0):
    """JSON cho trang demo 3D (docs/demo): điểm keyframe (lấy mẫu), cờ điểm object chèn, box GT + trạng thái, box pred."""
    from task_a.merge import points_in_new_box
    from tools.detect import SCORE_THRESH
    p = pts[pts[:, 4] == 0]
    p = p[np.hypot(p[:, 0], p[:, 1]) < radius]
    obj = np.zeros(len(p), bool)
    for g in gts[n_real:]:
        obj |= points_in_new_box(p, g)
    bg = np.flatnonzero(~obj)
    keep_bg = np.random.default_rng(seed).choice(bg, min(len(bg), max_points - obj.sum()), replace=False)
    idx = np.r_[np.flatnonzero(obj), keep_bg]
    boxes = []
    for i, (g, r) in enumerate(zip(gts, recs)):
        if np.hypot(*g["center"][:2]) > radius:
            continue
        status = "ok" if r["detected"] else r["miss"].split(":")[0]
        boxes.append({"c": [round(float(v), 2) for v in g["center"]], "s": [round(float(v), 2) for v in g["size"]],
                      "yaw": round(float(g["yaw"]), 3), "status": status, "inserted": i >= n_real,
                      "cls": g["cls"], "as": r["miss"].split(":")[1] if status == "confused" else None})
    keep = pred["scores"] >= SCORE_THRESH
    preds = [{"c": [round(float(v), 2) for v in b[:3]], "s": [round(float(b[4]), 2), round(float(b[3]), 2),
                                                             round(float(b[5]), 2)], "yaw": round(float(b[6]), 3),
              "label": str(lab)} for b, lab in zip(pred["boxes"][keep], pred["labels"][keep])
             if np.hypot(*b[:2]) < radius]
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"points": np.round(p[idx, :3], 2).ravel().tolist(), "obj": obj[idx].astype(int).tolist(),
                   "gt": boxes, "pred": preds}, f, separators=(",", ":"))


from utils.camera_proj import BOX_EDGES, CAMS  # noqa: E402
from utils.camera_proj import lidar_to_cam as _lidar_to_cam  # noqa: E402
from utils.camera_proj import project as _project  # noqa: E402


def camera_view(path, nusc, tok, pts, gts, recs, n_real, title):
    """Ảnh camera thật + điểm LiDAR chiếu lên (tím = object chèn) + box 3D chiếu lên (màu theo kết quả detector).

    Ảnh camera KHÔNG có object chèn (chỉ chèn vào LiDAR): object chèn chỉ hiện qua điểm LiDAR + box chiếu lên.
    Chọn camera thấy nhiều tâm object chèn nhất.
    """
    import matplotlib.pyplot as plt
    from PIL import Image

    from task_a.merge import points_in_new_box
    from utils.visualization import to_box
    ins = gts[n_real:]
    best = max(CAMS, key=lambda c: int(_project(*_lidar_to_cam(nusc, tok, c)[:2],
                                                np.array([g["center"] for g in ins]).reshape(-1, 3))[1].sum()))
    T, K, img_path = _lidar_to_cam(nusc, tok, best)
    p = pts[pts[:, 4] == 0]
    obj = np.zeros(len(p), bool)
    for g in ins:
        obj |= points_in_new_box(p, g)
    uv, ok, _ = _project(T, K, p[:, :3])
    depth = np.linalg.norm(p[:, :3], axis=1)
    fig, ax = plt.subplots(figsize=(12, 7.4), dpi=110)
    ax.imshow(Image.open(img_path))
    bg = ok & ~obj
    ax.scatter(uv[bg, 0], uv[bg, 1], s=0.6, c=depth[bg], cmap="viridis_r", vmin=0, vmax=40, alpha=0.35, linewidths=0)
    ax.scatter(uv[ok & obj, 0], uv[ok & obj, 1], s=5, c=OBJ_POINTS_HEX, linewidths=0)
    for i, (g, r) in enumerate(zip(gts, recs)):
        corners = to_box(g).corners().T
        cuv, cok, cfront = _project(T, K, corners)
        if not cfront.all() or not cok.any():
            continue
        status = "ok" if r["detected"] else r["miss"].split(":")[0]
        inserted = i >= n_real  # object chèn: đậm; object thật nuScenes: mờ
        for a, b in BOX_EDGES:
            ax.plot(cuv[[a, b], 0], cuv[[a, b], 1], color=STATUS_COLOR[status], lw=1.6 if inserted else 0.8,
                    alpha=1.0 if inserted else 0.35)
    ax.set(xlim=(0, 1600), ylim=(900, 0), xticks=[], yticks=[])
    ax.set_title(f"{title}  —  {best}\nảnh camera thật (không có object chèn); object chèn = điểm LiDAR tím + box 3D chiếu lên",
                 fontsize=10)
    legend(fig, ncol=3)
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    fig.savefig(path)
    plt.close(fig)
    return best


OBJ_POINTS_HEX = "#b14cff"  # tím sáng: nổi trên ảnh camera
TILE_W, TILE_H = 800, 450   # ảnh camera thu nhỏ cho trang demo (gốc 1600 x 900)


def camera_tiles(out_dir, nusc, tok, pts, gts, recs, n_real, weather=None, level=None):
    """6 ảnh camera (JPEG thu nhỏ) cho trang demo kiểu CVAT: điểm LiDAR object chèn + box 3D chiếu lên
    (object chèn: đậm, màu theo kết quả detector; object thật: mảnh). Returns {camera: số object chèn thấy được}.
    weather / level: phủ thời tiết lên ảnh camera (task_b/camera_weather.py) cho khớp LiDAR đã mô phỏng."""
    from PIL import Image, ImageDraw

    from task_a.merge import points_in_new_box
    from utils.visualization import to_box
    os.makedirs(out_dir, exist_ok=True)
    p = pts[pts[:, 4] == 0]
    obj = np.zeros(len(p), bool)
    for g in gts[n_real:]:
        obj |= points_in_new_box(p, g)
    sx, sy = TILE_W / 1600, TILE_H / 900
    from utils.camera_proj import render_camera
    boxes = []
    for i, (g, r) in enumerate(zip(gts, recs)):
        status = "ok" if r["detected"] else r["miss"].split(":")[0]
        boxes.append(dict(g, inserted=(i >= n_real), status=status))

    seen = {}
    for idx, cam in enumerate(CAMS):
        img, n_seen = render_camera(nusc, tok, cam, boxes, obj_points=None,
                                    size=(TILE_W, TILE_H), weather=weather, level=level,
                                    seed=idx, mode="photorealistic_box")
        img.save(os.path.join(out_dir, f"{cam}.jpg"), quality=82)
        seen[cam] = int(n_seen)
    return seen


def frame_line(recs):
    ok = sum(r["detected"] for r in recs)
    conf = sum(1 for r in recs if r["miss"] and r["miss"].startswith("confused"))
    return f"khung này: đúng {ok}/{len(recs)}, nhầm lớp {conf}, sót {len(recs) - ok - conf}"


def legend(fig, ncol=5):
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=STATUS_COLOR["ok"], lw=2, label="phát hiện đúng"),
                        Line2D([], [], color=STATUS_COLOR["missed"], lw=2, label="bỏ sót"),
                        Line2D([], [], color=STATUS_COLOR["confused"], lw=2, label="nhận nhầm lớp"),
                        Line2D([], [], color="#1f6fd1", lw=1, ls="--", label="prediction của model"),
                        Line2D([], [], color="#6250d6", marker="o", ls="", ms=4, label="điểm LiDAR object chèn"),
                        Line2D([], [], color=STATUS_COLOR["ok"], lw=0.7, alpha=0.35,
                               label="box mờ = object thật nuScenes")],
               loc="lower center", ncol=ncol, fontsize=9, frameon=False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tqdm import tqdm

    from task_a.sweeps import load_multisweep
    from task_b.run_weather import apply_weather
    from tools.detect import Detector
    from utils.nuscenes_loader import load_nuscenes

    nusc = load_nuscenes()
    det = Detector()
    os.makedirs(OUT, exist_ok=True)
    stats = {d: scenario_recall(os.path.join(config.OUTPUT_DIR, d, "records.json")) for d in ("t6", "vn")}
    rendered = defaultdict(list)  # nhóm -> [(kịch bản, tên, points, gts, recs, pred, n_real, sub)]
    frame_stats = {}
    for group, d, sc, label in tqdm(CASES, desc="cases"):
        aug_dir = os.path.join(config.OUTPUT_DIR, d, "seed0", sc)
        if not os.path.isdir(aug_dir):
            continue
        tok = pick_frame(aug_dir)
        with open(os.path.join(aug_dir, f"{tok}.json"), encoding="utf-8") as f:
            result = json.load(f)
        real = real_gts(nusc, tok)
        gts = real + inserted_gts(result)
        pts = load_multisweep(nusc, tok, aug_dir)
        pred = det.predict(pts)
        recs = object_records(gts, pred, sc, tok, 0)
        rec, n = stats[d].get(sc, (float("nan"), 0))
        sub = f"cả kịch bản: phát hiện {100 * rec:.0f}% (n={n})  |  {frame_line(recs[len(real):])}"
        rendered[group].append((sc, label, pts, gts, recs, pred, len(real), sub))
        os.makedirs(os.path.join(OUT, "3d"), exist_ok=True)
        export_3d(os.path.join(OUT, "3d", f"{sc}.json"), pts, gts, recs, pred, len(real))
        camera_view(os.path.join(OUT, f"cam_{sc}.png"), nusc, tok, pts, gts, recs, len(real), label)
        frame_stats[sc] = {"label": label, "group": group, "recall_all": rec, "n_all": n, "frame": frame_line(recs[len(real):]),
                           "cams": camera_tiles(os.path.join(OUT, "camtiles", sc), nusc, tok, pts, gts, recs, len(real))}
        c, half = view(gts, len(real))
        fig, ax = plt.subplots(figsize=(8, 9), dpi=110)
        draw_panel(ax, pts, gts, recs, pred, half, f"{label}\n{sub}", len(real), highlight=True, center=c)
        ax.title.set_fontsize(10)
        legend(fig, ncol=3)
        fig.tight_layout(rect=(0, 0.07, 1, 0.98))
        fig.savefig(os.path.join(OUT, f"{group}_{sc}.png"))
        plt.close(fig)

    for group, items in rendered.items():  # ghép cho slide
        cols = 3
        rows = int(np.ceil(len(items) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.4 * rows), dpi=90, squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (sc, label, pts, gts, recs, pred, n_real, sub) in zip(axes.ravel(), items):
            ax.axis("on")
            c, half = view(gts, n_real)
            draw_panel(ax, pts, gts, recs, pred, half, f"{label}\n{sub}", n_real, highlight=True, center=c)
            ax.title.set_fontsize(9)
        legend(fig)
        fig.tight_layout(rect=(0, 0.03, 1, 1))
        fig.savefig(os.path.join(OUT, f"grid_{group}.png"))
        plt.close(fig)

    # thời tiết: cùng 1 khung bầy xe máy
    aug_dir = os.path.join(config.OUTPUT_DIR, "t6", "seed0", "moto_swarm")
    if os.path.isdir(aug_dir):
        tok = pick_frame(aug_dir)
        with open(os.path.join(aug_dir, f"{tok}.json"), encoding="utf-8") as f:
            result = json.load(f)
        real = real_gts(nusc, tok)
        gts = real + inserted_gts(result)
        vehicles = [g for g in gts if g["cls"] in ("car", "truck", "bus", "motorcycle", "trailer")]
        fig, axes = plt.subplots(1, 4, figsize=(24, 6.8), dpi=90)
        for ax, (cond, weather, level, label) in zip(axes, WEATHER):
            rng = np.random.default_rng(config.SEED)

            def per_sweep(p, i):
                return p if weather is None else apply_weather(p, weather, level, rng, vehicles if i == 0 else ())[0]
            pts = load_multisweep(nusc, tok, aug_dir, per_sweep=per_sweep)
            pred = det.predict(pts)
            recs = object_records(gts, pred, cond, tok, 0)
            export_3d(os.path.join(OUT, "3d", f"weather_{cond}.json"), pts, gts, recs, pred, len(real))
            frame_stats[f"weather_{cond}"] = {"label": label, "group": "weather", "frame": frame_line(recs[len(real):]),
                                              "cams": camera_tiles(os.path.join(OUT, "camtiles", f"weather_{cond}"),
                                                                   nusc, tok, pts, gts, recs, len(real),
                                                                   weather, level)}
            draw_panel(ax, pts, gts, recs, pred, 18.0, f"{label}\n{frame_line(recs[len(real):])} (xe máy chèn)",
                       len(real), highlight=True)
            ax.title.set_fontsize(10)
        legend(fig)
        fig.suptitle("Cùng 1 khung bầy xe máy dưới các điều kiện thời tiết (detector pretrained nuScenes)", fontsize=12)
        fig.tight_layout(rect=(0, 0.05, 1, 0.95))
        fig.savefig(os.path.join(OUT, "weather_swarm.png"))
        plt.close(fig)
    with open(os.path.join(OUT, "3d", "cases.json"), "w", encoding="utf-8") as f:
        json.dump(frame_stats, f, ensure_ascii=False, indent=1)
    print("->", OUT, sorted(os.listdir(OUT)))


if __name__ == "__main__":
    main()
