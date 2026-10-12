"""Tự động render và chụp các góc nhìn 3D (3D Snapshots) của scene đã chèn vật thể.
Lưu kết quả trực tiếp thành các ảnh chất lượng cao vào output/vis/ để phục vụ báo cáo/slide.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from utils.nuscenes_loader import get_annotations, load_bin, load_nuscenes
from utils.visualization import to_box


def render_3d_snapshots(sample_token: str = "023c4df2d451409881d8e6ea82f14704"):
    nusc = load_nuscenes()
    bin_path = os.path.join(config.AUGMENTED_DIR, f"{sample_token}.bin")
    json_path = os.path.join(config.AUGMENTED_DIR, f"{sample_token}.json")

    if not os.path.isfile(bin_path):
        raise FileNotFoundError(f"Không tìm thấy {bin_path}")

    points = load_bin(bin_path)
    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    new_anns = meta["annotations_sensor"]
    existing_anns = get_annotations(nusc, sample_token)

    os.makedirs(config.VIS_DIR, exist_ok=True)

    # 1. TẠO BẢNG GHÉP 4 GÓC NHÌN 3D CHUẨN BÁO CÁO (3D Multi-View Scientific Panel)
    fig = plt.figure(figsize=(18, 14), facecolor="white")
    plt.subplots_adjust(wspace=0.15, hspace=0.2)

    # Lọc các điểm trong bán kính 45m quanh sensor
    r = np.linalg.norm(points[:, :3], axis=1)
    mask = (r < 45.0) & (points[:, 2] > -2.5) & (points[:, 2] < 3.0)
    vis_pts = points[mask]

    # --- Panel 1: 3D Isometric Overview (Góc nghiêng 3D toàn cảnh) ---
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    sc1 = ax1.scatter(vis_pts[:, 0], vis_pts[:, 1], vis_pts[:, 2],
                      c=vis_pts[:, 2], cmap="viridis", s=0.3, alpha=0.7)
    _draw_3d_boxes(ax1, existing_anns, color="lime", lw=1.0)
    _draw_3d_boxes(ax1, new_anns, color="red", lw=2.5)
    ax1.scatter([0], [0], [0], color="black", s=50, marker="^", label="Sensor (LIDAR_TOP)")
    ax1.set_title("1. Toàn cảnh 3D Phối cảnh (Isometric Overview)\n[Xanh lá: Vật thể gốc | Đỏ: Vật thể chèn mới]", fontsize=11, fontweight="bold")
    ax1.view_init(elev=35, azim=-60)
    ax1.set_xlim([-30, 30])
    ax1.set_ylim([-30, 30])
    ax1.set_zlim([-3, 5])
    ax1.set_xlabel("X (m)")
    ax1.set_ylabel("Y (m)")
    ax1.set_zlabel("Z (m)")

    # --- Panel 2: 3D Cận cảnh chiếc Xe tải & Vùng Occlusion (Close-up Truck & Occlusion) ---
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    truck_ann = next((a for a in new_anns if "truck" in a["category"]), new_anns[0])
    tx, ty, tz = truck_ann["translation"]
    
    # Lấy vùng quanh xe tải
    truck_mask = (np.abs(points[:, 0] - tx) < 12.0) & (np.abs(points[:, 1] - ty) < 12.0)
    truck_pts = points[truck_mask]
    
    ax2.scatter(truck_pts[:, 0], truck_pts[:, 1], truck_pts[:, 2],
                c=truck_pts[:, 2], cmap="plasma", s=1.2, alpha=0.8)
    _draw_3d_boxes(ax2, [truck_ann], color="red", lw=3.0)
    ax2.set_title(f"2. Cận cảnh 3D: Xe tải ({truck_ann['category']}) & Đổ bóng che khuất\n[Vùng khuyết điểm phía sau xe chứng minh Occlusion Ray Casting]", fontsize=11, fontweight="bold")
    ax2.view_init(elev=25, azim=-45)
    ax2.set_xlim([tx - 8, tx + 8])
    ax2.set_ylim([ty - 8, ty + 8])
    ax2.set_zlim([-2.5, 3.5])
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_zlabel("Z (m)")

    # --- Panel 3: Góc nhìn từ mắt tài xế (Driver Eye / Forward Perspective) ---
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    fwd_mask = (points[:, 1] > -30) & (points[:, 1] < 10) & (r < 35.0)
    fwd_pts = points[fwd_mask]
    ax3.scatter(fwd_pts[:, 0], fwd_pts[:, 1], fwd_pts[:, 2],
                c=fwd_pts[:, 3], cmap="cividis", s=0.5, alpha=0.75)
    _draw_3d_boxes(ax3, existing_anns, color="lime", lw=1.0)
    _draw_3d_boxes(ax3, new_anns, color="red", lw=2.5)
    ax3.set_title("3. Góc nhìn từ phía sau xe tự lái (Ego-Vehicle Perspective)", fontsize=11, fontweight="bold")
    ax3.view_init(elev=12, azim=-90)  # Nhìn thẳng về phía trước
    ax3.set_xlim([-20, 20])
    ax3.set_ylim([-30, 10])
    ax3.set_zlim([-2, 4])
    ax3.set_xlabel("X (m)")
    ax3.set_ylabel("Y (m)")
    ax3.set_zlabel("Z (m)")

    # --- Panel 4: Bird's Eye View (BEV) 2D Chiếu thẳng từ trên cao ---
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(points[m := (r < 35), 0], points[m, 1], s=0.3, c=points[m, 2], cmap="viridis")
    for a in existing_anns:
        _draw_2d_box(ax4, a, color="lime", lw=1.2)
    for a in new_anns:
        _draw_2d_box(ax4, a, color="red", lw=2.2)
        ax4.text(a["translation"][0], a["translation"][1] + 1.2, a["category"].split(".")[-1],
                 color="darkred", fontsize=8, fontweight="bold", ha="center")
    ax4.plot(0, 0, marker="^", color="black", ms=10, label="Ego LIDAR")
    ax4.set_xlim([-30, 30])
    ax4.set_ylim([-30, 30])
    ax4.set_aspect("equal")
    ax4.grid(True, linestyle="--", alpha=0.3)
    ax4.set_title("4. Bản đồ 2D Bird's Eye View (BEV Top-Down)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("X (m)")
    ax4.set_ylabel("Y (m)")
    ax4.legend(loc="upper right")

    output_panel_path = os.path.join(config.VIS_DIR, f"3d_multi_view_report_{sample_token}.png")
    fig.savefig(output_panel_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[Success] Saved 3D multi-view panel to: {output_panel_path}")

    # 2. RENDER ANH 3D RIENG DO PHAN GIAI CAO (HD CLOSE-UP)
    fig_close = plt.figure(figsize=(10, 8))
    ax_close = fig_close.add_subplot(1, 1, 1, projection="3d")
    ax_close.scatter(truck_pts[:, 0], truck_pts[:, 1], truck_pts[:, 2],
                     c=truck_pts[:, 2], cmap="turbo", s=1.5, alpha=0.85)
    _draw_3d_boxes(ax_close, [truck_ann], color="red", lw=3.0)
    ax_close.set_title(f"Can canh 3D: {truck_ann['category']}\nDa xu ly Occlusion & Point Removal", fontsize=12, fontweight="bold")
    ax_close.view_init(elev=20, azim=-55)
    ax_close.set_xlim([tx - 7, tx + 7])
    ax_close.set_ylim([ty - 7, ty + 7])
    ax_close.set_zlim([-2.5, 3.5])
    output_close_path = os.path.join(config.VIS_DIR, f"3d_closeup_truck_{sample_token}.png")
    fig_close.savefig(output_close_path, dpi=200, bbox_inches="tight")
    plt.close(fig_close)
    print(f"[Success] Saved 3D close-up truck to: {output_close_path}")


def _draw_3d_boxes(ax, anns, color="red", lw=1.5):
    edges = [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    for ann in anns:
        b = to_box(ann)
        corners = b.corners()  # (3, 8)
        for e in edges:
            ax.plot(corners[0, e], corners[1, e], corners[2, e], color=color, lw=lw)


def _draw_2d_box(ax, ann, color="red", lw=1.5):
    b = to_box(ann)
    xy = b.bottom_corners()[:2]
    ax.plot(np.append(xy[0], xy[0, 0]), np.append(xy[1], xy[1, 0]), color=color, lw=lw)


if __name__ == "__main__":
    render_3d_snapshots()
