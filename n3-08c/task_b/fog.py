"""Mô phỏng sương mù trên LiDAR point cloud.

2 bước: (1) exp-decay dropout, survival = exp(-d / visibility) -> (2) backscatter (điểm giả gần sensor).
visibility (m): 200 = light, 50 = moderate, 20 = dense.
Chạy demo: python task_b/fog.py
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402

BACKSCATTER_REF = 0.05  # tỉ lệ beam bị backscatter khi visibility = 20 m. TODO: hiệu chỉnh theo dữ liệu thật.


def simulate_fog(points: np.ndarray, visibility: float = 50.0,
                 rng: Optional[np.random.Generator] = None, return_labels: bool = False):
    """Sương mù với tầm nhìn visibility (m).

    Returns augmented points (K, 5); nếu return_labels thì thêm labels (K,) 0 = gốc, 1 = backscatter.
    """
    if visibility <= 0:
        raise ValueError("visibility phải > 0")
    rng = rng or np.random.default_rng()
    dist = np.linalg.norm(points[:, :3], axis=1)

    # Backscatter: xác định tia bị phản xạ ngược bởi sương mù.
    # Sử dụng BACKSCATTER_REF làm tham chiếu cho sương đặc (V=20m)
    ratio = min(1.0, BACKSCATTER_REF * 20.0 / visibility)
    is_scatter = rng.random(len(points)) < ratio

    # 1. Xử lý các điểm không bị backscatter (chịu suy giảm và dropout)
    points_normal = points[~is_scatter].copy()
    dist_normal = dist[~is_scatter]
    
    # Phụ thuộc reflectivity: điểm cường độ cao đâm xuyên sương tốt hơn
    # Chuẩn hóa intensity (nuScenes: 0-255)
    norm_intensity = np.clip(points_normal[:, 3] / 255.0, 0, 1)
    # Tăng visibility hiệu dụng cho điểm sáng (điểm sáng có thể kéo dài visibility thêm tới 50%)
    eff_visibility = visibility * (1.0 + 0.5 * norm_intensity)
    
    survival = np.exp(-dist_normal / eff_visibility)
    kept_mask = rng.random(len(points_normal)) < survival
    kept = points_normal[kept_mask]
    kept[:, 3] *= np.exp(-np.linalg.norm(kept[:, :3], axis=1) / visibility)  # intensity suy giảm

    # 2. Xử lý các điểm bị backscatter (thay thế điểm gốc trên cùng tia quét)
    scatter = points[is_scatter].copy()
    d_beam = np.maximum(dist[is_scatter], 1e-3)
    # Rút ngắn khoảng cách của tia (điểm tán xạ ngược luôn nằm gần sensor)
    r_scatter = np.minimum(rng.gamma(2.0, visibility / 20.0, len(scatter)) + 1.0, d_beam)
    scatter[:, :3] = scatter[:, :3] * (r_scatter / d_beam)[:, None]
    scatter[:, 3] = rng.uniform(0, 5, len(scatter))

    out = np.concatenate([kept, scatter.astype(points.dtype)], axis=0)
    if return_labels:
        return out, np.r_[np.zeros(len(kept), np.int8), np.ones(len(scatter), np.int8)]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    pts = np.c_[rng.uniform(-50, 50, (30000, 2)), rng.uniform(-2, 1, 30000),
                rng.uniform(0, 255, 30000), rng.integers(0, 32, 30000)].astype(np.float32)
    for level, vis in config.FOG_VISIBILITY.items():
        out, lab = simulate_fog(pts, vis, rng, return_labels=True)
        far = np.linalg.norm(out[lab == 0, :3], axis=1).max()
        print(f"{level:9s} (V={vis:.0f} m): {len(pts)} -> {len(out)} pts "
              f"(backscatter {lab.sum()}), max range kept {far:.1f} m")
