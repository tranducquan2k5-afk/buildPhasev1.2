"""Mô phỏng mặt đường ướt và vũng nước (Wet road / Puddles) cho LiDAR point cloud.

Tia LiDAR khi chiếu xuống vũng nước sẽ bị phản xạ gương (specular reflection), 
khiến tín hiệu không quay trở lại sensor -> tạo ra các mảng mất điểm trên mặt đường.
"""
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config


def simulate_puddles(points: np.ndarray, num_puddles: int = 15, max_radius: float = 4.0,
                     rng: Optional[np.random.Generator] = None, return_labels: bool = False):
    """Mô phỏng vũng nước bằng cách tạo các mảng hụt điểm ngẫu nhiên trên mặt đường.
    
    Args:
        points: (N, 5) mảng point cloud.
        num_puddles: Số lượng vũng nước tối đa.
        max_radius: Bán kính lớn nhất của vũng nước (m).
        rng: Random generator.
    """
    rng = rng or np.random.default_rng()
    out_points = points.copy()
    labels = np.zeros(len(points), dtype=np.int8)

    # Lọc các điểm thuộc mặt đường (sensor LiDAR cao ~1.84m, ground z < -1.4m)
    ground_mask = out_points[:, 2] < -1.4
    ground_idx = np.where(ground_mask)[0]
    
    if len(ground_idx) == 0:
        return (out_points, labels) if return_labels else out_points

    ground_pts = out_points[ground_idx]
    
    # Giới hạn tọa độ để sinh vũng nước
    x_min, x_max = ground_pts[:, 0].min(), ground_pts[:, 0].max()
    y_min, y_max = ground_pts[:, 1].min(), ground_pts[:, 1].max()

    puddle_mask_ground = np.zeros(len(ground_pts), dtype=bool)

    # Tạo các vũng nước (hình ellipse ngẫu nhiên)
    for _ in range(num_puddles):
        cx = rng.uniform(x_min, x_max)
        cy = rng.uniform(y_min, y_max)
        rx = rng.uniform(0.5, max_radius)
        ry = rng.uniform(0.5, max_radius)
        
        # Phương trình ellipse
        dist_sq = ((ground_pts[:, 0] - cx)**2 / rx**2) + ((ground_pts[:, 1] - cy)**2 / ry**2)
        in_puddle = dist_sq <= 1.0
        puddle_mask_ground |= in_puddle

    # Lấy chỉ số các điểm ground nằm trong vũng nước đối chiếu lại mảng gốc
    puddle_idx = ground_idx[puddle_mask_ground]
    
    if len(puddle_idx) > 0:
        # Dropout 85% số điểm trong vũng nước (phản xạ gương)
        survival = rng.random(len(puddle_idx)) > 0.85
        survived_puddle_idx = puddle_idx[survival]
        dropped_puddle_idx = puddle_idx[~survival]
        
        # Những điểm sống sót trong vũng nước sẽ bị giảm intensity mạnh (nước hấp thụ tia IR)
        out_points[survived_puddle_idx, 3] *= 0.3 
        
        # Xoá các điểm bị dropout
        keep_mask = np.ones(len(out_points), dtype=bool)
        keep_mask[dropped_puddle_idx] = False
        
        out_points = out_points[keep_mask]
        labels = labels[keep_mask]
        
        # Đánh dấu label cho những điểm vũng nước còn sót lại
        # Re-map index vì mảng đã bị thu ngắn
        # Nhưng thay vì re-map phức tạp, ta có thể sinh mask ngay từ đầu
    
    # Cách làm an toàn hơn cho dropout:
    # 1. Tạo mask keep_mask toàn cục
    keep_mask = np.ones(len(points), dtype=bool)
    
    # Các điểm thuộc vũng nước
    global_puddle_mask = np.zeros(len(points), dtype=bool)
    global_puddle_mask[ground_idx] = puddle_mask_ground
    
    # Dropout trên vũng nước
    puddle_dropout = rng.random(len(points)) < 0.85
    dropped = global_puddle_mask & puddle_dropout
    keep_mask[dropped] = False
    
    # Các điểm vũng nước sống sót
    survived_puddle = global_puddle_mask & ~dropped
    
    out_points = points[keep_mask].copy()
    
    # Giảm intensity cho các điểm sống sót (phải map qua mask mới)
    survived_in_out = survived_puddle[keep_mask]
    out_points[survived_in_out, 3] *= 0.3
    
    labels = np.zeros(len(out_points), dtype=np.int8)
    labels[survived_in_out] = 2  # Label 2 = Vũng nước
    
    if return_labels:
        return out_points, labels
    return out_points


if __name__ == "__main__":
    rng = np.random.default_rng(config.SEED)
    pts = np.c_[rng.uniform(-30, 30, (50000, 2)), 
                np.full(50000, -1.8), # Đều nằm trên mặt đất
                rng.uniform(50, 255, 50000), 
                rng.integers(0, 32, 50000)].astype(np.float32)
                
    out, lab = simulate_puddles(pts, num_puddles=10, max_radius=5.0, rng=rng, return_labels=True)
    print("Mo phong vung nuoc (puddles):")
    print(f"  So diem ban dau: {len(pts)}")
    print(f"  So diem sau khi phan xa guong: {len(out)}")
    print(f"  So diem vung nuoc con sot lai: {np.sum(lab == 2)}")
    print(f"  Intensity trung binh: {pts[:, 3].mean():.1f} -> {out[:, 3].mean():.1f}")
