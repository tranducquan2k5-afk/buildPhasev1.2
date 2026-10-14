# N3-08C — Synthetic Point Cloud Augmentation (nuScenes)

Sinh dữ liệu LiDAR tổng hợp từ nuScenes v1.0-mini: chèn object (Task A) và mô phỏng mưa / sương mù (Task B).

**Hướng dẫn đầy đủ: [HUONG_DAN.md](HUONG_DAN.md)** — cài đặt, tải dữ liệu, cách chạy, kết quả mong đợi, lỗi thường gặp.

Bắt đầu nhanh (Windows PowerShell, Python 3.10):

```powershell
py -3.10 -m venv nuscenes-env
.\nuscenes-env\Scripts\Activate.ps1
pip install -r n3-08c\requirements.txt
# tải nuScenes Mini + Map expansion vào datasets\nuscenes\ (xem HUONG_DAN.md mục 2.3)
cd n3-08c
python verify_setup.py --no-vis
```

---

## Phiên bản v1.3 — Cloud SaaS Web App, 3D WebGL Viewer & Docker Containerization

Bản nâng cấp **v1.3** chuyển đổi toàn diện dự án nghiên cứu thành sản phẩm SaaS hoàn chỉnh:

1. **Full-Stack SaaS Architecture**:
   - **Backend**: FastAPI hiệu năng cao, cơ chế Lazy Global Cache nạp Singleton RAM cho nuScenes và Template Bank.
   - **Frontend**: Single-Page App (SPA) với **3D WebGL Point Cloud Viewer** (Three.js), hiển thị bounding box 3D khít 100% kèm vạch chéo mũi xe và mũi tên chỉ hướng lái (Road Heading).
   - **Chế độ kép (Dual Mode)**:
     - *Mode 1 (nuScenes Sandbox)*: Chèn vật thể dựa trên bản đồ vector (`drivable_area`, `walkway`) với dung sai khử che khuất $\pm 15\text{ cm}$.
     - *Mode 2 (Custom Upload / Mapless Fallback)*: Tải lên file point cloud tùy ý, tự động nhận diện mặt đường RANSAC và chèn theo luồng giao thông.

2. **Hỗ trợ Đa Định dạng LiDAR (Universal IO)**:
   - Hỗ trợ đầy đủ các định dạng công nghiệp phổ biến từ CVAT, ROS, nuScenes: `.bin`, `.pcd`, `.ply`, `.las`, `.laz`, `.xyz`, `.csv`, `.npy`.
   - Tự động đóng gói kết quả tải về dạng `.zip` gồm file đám mây điểm đã augment và file nhãn 3D Bounding Box (`.json`).

3. **Cải tiến Thuật toán Vật lý Lõi**:
   - **Cách ly vật lý (Mutual Separation)**: Bắt buộc khoảng cách giữa 2 xe mới chèn $\ge 15.0\text{m}$ và lệch góc phương vị $\vert\Delta\theta\vert \ge 40.0^\circ$ quanh cảm biến, triệt tiêu 100% hiện tượng tự che khuất hay xóa điểm của nhau.
   - **Buffer an toàn giao thông**: Mở rộng $\pm 1.5\text{m}$ mỗi chiều khi kiểm tra va chạm BEV Shapely.
   - **Lọc Template Bank chất lượng cao**: Xe tải/buýt $\ge 400$ điểm & độ phủ chiều dài $\ge 55\%$; ô tô con $\ge 180$ điểm & độ phủ $\ge 55\%$.
   - **Mô phỏng thời tiết nhiệt đới**: Mưa Beer-Lambert, sương mù tán xạ ngược và vũng nước mặt đường phản xạ gương (15 vũng nước/scene).

### Khởi chạy nhanh Web App (Local):
```powershell
pip install -r requirements_web.txt
python run_app.py
```
- Dashboard trực quan: `http://localhost:8050`
- Interactive API Swagger: `http://localhost:8050/docs`
- Slide thuyết trình & Phản biện: `http://localhost:8050/slides`

### Khởi chạy bằng Docker:
```bash
docker compose up --build
```
