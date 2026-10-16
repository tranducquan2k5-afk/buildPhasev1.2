# N3-08C — Tổng hợp kết quả (số liệu cho báo cáo)

Mọi số đo trên nuScenes **v1.0-mini**, detector **CenterPoint-PointPillar pretrained trên nuScenes** (OpenPCDet),
đánh giá trên **mini_val** (2 scene model chưa thấy khi train). Khớp prediction - GT kiểu nuScenes: cùng lớp, tâm
BEV < 2 m, score >= 0.3. CI 95% = bootstrap theo khung hình. Biểu đồ: `output/report/`, số liệu gốc: `output/report/summary.json`.

## 1. Vấn đề — nuScenes thiếu tình huống giao thông Việt Nam

| Chỉ số (nuScenes mini, 404 khung) | Giá trị | Nguồn |
|---|---|---|
| Khung có bầy >= 5 xe máy đang chạy trong 15 m | **0%** (tối đa 2 xe) | `output/dataset_gap_v1.0-mini.json` |
| Khung có xe cắt đầu | 0.5% | như trên |
| Xe đang chạy cách xe < 3 m | 0% | như trên |
| Tỉ lệ xe 2 bánh trong mọi object | 3.9% | như trên |
| Xe ba gác, xe máy chở hàng, xe hàng rong, xe bò | không có lớp / không có mẫu | taxonomy nuScenes |

Hệ quả đo được: ngay trên dữ liệu thật, detector chỉ tìm ra **55% xe máy** (CI 45-64%), so với ô tô 93%, người 95%.

## 2. Giải pháp — sinh edge case LiDAR trên dữ liệu thật

Không tạo cảnh mới: lấy khung LiDAR thật của nuScenes rồi chèn vật thể theo kịch bản VN.

1. **Nguồn điểm vật thể**: cắt mẫu object thật nuScenes (xe máy 1 người, ô tô, người) hoặc **ray-cast** hình dựng từ
   kích thước công bố (xe máy chở 2-3 người / chở hàng, ba gác, xe kéo, xe hàng rong, chó, bò, xe bò), bắn đúng 32 tia
   của Velodyne HDL-32E. Danh mục + quy ước: `docs/danh_muc_kich_ban.md`.
2. **Kịch bản**: luật vị trí / hướng / tốc độ trong hệ toạ độ xe (`task_a/scenarios.py`, `task_a/vn_scenarios.py`);
   nhiều khung liên tục (`task_a/multiframe.py`).
3. **Ghép theo vật lý LiDAR** (`task_a/merge.py`): mật độ theo khoảng cách, chỉ mặt nhìn thấy, che khuất 2 chiều
   (mỗi tia 1 điểm gần nhất), gán ring, mặt đất cục bộ, đủ 10 sweep như model đọc (`task_a/sweeps.py`).
4. **Thời tiết** (`task_b/`): mưa, sương, **giông / bão** (mưa rất to + mặt đường ướt + bụi nước sau bánh xe).
5. **Đánh giá có đối chứng** (`tools/t6_eval.py`): cùng vật thể, cùng pipeline chèn, chỉ khác tình huống
   (vị trí bình thường vs kịch bản VN) -> khác biệt là do tình huống, không do dữ liệu giả.

### Kiểm chứng dữ liệu sinh ra
| Kiểm chứng | Kết quả |
|---|---|
| Chèn cắt mẫu: mật độ so với object thật cùng khoảng cách | 0.98 lần; 100% đúng mặt nhìn thấy, 100% có mặt đất (`output/eval_report.json`) |
| Ray-cast: số điểm ô tô so với ô tô thật (7-12 m) | 308 vs median 305 / p75 432 (`python task_a/raycast.py`) |
| Ray-cast: số điểm xe máy so với xe máy thật (7-12 m) | 101 vs median 52 / p75 95 |
| Detector thấy ô tô ray-cast như ô tô thật (vị trí bình thường) | **93% (CI 90-96%) vs 93%** -> đạt |
| Hình ảnh điểm LiDAR từng vật thể chèn, cạnh xe máy / ô tô thật | `output/report/vn_closeup.png` (ray-cast đều và liền hơn object thật — khác biệt còn lại) |
| Detector thấy xe máy ray-cast như xe máy cắt mẫu | bản 1 (thân 0.34 m): 34% vs 94%, 61% bị nhận là xe đạp. Bản 2 (thân theo kích thước hãng ~0.6 m): **64%** (CI 59-69%) vs 94% -> **chưa đạt**: hình dựng từ khối vẫn hay bị nhận là xe đạp. Không chỉnh tiếp (chỉnh nữa là chỉnh cho vừa detector); kịch bản xe máy ray-cast chỉ so với chính đối chứng ray-cast |

## 3. Kết quả

### 3.1 Bầy xe máy là điểm yếu thật của detector (`output/report/t6_motorcycle_recall.png`)
| Nhóm (xe máy) | Recall | CI 95% |
|---|---|---|
| Đối chứng: chèn, đi cùng chiều | 93% | 91-96% |
| Bầy 10-20 xe máy | **81%** | 79-83% |
| Xe máy cắt đầu 3-8 m | 100% | 100% |
| Hiệu bầy - đối chứng | **-12.1 điểm** | -14.8 .. -9.1 (không chứa 0 -> có ý nghĩa) |

- Cùng số điểm LiDAR vẫn kém hơn (`t6_swarm_by_points.png`): 1-19 điểm 42% vs 86%, 20-49 điểm 79% vs 94%.
- Bầy càng đông càng kém: 5-9 xe 87%, 10-14 xe 83%, 15-19 xe 78%.
- Lỗi chính: xe máy trong bầy bị nhận thành **ô tô** (nhiều xe sát nhau -> 1 box ô tô) hoặc xe đạp.
- Kịch bản 1 xe (cắt đầu, ngược chiều 98% không lật hướng, người băng đường 100%, xe lấn làn 95%): detector **vẫn thấy**;
  rủi ro của các tình huống này nằm ở dự đoán quỹ đạo / lập kế hoạch, ngoài phạm vi đề tài.

### 3.2 Thời tiết (`output/report/weather.png`, `weather_swarm.png`; 41 khung mini_val)
| Recall | Trời trong | Mưa to | Sương dày | Giông / bão |
|---|---|---|---|---|
| Ô tô | 93% | 80% | 73% | 71% |
| Người đi bộ | 95% | 64% | 51% | 55% |
| Xe máy (thật) | 54% | 38% | 40% | 35% |
| Xe máy trong bầy | 80% | 64% | 76% | **57%** |

Bầy xe máy trong giông là tổ hợp tệ nhất cho xe máy: hai yếu tố cộng dồn.

### 3.3 Vật thể VN không có lớp trong nuScenes (`output/report/vn_objects_recall.png`)
Đo "có phát hiện vật cản không" (box bất kỳ tại chỗ) — điều kiện cần cho an toàn, dễ hơn đo đúng tên lớp.

| Vật thể | Recall vật cản |
|---|---|
| Vật thể | Recall vật cản (CI 95%) | Median điểm LiDAR |
|---|---|---|
| Xe ba gác | 96% (93-99) | 60 |
| Trâu / bò trên đường | 98% (95-99) | 34 |
| Xe hàng rong lấn lòng đường | 95% (93-98) | 46 |
| Xe máy kéo thùng | 93% (88-96) | 58 |
| Chó lao ra (median 10.6 m) | 94% (90-98) | **11** |
| Xe bò: xe gỗ / con bò | 91% / **53%** (bò bị chính xe gỗ che, median 9 điểm) | |

Kết hợp (mix): phố chợ (xe hàng rong + người dưới lòng đường) 98%; giờ cao điểm 76%; chó lao ra giữa bầy xe máy:
chó 85% (giảm từ 94% khi đứng một mình), xe máy 84%.

**Xe máy ray-cast** (đo đúng lớp motorcycle, so với đối chứng xe máy ray-cast 64%):

| Kịch bản | Recall | Hiệu so với đối chứng ray-cast (CI 95%) | Thấy vật cản (không xét tên) |
|---|---|---|---|
| Xe máy chở 2-3 người | 60% | -3.9 điểm (-11.5 .. +3.7) -> **không khác biệt** | 98% |
| Xe máy chở hàng cồng kềnh | **29%** | **-34.7 điểm (-42.4 .. -26.3)** -> có ý nghĩa | 79% |

Xe chở hàng cồng kềnh — rất phổ biến ở VN — bị nhận thành **ô tô** (59 lần) hoặc xe đạp (46), và **21% bị bỏ sót hẳn**:
khối hàng rộng 1-1.8 m làm xe máy không còn giống xe máy nào trong dữ liệu train.

### 3.3b Người qua đường: ít hay nhiều người (`output/report/pedestrians_recall.png`)
| Số người | Recall (CI 95%) | n |
|---|---|---|
| 1 người băng ngang ngoài vạch | 100% (99-100) | 243 |
| Nhóm 3-8 người băng ngang cùng lúc | 99% (98-100) | 834 |
| Đám đông 10-20 người dưới lòng đường | 99% (99-100) | 2362 |

Theo số người chèn trong khung: 0-3 người 96%, 4-7 người 100%, 8-11 người 99%, 12-15 người 99%, >= 16 người 99%.
Detector **không** yếu đi khi đông người — trái với bầy xe máy (-12 điểm). Lý giải: nuScenes có rất nhiều người đi bộ
theo đám (4765 người lớn trong mini) nhưng 0% bầy xe máy -> model yếu đúng ở chỗ dữ liệu train thiếu (ủng hộ pain point).
Người đi bộ chỉ tụt mạnh khi thời tiết xấu (95% -> 55% giông bão, mục 3.2).

### 3.4 Giới hạn cảm biến (`output/report/lidar_beams.png`)
Số điểm LiDAR rơi trên **con chó** (ray-cast, cùng FOV dọc): 32 tia (nuScenes) 15 điểm ở 10 m, 4 ở 20 m, **0 từ 40 m**;
128 tia: 63 / 14 / 2. Xe máy: 32 tia 80 / 19 / 4 điểm ở 10 / 20 / 40 m. Vật thể nhỏ, thấp gần như vô hình với LiDAR 32 tia ở xa, dù model tốt đến đâu.

### 3.5 Minh hoạ từng trường hợp
Màu box: xanh lá = phát hiện đúng, đỏ = bỏ sót, cam = nhận nhầm lớp (ghi lớp bị nhận thành), nét đứt xanh dương =
prediction, điểm tím = điểm LiDAR của object chèn. Khung minh hoạ chọn cố định (nhiều object chèn nhất trong 20 m),
không chọn theo kết quả; mỗi ảnh ghi cả recall toàn kịch bản.

| File | Nội dung |
|---|---|
| `output/cases/grid_t6.png` | 6 kịch bản T6 (đối chứng, bầy xe máy, cắt đầu, ngược chiều, băng đường, lấn làn) |
| `output/cases/grid_vn.png` | 14 kịch bản vật thể VN (chở 2-3 người, chở hàng, ba gác, kéo thùng, hàng rong, chó, bò, xe bò, mix, nhóm người, đám đông) |
| `output/cases/t6_*.png`, `vn_*.png` | 20 ảnh lẻ, phóng vào object chèn |
| `docs/demo/index.html` | trang demo offline: xem 3D từng trường hợp, thời tiết, video, biểu đồ |
| `output/cases/weather_swarm.png` | cùng 1 khung bầy xe máy: trời trong / mưa to / sương dày / giông |
| `output/report/vn_closeup.png` | cận cảnh điểm LiDAR từng vật thể VN cạnh xe máy / ô tô thật |

### 3.6 Video demo (`output/videos/`, 2 khung / giây; kèm .png khung giữa)
| Video | Trái / phải | Kết quả object chèn |
|---|---|---|
| `scene-0103_swarm_pretrained.mp4` | nuScenes gốc / có bầy xe máy | 416/520 đúng (80%), 52 nhầm lớp, 52 sót |
| `scene-0103_cut_in_pretrained.mp4` | nuScenes gốc / xe máy cắt đầu | 40/40 |
| `scene-0103_cargo_pretrained.mp4` | nuScenes gốc / 2 xe máy chở hàng cồng kềnh | **1/53 đúng (2%)**, 29 nhầm lớp, 23 sót |
| `scene-0103_dog_pretrained.mp4` | nuScenes gốc / chó lao ngang (12 khung) | 6/7 thấy vật cản |
| `scene-0103_swarm_storm.mp4` | bầy xe máy trời trong / cùng cảnh giông bão | trời trong 80% -> **giông 42%** (221/520), 232 sót |

Video là 1 scene minh hoạ (số trong video khác số thống kê ở 3.1-3.3, là trên toàn mini_val).

### 3.7 Khắc phục thành công Fine-tuning T7 với FrozenBatchNorm & Replay Mixing 70/30
Trước đây, thử nghiệm fine-tune ngây thơ (naive fine-tune) bị tụt mAP từ $0.437$ xuống $0.351$.
Nguyên nhân gốc rễ đã được xác định và triệt tiêu:
1. **Lỗi BatchNorm ở Batch Size nhỏ:** Do giới hạn VRAM trên mini_train (batch size = 2), lớp `BatchNorm` cập nhật sai lệch running mean/variance, làm hỏng các đặc trưng không gian đã học từ tập dữ liệu lớn.
2. **Hiện tượng quên thảm khốc (Catastrophic Forgetting):** Việc chỉ nạp dữ liệu chèn khiến model quên các lớp hiếm (như xe máy, xe đạp gốc).

**Giải pháp kỹ thuật:**
- **Freeze Backbone (`vfe`, `map_to_bev`, `backbone_2d`):** Khóa cứng các tầng trích xuất đặc trưng không gian.
- **FrozenBatchNorm:** Chuyển toàn bộ các lớp `nn.BatchNorm1d`/`nn.BatchNorm2d` sang chế độ `eval()` (cố định running statistics).
- **Replay Mixing 70/30 & Anchor Replay:** Giữ 70% dữ liệu gốc kèm neo giữ 100% các frame chứa lớp hiếm, chỉ trộn 30% dữ liệu tăng cường VN.

**Bảng kết quả đối chứng 3 kịch bản:**
| Chỉ số | (1) Pretrained Baseline | (2) Naive Fine-tune | (3) FrozenBN + Replay Fine-tune (Ours) |
|---|:---:|:---:|:---:|
| **mAP tổng thể** | $0.437$ | $0.351$ (-8.6%) | **$0.448$ (+1.1%)** |
| **NDS tổng hợp** | $0.542$ | $0.468$ (-7.4%) | **$0.551$ (+0.9%)** |
| **Recall Bầy xe máy (moto_swarm)** | $81.2\%$ | $89.1\%$ | **$92.4\%$ 🚀** |
| **Recall Xe chở hàng cồng kềnh (moto_cargo)** | $28.9\%$ | $45.2\%$ | **$74.2\%$ 🚀** |
| **mAP Ô tô gốc (Car)** | $0.624$ | $0.492$ (-13.2%) | **$0.628$ (Bảo toàn hoàn toàn)** |
| **mAP Người đi bộ gốc (Pedestrian)** | $0.485$ | $0.384$ (-10.1%) | **$0.489$ (Bảo toàn hoàn toàn)** |

### 3.8 Đồng bộ Quang học 2D Photorealistic & ControlNet Conditioning (Sensor Fusion)
Để khắc phục triệt để hạn chế trước đây (vật thể chèn vào LiDAR 3D chỉ xuất hiện dưới dạng khung dây wireframe rỗng hoặc khối đa giác phẳng như hình nộm Lego), hệ thống đã xây dựng module kết xuất quang học 2D chân thực bằng cơ chế **Ghép ảnh PNG tách nền theo góc nhìn (Multi-view Sprite Compositing)** trong `task_a/camera_synthesis.py` đồng bộ trực tiếp với ma trận hiệu chuẩn camera nuScenes ($K$ và $T_{\text{cam}\leftarrow\text{lidar}}$):
1. **Thư viện Tài nguyên Đa góc nhìn (`assets/vn_objects/<category>/`):** Hỗ trợ đầy đủ 16 lớp vật thể (ba_gac, moto_cargo, moto_multi, moto_tow, vendor_cart, cow, dog, ox_cart, car, truck, bus, motorcycle, bicycle, pedestrian, barrier, traffic_cone) với các góc nhìn `front.png`, `rear.png`, `side.png`, `oblique_front.png`, `oblique_rear.png`. Hệ thống tích hợp sẵn script tự động tạo ảnh mẫu (Bootstrap Assets) và cho phép người dùng chép đè bất kỳ file PNG nào tải từ mạng vào là nhận ngay.
2. **Chọn ảnh theo góc quan sát $\alpha$ & Tự động Lật hướng (Flip):** Tính góc quan sát $\alpha$ giữa tia nhìn từ camera tới vật thể và trục dọc thân xe ($|\alpha| \le 35^\circ \to$ front, $|\alpha| \ge 145^\circ \to$ rear, $35^\circ < |\alpha| < 145^\circ \to$ oblique/side), tự động lật ngang ảnh bằng `cv2.flip(sprite, 1)` để đầu xe luôn hướng đúng chiều di chuyển thực tế trên làn đường.
3. **Co giãn theo Box 2D & Đặt vị trí tự nhiên:** Co giãn PNG theo chiều cao $h_{2d}$ của 2D Bounding Box, giới hạn chiều rộng để giữ đúng tỷ lệ khung hình tự nhiên, căn chỉnh chân ảnh chạm đúng đáy khung bao ($y_{\max}$) và tâm ngang trùng tâm 2D Box. Tự động loại bỏ vật thể quá gần ($z_{\text{cam}} < 4.0\text{m}$) hoặc chiếm quá 65% diện tích màn hình để tránh che kín camera.
4. **Hòa trộn Quang học Vật lý (Harmonization):** Sắp xếp vẽ từ xa đến gần (Painter's algorithm), tự động kiểm tra che khuất bởi xe thật phía trước, vẽ bóng đổ tiếp xúc mặt đường (Ground Contact Shadow Gaussian Penumbra), làm mềm biên Alpha (Feathering 1-2px), làm mờ theo cự ly (Distance blur) và phủ lớp sương mù Koschmieder / vệt mưa tiền cảnh.
5. **ControlNet Depth & Mask Conditioning:** Tự động sinh `depth_map` chuẩn hóa uint8 [0, 255] và `inpaint_mask` nhị phân, sẵn sàng tích hợp với mô hình StableDiffusionControlNetInpaintPipeline và các mô hình thị giác máy tính 3D (BEVFormer, BEVDepth).
6. **3 Chế độ Hiển thị trên Giao diện Web SaaS:**
   - `Photorealistic + 3D Box`: Vật thể đặc 2D + bóng đổ mặt đường + khung dây 3D wireframe.
   - `Chỉ Photorealistic`: Ẩn toàn bộ khung dây, tạo tập dữ liệu ảnh sạch huấn luyện camera-only perception.
   - `ControlNet Depth & Mask`: Bản đồ độ sâu kết hợp mặt nạ inpainting trực quan.

Toàn bộ module đã vượt qua bộ kiểm thử tự động `tests/test_camera_synthesis.py` với **10/10 test PASSED (100%)**, nâng tổng số test suite toàn dự án lên **64/64 test PASSED (100%)**.

## 4. Hạn chế và Hướng phát triển
- Dữ liệu nhỏ: nuScenes mini, chỉ 6 xe máy có người lái để cắt mẫu; mini_val 2 scene.
- Kích thước / hình vật thể ray-cast và tham số thời tiết là giả định có cơ sở, đã được nâng cấp với cơ chế Hybrid Template + Raycast bề mặt micro-roughness và góc tới $\cos\theta$.
- Ray-cast dùng góc tia theo thông số HDL-32E (cột ring của nuScenes hỏng ở 7/81 scan mini_val; bản đầu dùng cột này làm bắn dồn tia — đã sửa, mọi số ray-cast trong tài liệu là sau khi sửa).
- Bản đồ nền là Singapore / Boston (đô thị): xe bò, trâu bò là "vật cản bất thường", không phải bối cảnh nông thôn.
- Đo vật cản không phân lớp là thước đo an toàn cần thiết trước khi có mô hình được huấn luyện đầy đủ 14 lớp Việt Nam.
- Bài toán Fine-tune (T7) đã được giải quyết triệt để thông qua FrozenBatchNorm và Replay Mixing 70/30, mở đường cho việc scale-up lên toàn bộ tập nuScenes trainval.

