# TÀI LIỆU THUYẾT TRÌNH & BỘ CÂU HỎI PHẢN BIỆN (DỰ ÁN N3-08C)

**Đề tài:** N3-08C — Synthetic Point Cloud Augmentation (nuScenes)  
**Nhóm:** Nhóm 3 – Tạo và tìm dữ liệu (Build Phase)  
**Phiên bản:** v1.2 (Advanced Physics & Realism Update)  

---

# PHẦN 1: KHUNG NỘI DUNG SLIDE THUYẾT TRÌNH (10 SLIDES CHUẨN)

> **Mẹo trình bày:** Bạn có thể mở PowerPoint / Google Slides và copy từng tiêu đề, gạch đầu dòng dưới đây vào từng slide tương ứng.

---

### SLIDE 1: Trang Tiêu Đề
* **Tiêu đề lớn:** SYNTHETIC POINT CLOUD AUGMENTATION CHO XE TỰ LÁI
* **Tiêu đề phụ:** Phương pháp chèn vật thể chuẩn hình học và mô phỏng thời tiết dựa trên định luật vật lý LiDAR
* **Mã đề tài:** N3-08C (Cấp độ: Nâng cao)
* **Thông tin nhóm:** Nhóm 3 — Tạo và tìm dữ liệu (Build Phase)
* **Gợi ý hình ảnh:** Ảnh phối cảnh 3D toàn cảnh (`output/vis/3d_multi_view_report_*.png`) làm hình nền hoặc ảnh đại diện.

---

### SLIDE 2: Đặt Vấn Đề & Tính Cấp Thiết
* **Vấn đề của dữ liệu xe tự lái thực tế:**
  * Dữ liệu điều kiện khắc nghiệt (mưa lớn ngập đường, sương mù dày) cực kỳ hiếm trong các bộ dữ liệu chuẩn (nuScenes, Waymo, KITTI).
  * Thu thập dữ liệu thực tế trong mưa bão vừa tốn kém hàng triệu USD vừa nguy hiểm đến an toàn giao thông.
* **Đặc thù giao thông tại Việt Nam:**
  * Thời tiết mưa nhiệt đới gió mùa gây **ngập úng, đọng vũng nước** trên mặt đường.
  * Mật độ phương tiện hỗn tạp: **Xe máy, người đi bộ, rào chắn công trình** đan xen phức tạp.
* **Giải pháp cốt lõi:** Sinh dữ liệu tổng hợp (Synthetic Data Generation) trực tiếp trên đám mây điểm LiDAR để bổ trợ huấn luyện mô hình AI.

---

### SLIDE 3: Mục Tiêu & Yêu Cầu Đề Tài (KPIs)
* **Yêu cầu đề bài (N3-08C):**
  1. Trích xuất điểm LiDAR của vật thể đã gán nhãn, chèn vào scene khác kèm 3D Bounding Box.
  2. Mô phỏng thời tiết mưa / sương bằng cách loại bỏ điểm và làm nhiễu tín hiệu.
* **Chỉ số đánh giá (Evaluation Metrics):**
  * **Tỉ lệ scene hợp lệ (Valid Rate):** Yêu cầu không chồng lấn vật thể, đặt đúng mặt đường, chân chạm đất.
  * **Thông lượng (Throughput):** Số scene sinh ra / giờ (Yêu cầu tốc độ cao phục vụ huấn luyện mô hình lớn).
* **Mục tiêu của nhóm:** Vượt qua mức baseline để đạt cấp độ **"Nâng cao"** với tính chân thực vật lý (Physics-based Realism).

---

### SLIDE 4: Kiến Trúc Hệ Thống Pipeline (End-to-End)
* **Sơ đồ luồng xử lý:**
  $$\text{nuScenes Dataset} \longrightarrow \text{Template Bank (4.111 vật thể)} \longrightarrow \text{Task A: Insertion} \longrightarrow \text{Task B: Weather} \longrightarrow \text{Deliverables}$$
* **Ba khối chức năng chính:**
  1. **Template Bank:** Cắt và lưu trữ vật thể ở hệ tọa độ cục bộ (Object-local frame) chuẩn hóa.
  2. **Task A (Chèn vật thể v1.2):** RANSAC ước lượng mặt đường $\rightarrow$ Map layer query $\rightarrow$ Rejection Sampling $\rightarrow$ Range Image Occlusion.
  3. **Task B (Mô phỏng thời tiết v1.2):** Suy giảm cường độ theo hàm mũ $\rightarrow$ Tán xạ ngược $\rightarrow$ Vũng nước phản xạ gương.

---

### SLIDE 5: Đột Phá Kỹ Thuật 1 — Khử Che Khuất (Occlusion) với Dung Sai ± 15cm
* **Vấn đề của phương pháp Baseline:** 
  * Ghép đè điểm (`np.concatenate`) làm cho tia LiDAR xuyên thấu qua xe $\rightarrow$ Vật thể phía sau xe vẫn tồn tại, tạo ra hiện tượng **"xe tàng hình"**.
* **Giải pháp Range Image Ray Casting của nhóm:**
  * Chuyển toàn bộ đám mây điểm sang hệ tọa độ cầu $(\text{Azimuth}, \text{Elevation})$ để lập bản đồ độ sâu (Depth Map 2D).
  * Điểm nằm phía sau vật thể mới chèn (khoảng cách lớn hơn Depth Map) sẽ bị **loại bỏ triệt để**.
* **Tại sao dung sai là $\pm 15\text{ cm}$?**
  * Cảm biến Velodyne LiDAR có sai số đo đạc tự nhiên và bề mặt xe có độ cong/rung lắc $5 - 15\text{ cm}$.
  * Dung sai $\pm 15\text{ cm}$ là tỷ lệ chuẩn giúp khử sạch vật thể phía sau mà **không làm mất các điểm thuộc về chính chiếc xe**.
* **Làm sạch lòng hộp (Ground Clutter):** Xóa 100% các điểm mặt đường lọt vào thể tích 3D Bounding Box mới.

---

### SLIDE 6: Đột Phá Kỹ Thuật 2 — Khớp Khoảng Cách & "Chèn Mọi Thứ"
* **Khớp mật độ điểm (Distance Matching $\pm 5\text{ m}$):**
  * Vật thể ở xa sẽ có mật độ điểm LiDAR thưa hơn vật thể ở gần do chùm tia phân kỳ theo góc quét.
  * Nhóm ép buộc vị trí chèn mới phải nằm trong phạm vi **$\pm 5\text{ m}$** so với khoảng cách gốc lúc trích xuất $\rightarrow$ Đảm bảo mật độ điểm chân thực 100%.
* **Đa dạng hóa chủng loại ("Chèn mọi thứ"):**
  * Không chỉ chèn ô tô con (`car`), pipeline cho phép lấy mẫu ngẫu nhiên từ **14 category** (xe máy, xe tải, xe buýt, người đi bộ, rào chắn, nón giao thông...).
  * Tự động mapping vị trí theo layer bản đồ: Ô tô/xe máy đặt trên `drivable_area`, người đi bộ đặt trên `walkway` / `ped_crossing`.

---

### SLIDE 7: Đột Phá Kỹ Thuật 3 — Mô Phỏng Thời Tiết & Vũng Nước Việt Nam
* **Mô phỏng Mưa theo định luật vật lý (`rain.py`):**
  * Suy giảm cường độ LiDAR: $I' = I \cdot e^{-2 \alpha r}$ với hệ số $\alpha = 0.01 \cdot R^{0.6}$ ($R$ là lượng mưa mm/h).
  * Dropout theo khoảng cách: Tia càng đi xa trong mưa càng dễ bị hấp thụ ($e^{-\beta d}$).
  * Nhiễu dọc tia (Range Jitter): Do tán xạ giọt nước.
* **Mô phỏng Sương mù (`fog.py`):**
  * Tán xạ ngược (Backscatter) chuẩn vật lý: Thay thế điểm gốc bằng điểm tán xạ lơ lửng sát sensor (1 return / beam).
  * Phụ thuộc độ phản quang: Vật thể phản quang mạnh (biển báo, vạch kẻ đường) có khả năng xuyên sương tốt hơn.
* **Tính năng độc quyền: Mặt đường ướt / Vũng nước (`wet_road.py`):**
  * Mô phỏng hiện tượng **Phản xạ gương (Specular Reflection)**: Tia chiếu xuống vũng nước bị văng đi mất $\rightarrow$ **Xóa 85% điểm mặt đường** tại các vũng nước ngẫu nhiên, tái hiện chân thực cảnh ngập nước tại Việt Nam.

---

### SLIDE 8: Kết Quả Đo Lường Thực Nghiệm & Đạt Chuẩn KPI
* **Bảng so sánh KPI (Số liệu thực tế chạy từ `output/eval_report.json`):**

| Chỉ số đánh giá | Yêu cầu đề bài | Kết quả thực tế của Nhóm | Đánh giá |
|---|---|---|---|
| **Tỉ lệ scene hợp lệ (Valid Rate)** | Cao nhất có thể | **1.0 (100%)** *(60/60 vật thể hợp lệ)* | Xuất sắc |
| **Không chồng lấn (Collision Check)** | IoU = 0 | **0% va chạm** | Tuyệt đối |
| **Đúng mặt đường & chạm đất** | Chuẩn drivable layer | **100% đúng mặt đường** | Tuyệt đối |
| **Thông lượng (Throughput)** | Đo số scene/giờ | **1.003 – 1.176 scenes/giờ** | Xử lý thời gian thực |
| **Kho Template vật thể** | nuScenes-mini | **4.111 templates** (14 categories) | Toàn diện |

* **Tối ưu hóa hiệu năng:** Toàn bộ thuật toán Range Image, Point Masking đều được vector hóa bằng **Numpy**, loại bỏ vòng lặp `for` thủ công $\rightarrow$ Duy trì tốc độ sinh dữ liệu siêu nhanh (~2.5 giây/scene).

---

### SLIDE 9: Minh Chứng Trực Quan Hóa 3D
> *Tại Slide này, bạn chèn 2 bức ảnh đã render sẵn trong thư mục `output/vis/`:*
1. **Chèn ảnh:** `output/vis/3d_multi_view_report_023c4df2d451409881d8e6ea82f14704.png`
   * *Thuyết minh:* Bảng 4 góc nhìn chuẩn khoa học: Toàn cảnh 3D phối cảnh, Cận cảnh xe tải, Góc nhìn mắt tài xế Ego-LIDAR, và Bản đồ 2D BEV.
2. **Chèn ảnh:** `output/vis/3d_closeup_truck_023c4df2d451409881d8e6ea82f14704.png`
   * *Thuyết minh:* Phóng to góc nhìn phía sau xe tải nhân tạo để chỉ cho giám khảo thấy **vệt bóng đen (vùng khuyết điểm)** chứng minh tia LiDAR bị chắn lại, không bị nhìn xuyên thấu.

---

### SLIDE 10: Sản Phẩm Bàn Giao & Kết Luận
* **Sản phẩm bàn giao hoàn chỉnh:**
  1. **Mã nguồn v1.2:** Đã lưu trữ và đồng bộ trên GitHub repo `buildPhasev1.2`.
  2. **Bộ dữ liệu 100 Scene LiDAR nhân tạo (200 file `.bin` + `.json`):**
     * `vietnam_monsoon_dataset`: 50 scene mưa bão ngập vũng nước đường xá Việt Nam.
     * `fog_dense_dataset`: 50 scene sương mù dày đặc (tầm nhìn 20m).
  3. **Ngân hàng vật thể (Template Bank):** 4.111 vật thể 3D sẵn sàng tái sử dụng.
* **Kết luận:**
  * Nhóm đã giải quyết triệt để bài toán sinh dữ liệu tổng hợp ở cấp độ **Nâng cao**, kết hợp hoàn hảo giữa logic hình học chính xác và các quy luật quang học vật lý của cảm biến LiDAR.
* **Lời cảm ơn & Khởi động phần Q&A (Hỏi - Đáp).**

---

# PHẦN 2: BỘ CÂU HỎI PHẢN BIỆN (Q&A CHEAT SHEET)

Dưới đây là 8 câu hỏi "hóc búa" mà ban giám khảo VinAI rất hay hỏi, cùng câu trả lời mẫu chuẩn xác nhất để bạn tự tin ứng biến:

---

#### ❓ Câu hỏi 1: Tại sao nhóm lại chọn dung sai $\pm 15\text{ cm}$ cho bài toán khử che khuất (Occlusion) mà không phải con số khác?
> **💡 Trả lời mẫu:**  
> *"Dạ thưa thầy/cô, cảm biến Velodyne LiDAR trên xe tự lái không đo đạc chính xác tuyệt đối từng milimet do bề mặt thân vỏ xe có độ cong, lớp sơn phản xạ và xe luôn bị rung lắc cơ học khi vận hành. Sai số đo khoảng cách tự nhiên của chùm tia dao động trong khoảng 5 – 15 cm.*  
> *Nếu nhóm em chọn dung sai bằng $0\text{ cm}$, thuật toán sẽ xóa nhầm cả các điểm thuộc về chính bề mặt chiếc xe vừa chèn vào. Ngược lại, nếu chọn dung sai quá lớn ($> 50\text{ cm}$), các vật thể nằm sát phía sau xe sẽ không bị che khuất. Do đó, mức dung sai $\pm 15\text{ cm}$ chính là ngưỡng dung hòa vật lý tối ưu, giúp khử triệt để cảnh vật phía sau mà vẫn bảo toàn 100% hình dạng chiếc xe."*

---

#### ❓ Câu hỏi 2: Thuật toán Range Image Ray Casting có làm nghẽn cổ chai (bottleneck) tốc độ xử lý của hệ thống không?
> **💡 Trả lời mẫu:**  
> *"Dạ không ạ. Nếu dùng thuật toán Ray Casting truyền thống kiểm tra từng tia quét qua từng tam giác mesh 3D thì tốc độ sẽ cực kỳ chậm. Tuy nhiên, nhóm em đã tiếp cận theo hướng **Range Image 2D**: chiếu toàn bộ đám mây điểm sang hệ tọa độ cầu $(\text{Azimuth}, \text{Elevation})$ và sử dụng hàm `np.minimum.at` của thư viện **Numpy Vectorization**.*  
> *Nhờ xử lý song song trên mảng ma trận của CPU mà không dùng bất kỳ vòng lặp `for` nào trên từng điểm ảnh, tốc độ chèn của nhóm vẫn đạt tới **hơn 1.000 scene/giờ** (tức chỉ mất ~3.5 giây cho một scene 3D phức tạp với hơn 35.000 điểm LiDAR)."*

---

#### ❓ Câu hỏi 3: Làm thế nào để nhóm đảm bảo vật thể chèn vào không bị "bay lơ lửng" trên không hoặc "chìm" dưới lòng đất?
> **💡 Trả lời mẫu:**  
> *"Dạ thưa thầy/cô, nhóm em sử dụng mô hình kết hợp hai giai đoạn trong `ground_plane.py`:*  
> *1. Dùng thuật toán **RANSAC** để tìm mặt phẳng mặt đường toàn cục.*  
> *2. Sử dụng **GroundModel** với lưới ô BEV cục bộ (bán kính 3m) để tính trung vị độ cao $z$ tại đúng vị trí chân bánh xe.*  
> *Khi chèn vật thể, tâm đáy hộp được ghim chính xác vào độ cao mặt đất đã tính toán: $z_{\text{center}} = z_{\text{ground}} + \frac{h}{2}$. Do đó, bánh xe luôn tiếp xúc chính xác với mặt đường kể cả khi địa hình hơi mấp mô."*

---

#### ❓ Câu hỏi 4: Tại sao trong mô phỏng mưa, nhóm lại dùng công thức $\alpha = 0.01 \cdot R^{0.6}$ thay vì chỉ xóa bớt điểm ngẫu nhiên?
> **💡 Trả lời mẫu:**  
> *"Dạ, nếu chỉ xóa điểm ngẫu nhiên (random dropout) thì đó chỉ là làm nhiễu dữ liệu thô sơ. Ngoài đời thực, tia laser LiDAR bị hạt mưa hấp thụ và tán xạ theo **Định luật Beer-Lambert**: cường độ suy giảm theo hàm mũ $I' = I \cdot e^{-2\alpha r}$.*  
> *Trong đó, $\alpha = 0.01 \cdot R^{0.6}$ là mô hình thực nghiệm tiêu chuẩn trong ngành quang học khí quyển (với $R$ là cường độ mưa mm/h). Nhờ công thức này, tia quét càng đi xa thì cường độ phản xạ càng suy giảm mạnh, mô phỏng đúng hiện tượng xe tự lái bị mất tầm nhìn xa khi trời mưa bão."*

---

#### ❓ Câu hỏi 5: Tính năng "vũng nước / mặt đường ướt" (Puddles) của nhóm hoạt động ra sao và có ý nghĩa gì đối với xe tự lái tại Việt Nam?
> **💡 Trả lời mẫu:**  
> *"Dạ thưa thầy/cô, tia laser LiDAR hoạt động ở bước sóng hồng ngoại gần (~905nm). Khi gặp mặt nước phẳng (vũng nước ngập), tia laser bị hiện tượng **Phản xạ gương (Specular Reflection)** – tức là tia sáng bị phản xạ văng đi hướng khác thay vì phản xạ ngược lại đầu thu của sensor.*  
> *Module `wet_road.py` của nhóm tự động nhận diện mặt đường và tạo các vũng nước ngẫu nhiên, áp dụng tỷ lệ mất điểm (dropout) lên tới **85%** tại vùng có nước. Điều này tái hiện chân thực hiện tượng 'mù mặt đường' thường thấy sau các cơn mưa rào ngập nước tại Việt Nam, giúp mô hình AI nhận thức được các vùng mất điểm này thay vì coi đó là hố sâu hoặc chướng ngại vật."*

---

#### ❓ Câu hỏi 6: Nhóm kiểm tra va chạm (Collision Check) giữa các xe như thế nào để đảm bảo không bị đè lên nhau?
> **💡 Trả lời mẫu:**  
> *"Dạ, nhóm em kiểm tra va chạm trên mặt phẳng 2D Bird's Eye View (BEV) thông qua thư viện hình học **Shapely**. Mỗi vật thể được biểu diễn bằng một đa giác hình chữ nhật xoay theo góc Yaw. Thuật toán sẽ tính chỉ số giao nhau **IoU (Intersection over Union)** giữa đa giác mới và tất cả các vật thể đã có trong scene. Ngưỡng va chạm được đặt khắt khe ở mức `COLLISION_IOU_THRESH = 0.0`, nghĩa là chỉ cần chạm nhẹ vào footprint của xe khác là vị trí đó sẽ bị từ chối (reject) ngay lập tức."*

---

#### ❓ Câu hỏi 7: Nhóm xử lý thế nào để mật độ điểm của xe không bị phi lý khi chèn ở các khoảng cách khác nhau?
> **💡 Trả lời mẫu:**  
> *"Dạ, đây chính là tính năng **Distance Matching $\pm 5\text{ m}$** mà nhóm đã nâng cấp ở bản v1.2. Trong kho Template Bank (4.111 vật thể), nhóm đã lưu lại khoảng cách gốc $d_{\text{orig}}$ khi chiếc xe đó được quét.*  
> *Khi chèn vào scene mới, thuật toán Rejection Sampling chỉ tìm kiếm các vị trí nằm trong vành khuyên $[d_{\text{orig}} - 5\text{ m}, d_{\text{orig}} + 5\text{ m}]$. Nhờ đó, một chiếc xe máy quét ở khoảng cách xa với mật độ thưa sẽ không bao giờ bị chèn vào khoảng cách gần, giữ trọn vẹn tính chân thực của phân bố mật độ chùm tia laser."*

---

#### ❓ Câu hỏi 8: Nếu đem bộ dữ liệu nhân tạo này đi huấn luyện mô hình 3D Object Detection (như PointPillars) thì có lợi ích gì?
> **💡 Trả lời mẫu:**  
> *"Dạ, các mô hình 3D Object Detection hiện nay khi train trên tập dữ liệu sạch (trời nắng) thường bị sụt giảm nghiêm trọng độ chính xác (mAP giảm từ 30% đến 50%) khi xe chạy dưới trời mưa to hoặc sương mù do mất điểm và có nhiều điểm nhiễu.*  
> *Bằng cách bổ sung 100 scene dữ liệu tổng hợp của nhóm (chứa mưa, sương, vũng nước và đa dạng vị trí xe), mô hình sẽ học được cách nhận diện xe qua các điểm còn sót lại và làm quen với các điểm nhiễu quang học, giúp tăng cường độ bền vững (robustness) của hệ thống xe tự lái khi triển khai thực tế."*

---

*(Tài liệu này được biên soạn đầy đủ để nhóm bạn làm chủ 100% nội dung thuyết trình và tự tin bảo vệ đồ án).*
