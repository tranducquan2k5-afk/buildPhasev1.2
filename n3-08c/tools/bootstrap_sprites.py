"""Bootstrap PNG Sprites cho 2D Multi-view Camera Synthesis.

Tạo sẵn bộ ảnh PNG RGBA tách nền độ chi tiết cao cho 16 lớp vật thể (Việt Nam + chuẩn nuScenes):
ba_gac, moto_cargo, moto_multi, moto_tow, vendor_cart, cow, dog, ox_cart,
car, truck, bus, motorcycle, bicycle, pedestrian, barrier, traffic_cone.

Mỗi thư mục hỗ trợ: front.png, rear.png, side.png, oblique_front.png, oblique_rear.png.
Người dùng có thể chép đè bất kỳ ảnh PNG nào vào assets/vn_objects/<category>/
để thay thế hình ảnh tùy thích.
"""
from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def _create_canvas(size: Tuple[int, int]) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(img, "RGBA")
    return img, dr


def _add_wheel(dr: ImageDraw.ImageDraw, cx: int, cy: int, r: int, rim_color=(180, 185, 190)):
    """Vẽ bánh xe chi tiết (lốp đen, vành kim loại mạ, đĩa phanh và ốc)."""
    # Lốp cao su ngoài
    dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(25, 25, 28, 255), outline=(15, 15, 18, 255), width=2)
    # Vành mâm xe (rim)
    rim_r = int(r * 0.65)
    dr.ellipse([cx - rim_r, cy - rim_r, cx + rim_r, cy + rim_r], fill=rim_color + (255,), outline=(100, 105, 110, 255), width=1)
    # Trục tâm & nan hoa
    hub_r = max(2, int(r * 0.25))
    dr.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r], fill=(50, 50, 55, 255))
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):
        x1 = cx + int(hub_r * np.cos(ang))
        y1 = cy + int(hub_r * np.sin(ang))
        x2 = cx + int(rim_r * 0.9 * np.cos(ang))
        y2 = cy + int(rim_r * 0.9 * np.sin(ang))
        dr.line([(x1, y1), (x2, y2)], fill=(80, 85, 90, 255), width=max(1, int(r * 0.08)))


def draw_car(view: str) -> Image.Image:
    w, h = 480, 260
    img, dr = _create_canvas((w, h))
    body_col = (30, 80, 175)      # Xanh lam bóng
    body_dark = (20, 55, 125)
    glass_col = (130, 190, 230, 220)
    glass_line = (235, 245, 255, 180)

    if view == "side":
        # Thân xe con dáng Sedan nhìn ngang (đầu bên trái)
        # Gầm xe
        dr.rounded_rectangle([40, 140, 440, 215], radius=15, fill=body_col)
        # Nóc cabin
        dr.polygon([(110, 140), (160, 80), (320, 80), (370, 140)], fill=body_col)
        # Kính cửa sổ trước & sau
        dr.polygon([(165, 85), (235, 85), (235, 135), (125, 135)], fill=glass_col)
        dr.polygon([(245, 85), (315, 85), (355, 135), (245, 135)], fill=glass_col)
        dr.line([(150, 115), (230, 90)], fill=glass_line, width=2)
        # Cản trước & đèn pha
        dr.polygon([(40, 155), (60, 155), (55, 175), (40, 175)], fill=(255, 230, 90, 255)) # Đèn pha trước
        dr.polygon([(425, 155), (440, 155), (440, 175), (430, 175)], fill=(230, 40, 30, 255)) # Đèn hậu
        # Vòm hốc bánh xe & 2 bánh
        _add_wheel(dr, 125, 215, 38)
        _add_wheel(dr, 355, 215, 38)
        # Tay nắm cửa & gương
        dr.rectangle([195, 145, 215, 150], fill=(200, 205, 210, 255))
        dr.rectangle([275, 145, 295, 150], fill=(200, 205, 210, 255))
        dr.ellipse([145, 128, 160, 138], fill=body_dark + (255,))

    elif view == "front":
        # Nhìn chính diện đầu xe
        # Nóc kính lái
        dr.polygon([(140, 70), (340, 70), (380, 140), (100, 140)], fill=glass_col)
        dr.line([(130, 130), (280, 75)], fill=glass_line, width=3)
        # Thân xe chính
        dr.rounded_rectangle([90, 135, 390, 215], radius=20, fill=body_col)
        # Lưới tản nhiệt (Grille)
        dr.rounded_rectangle([170, 165, 310, 205], radius=8, fill=(35, 35, 40, 255), outline=(180, 185, 190, 255), width=2)
        # Đèn pha 2 bên
        dr.polygon([(110, 145), (160, 145), (155, 170), (115, 165)], fill=(255, 240, 140, 255))
        dr.polygon([(325, 145), (370, 145), (365, 165), (320, 170)], fill=(255, 240, 140, 255))
        # Biển số trước
        dr.rectangle([205, 195, 275, 212], fill=(245, 245, 245, 255), outline=(20, 20, 20, 255), width=1)
        # 2 bánh xe phía trước chạm đất
        _add_wheel(dr, 115, 222, 32)
        _add_wheel(dr, 365, 222, 32)
        # Gương chiếu hậu 2 bên
        dr.ellipse([70, 125, 95, 145], fill=body_dark + (255,))
        dr.ellipse([385, 125, 410, 145], fill=body_dark + (255,))

    elif view == "rear":
        # Nhìn từ sau đuôi xe
        dr.polygon([(145, 75), (335, 75), (375, 140), (105, 140)], fill=glass_col)
        dr.rounded_rectangle([90, 135, 390, 215], radius=20, fill=body_col)
        # Đèn hậu đỏ 2 bên
        dr.polygon([(105, 145), (165, 145), (160, 170), (110, 170)], fill=(225, 25, 20, 255))
        dr.polygon([(315, 145), (375, 145), (370, 170), (320, 170)], fill=(225, 25, 20, 255))
        # Biển số vàng/trắng & cản sau
        dr.rectangle([200, 175, 280, 198], fill=(250, 250, 250, 255), outline=(30, 30, 30, 255), width=1)
        # Ống xả kép
        dr.ellipse([110, 210, 126, 218], fill=(130, 135, 140, 255))
        dr.ellipse([354, 210, 370, 218], fill=(130, 135, 140, 255))
        _add_wheel(dr, 115, 222, 32)
        _add_wheel(dr, 365, 222, 32)

    else:  # oblique_front / oblique_rear
        # Góc 3/4 phối cảnh
        dr.rounded_rectangle([70, 135, 410, 215], radius=18, fill=body_col)
        dr.polygon([(130, 75), (300, 80), (350, 140), (95, 140)], fill=glass_col)
        dr.line([(120, 125), (280, 82)], fill=glass_line, width=2)
        if "rear" in view:
            dr.polygon([(360, 150), (405, 150), (400, 175), (355, 175)], fill=(225, 30, 25, 255))
        else:
            dr.polygon([(75, 150), (130, 150), (125, 175), (80, 175)], fill=(255, 240, 140, 255))
        _add_wheel(dr, 135, 218, 36)
        _add_wheel(dr, 345, 218, 36)

    return img


def draw_truck(view: str) -> Image.Image:
    w, h = 540, 310
    img, dr = _create_canvas((w, h))
    cab_col = (195, 45, 35)       # Đỏ cam xe tải
    bed_col = (110, 120, 130)      # Thùng xe xám thép
    tarp_col = (45, 95, 145)      # Bạt phủ xanh dương

    if view == "side":
        # Đầu cabin bên trái
        dr.rounded_rectangle([45, 110, 165, 250], radius=12, fill=cab_col)
        dr.polygon([(55, 115), (145, 115), (145, 175), (55, 175)], fill=(140, 200, 240, 230)) # Kính cabin
        # Thùng hàng lớn phía sau
        dr.rectangle([170, 75, 495, 245], fill=bed_col)
        dr.rectangle([170, 65, 495, 185], fill=tarp_col) # Bạt phủ nóc
        for x in range(180, 490, 45): # Gân thùng xe
            dr.line([(x, 65), (x, 245)], fill=(30, 65, 100, 200), width=2)
        # Đèn pha & cản trước
        dr.rectangle([45, 205, 60, 230], fill=(255, 235, 120, 255))
        # 3 cụm bánh xe tải
        _add_wheel(dr, 110, 255, 42, rim_color=(120, 125, 130))
        _add_wheel(dr, 350, 255, 42, rim_color=(120, 125, 130))
        _add_wheel(dr, 440, 255, 42, rim_color=(120, 125, 130))
        # Bình nhiên liệu
        dr.rectangle([190, 230, 290, 255], fill=(70, 75, 80, 255))

    elif view == "front":
        # Mặt trước xe tải
        dr.rounded_rectangle([130, 50, 410, 250], radius=18, fill=cab_col)
        dr.rectangle([155, 65, 385, 145], fill=(140, 200, 240, 230)) # Kính lái to
        # Lưới tản nhiệt lớn
        dr.rectangle([175, 160, 365, 220], fill=(40, 40, 45, 255), outline=(210, 215, 220, 255), width=2)
        dr.rectangle([140, 180, 170, 215], fill=(255, 235, 100, 255)) # Đèn trái
        dr.rectangle([370, 180, 400, 215], fill=(255, 235, 100, 255)) # Đèn phải
        _add_wheel(dr, 145, 260, 40)
        _add_wheel(dr, 395, 260, 40)
        # Gương chiếu hậu lớn 2 bên
        dr.rectangle([100, 90, 125, 165], fill=(40, 40, 45, 255))
        dr.rectangle([415, 90, 440, 165], fill=(40, 40, 45, 255))

    elif view == "rear":
        # Đuôi thùng xe tải
        dr.rectangle([110, 50, 430, 250], fill=bed_col)
        dr.rectangle([110, 45, 430, 150], fill=tarp_col)
        # Cửa thùng & then cài thanh kim loại
        dr.line([(270, 50), (270, 250)], fill=(40, 45, 50, 255), width=3)
        dr.line([(190, 60), (190, 240)], fill=(180, 185, 190, 255), width=4)
        dr.line([(350, 60), (350, 240)], fill=(180, 185, 190, 255), width=4)
        # Đèn hậu & chắn bùn
        dr.rectangle([125, 235, 185, 252], fill=(230, 30, 20, 255))
        dr.rectangle([355, 235, 415, 252], fill=(230, 30, 20, 255))
        _add_wheel(dr, 145, 260, 40)
        _add_wheel(dr, 395, 260, 40)

    else:
        # Oblique
        dr.rounded_rectangle([70, 80, 470, 250], radius=15, fill=cab_col)
        dr.rectangle([180, 60, 470, 175], fill=tarp_col)
        _add_wheel(dr, 140, 255, 40)
        _add_wheel(dr, 390, 255, 40)

    return img


def draw_bus(view: str) -> Image.Image:
    w, h = 600, 290
    img, dr = _create_canvas((w, h))
    bus_col = (235, 160, 20)      # Vàng cam xe buýt
    stripe = (35, 75, 145)

    if view == "side":
        dr.rounded_rectangle([40, 55, 560, 230], radius=16, fill=bus_col)
        dr.rectangle([40, 160, 560, 180], fill=stripe)
        # Hàng kính hành khách
        for x in range(65, 520, 65):
            dr.rectangle([x, 70, x + 55, 135], fill=(150, 210, 245, 230))
        # Cửa lên xuống
        dr.rectangle([130, 135, 185, 225], fill=(70, 75, 80, 240), outline=(200, 200, 200, 255), width=2)
        _add_wheel(dr, 130, 235, 42)
        _add_wheel(dr, 470, 235, 42)

    elif view in ("front", "rear"):
        dr.rounded_rectangle([150, 45, 450, 240], radius=20, fill=bus_col)
        if view == "front":
            # Kính chắn gió lớn
            dr.rectangle([170, 65, 430, 150], fill=(150, 210, 245, 230))
            # Bảng lộ trình điện tử
            dr.rectangle([210, 50, 390, 63], fill=(20, 20, 25, 255))
            # Đèn pha
            dr.rectangle([170, 185, 210, 210], fill=(255, 240, 130, 255))
            dr.rectangle([390, 185, 430, 210], fill=(255, 240, 130, 255))
        else:
            dr.rectangle([180, 70, 420, 130], fill=(150, 210, 245, 230))
            dr.rectangle([170, 185, 210, 210], fill=(225, 25, 20, 255))
            dr.rectangle([390, 185, 430, 210], fill=(225, 25, 20, 255))
        _add_wheel(dr, 185, 245, 38)
        _add_wheel(dr, 415, 245, 38)
    else:
        dr.rounded_rectangle([80, 50, 520, 235], radius=18, fill=bus_col)
        dr.rectangle([120, 70, 480, 140], fill=(150, 210, 245, 230))
        _add_wheel(dr, 150, 240, 40)
        _add_wheel(dr, 440, 240, 40)

    return img


def draw_motorcycle_family(category: str, view: str) -> Image.Image:
    """Vẽ xe máy thường, xe chở hàng cồng kềnh, kẹp 2-3 người hoặc kéo xe."""
    w, h = 420, 360
    img, dr = _create_canvas((w, h))

    is_cargo = (category == "moto_cargo")
    is_multi = (category == "moto_multi")
    is_tow = (category == "moto_tow")

    body_col = (185, 30, 25)  # Đỏ Honda Wave
    rider_skin = (235, 185, 145)
    rider_cloth = (40, 75, 135)
    helmet_col = (225, 210, 35)

    if view == "side":
        # Khung sườn xe máy (đầu xe bên trái)
        dr.polygon([(110, 255), (175, 220), (250, 220), (285, 275), (195, 275)], fill=body_col)
        dr.line([(110, 260), (145, 175)], fill=(120, 125, 130, 255), width=5) # Phuộc trước & ghi-đông
        dr.rectangle([135, 170, 160, 180], fill=(20, 20, 25, 255)) # Tay lái
        # 2 bánh xe căm
        _add_wheel(dr, 110, 275, 40, rim_color=(210, 215, 220))
        _add_wheel(dr, 290, 275, 40, rim_color=(210, 215, 220))
        # Ống bô xả
        dr.line([(210, 280), (305, 270)], fill=(160, 165, 170, 255), width=6)

        # Người lái xe máy
        dr.ellipse([165, 95, 215, 145], fill=helmet_col + (255,)) # Mũ bảo hiểm
        dr.polygon([(170, 140), (225, 140), (235, 225), (165, 225)], fill=rider_cloth + (255,)) # Thân áo
        dr.line([(185, 165), (150, 175)], fill=rider_skin + (255,), width=6) # Cánh tay cầm lái

        if is_multi:
            # Người ngồi sau 1 & 2
            dr.ellipse([215, 105, 255, 145], fill=(220, 35, 30, 255)) # Mũ người thứ 2
            dr.polygon([(215, 145), (265, 145), (265, 225), (220, 225)], fill=(50, 130, 80, 255))
            dr.ellipse([255, 115, 290, 150], fill=(230, 230, 235, 255)) # Mũ người thứ 3

        if is_cargo:
            # KIỆN HÀNG CỒNG KỀNH VIỆT NAM (Rộng cao vượt trội)
            # Thùng các-tông và bao tải bọc bạt xanh
            dr.rectangle([230, 70, 385, 240], fill=(185, 140, 80, 255), outline=(130, 95, 50, 255), width=2)
            dr.rectangle([250, 40, 365, 110], fill=(40, 95, 155, 255), outline=(25, 65, 110, 255), width=2)
            # Dây chằng chun hoa văn chéo
            dr.line([(230, 70), (385, 240)], fill=(225, 40, 35, 255), width=3)
            dr.line([(385, 70), (230, 240)], fill=(225, 40, 35, 255), width=3)
            dr.line([(250, 40), (365, 240)], fill=(240, 220, 40, 255), width=2)

        if is_tow:
            # Rơ-moóc / xe lôi phía sau
            dr.line([(290, 260), (330, 270)], fill=(50, 50, 50, 255), width=5)
            dr.rectangle([330, 220, 410, 280], fill=(135, 95, 55, 255))
            _add_wheel(dr, 385, 285, 25)

    elif view == "front":
        # Nhìn chính diện xe máy
        _add_wheel(dr, 210, 285, 35)
        dr.polygon([(185, 240), (235, 240), (225, 280), (195, 280)], fill=body_col) # Chắn bùn
        dr.line([(160, 175), (260, 175)], fill=(40, 40, 45, 255), width=6) # Ghi-đông ngang
        dr.ellipse([195, 190, 225, 220], fill=(255, 245, 150, 255)) # Đèn pha tròn
        # Người lái
        dr.ellipse([190, 95, 230, 135], fill=helmet_col + (255,))
        dr.polygon([(180, 135), (240, 135), (230, 210), (190, 210)], fill=rider_cloth + (255,))

        if is_cargo:
            # Hàng cồng kềnh bè ngang 2 bên cánh
            dr.rectangle([90, 110, 180, 240], fill=(185, 140, 80, 255), outline=(120, 80, 40, 255), width=2)
            dr.rectangle([240, 110, 330, 240], fill=(185, 140, 80, 255), outline=(120, 80, 40, 255), width=2)
            dr.rectangle([130, 60, 290, 115], fill=(45, 105, 165, 255))

    elif view == "rear":
        # Nhìn từ sau xe máy
        _add_wheel(dr, 210, 285, 35)
        dr.rectangle([195, 250, 225, 265], fill=(225, 30, 25, 255)) # Đèn hậu đỏ
        dr.rectangle([195, 268, 225, 280], fill=(245, 245, 245, 255)) # Biển số
        # Lưng người lái
        dr.ellipse([190, 95, 230, 135], fill=helmet_col + (255,))
        dr.polygon([(175, 135), (245, 135), (235, 230), (185, 230)], fill=rider_cloth + (255,))

        if is_cargo:
            # Đuôi kiện hàng che kín toàn bộ phía sau
            dr.rectangle([85, 60, 335, 240], fill=(185, 140, 80, 255), outline=(120, 85, 45, 255), width=3)
            # Bó hàng cao & dây chằng
            dr.line([(85, 60), (335, 240)], fill=(225, 30, 25, 255), width=4)
            dr.line([(335, 60), (85, 240)], fill=(225, 30, 25, 255), width=4)

    else:
        # Góc chéo
        _add_wheel(dr, 150, 280, 36)
        _add_wheel(dr, 270, 280, 36)
        dr.polygon([(140, 250), (280, 250), (250, 220), (170, 220)], fill=body_col)
        dr.ellipse([180, 100, 225, 145], fill=helmet_col + (255,))
        dr.polygon([(175, 145), (235, 145), (240, 220), (170, 220)], fill=rider_cloth + (255,))
        if is_cargo:
            dr.rectangle([210, 70, 350, 230], fill=(185, 140, 80, 255), outline=(120, 85, 45, 255), width=2)

    return img


def draw_ba_gac(view: str) -> Image.Image:
    """Xe ba gác máy chở hàng Nam Định (đặc thù VN)."""
    w, h = 480, 320
    img, dr = _create_canvas((w, h))
    green_bed = (35, 115, 75)   # Xanh lá cây đặc trưng ba gác
    rail_col = (195, 205, 210)

    if view == "side":
        # Đầu xe máy phía trước bên trái
        dr.line([(95, 245), (135, 165)], fill=(120, 125, 130, 255), width=6)
        dr.ellipse([145, 95, 190, 140], fill=(225, 210, 40, 255)) # Người lái
        dr.polygon([(140, 140), (195, 140), (190, 225), (145, 225)], fill=(45, 80, 145, 255))
        # Bánh trước
        _add_wheel(dr, 95, 255, 38)
        # Thùng hàng lớn phía sau
        dr.rectangle([195, 130, 445, 250], fill=green_bed, outline=(25, 75, 50, 255), width=3)
        # Khung lan can sắt
        for y in (150, 180, 210):
            dr.line([(195, y), (445, y)], fill=rail_col + (255,), width=2)
        # Bánh sau lớn chịu tải
        _add_wheel(dr, 360, 255, 46, rim_color=(140, 145, 150))

    elif view == "front":
        _add_wheel(dr, 240, 255, 38)
        dr.ellipse([225, 180, 255, 210], fill=(255, 240, 120, 255)) # Đèn tròn
        dr.ellipse([220, 100, 260, 140], fill=(225, 210, 40, 255)) # Người lái
        # Thùng hàng bè rộng phía sau
        dr.rectangle([120, 145, 360, 245], fill=green_bed, outline=(20, 60, 40, 255), width=2)

    elif view == "rear":
        # Mặt sau thùng hàng
        dr.rectangle([120, 120, 360, 245], fill=green_bed, outline=(20, 60, 40, 255), width=3)
        dr.line([(120, 180), (360, 180)], fill=rail_col + (255,), width=3)
        # Xích khóa thùng & biển số
        dr.rectangle([215, 215, 265, 238], fill=(245, 245, 245, 255))
        _add_wheel(dr, 155, 255, 42)
        _add_wheel(dr, 325, 255, 42)
    else:
        dr.rectangle([160, 130, 420, 245], fill=green_bed)
        _add_wheel(dr, 120, 255, 36)
        _add_wheel(dr, 350, 255, 42)

    return img


def draw_vendor_cart(view: str) -> Image.Image:
    """Xe đẩy bán hàng rong / bánh mì / nước mía vỉa hè."""
    w, h = 380, 360
    img, dr = _create_canvas((w, h))

    wood_col = (165, 115, 65)
    glass_col = (160, 220, 255, 190)
    awning_red = (215, 40, 35)
    awning_white = (245, 245, 245)

    if view in ("side", "oblique_front", "oblique_rear"):
        # Thân tủ gỗ
        dr.rectangle([80, 160, 300, 290], fill=wood_col, outline=(100, 65, 35, 255), width=3)
        # Tủ kính trưng bày
        dr.rectangle([95, 90, 285, 160], fill=glass_col, outline=(120, 170, 200, 255), width=2)
        # Mái che bạt sọc đỏ trắng
        for i, x in enumerate(range(70, 310, 24)):
            c = awning_red if i % 2 == 0 else awning_white
            dr.rectangle([x, 55, x + 24, 90], fill=c)
        # Tay đẩy
        dr.line([(60, 180), (80, 190)], fill=(80, 85, 90, 255), width=5)
        # 4 bánh xe đẩy nhỏ
        _add_wheel(dr, 120, 305, 22)
        _add_wheel(dr, 260, 305, 22)

    else:  # front / rear
        dr.rectangle([100, 150, 280, 290], fill=wood_col, outline=(100, 65, 35, 255), width=3)
        dr.rectangle([110, 90, 270, 150], fill=glass_col)
        for i, x in enumerate(range(95, 285, 20)):
            c = awning_red if i % 2 == 0 else awning_white
            dr.rectangle([x, 55, x + 20, 90], fill=c)
        _add_wheel(dr, 130, 305, 22)
        _add_wheel(dr, 250, 305, 22)

    return img


def draw_pedestrian(view: str) -> Image.Image:
    w, h = 260, 420
    img, dr = _create_canvas((w, h))
    skin = (235, 185, 145)
    shirt = (45, 95, 165)
    pants = (40, 45, 55)

    if view == "side":
        # Đầu & tóc
        dr.ellipse([110, 45, 150, 95], fill=skin)
        dr.arc([108, 42, 145, 75], start=160, end=360, fill=(30, 25, 20, 255), width=8)
        # Thân áo
        dr.polygon([(115, 95), (150, 95), (155, 220), (110, 220)], fill=shirt)
        # Hai chân bước sải
        dr.line([(125, 220), (95, 360)], fill=pants, width=16)  # Chân trước
        dr.line([(140, 220), (175, 360)], fill=pants, width=16) # Chân sau
        # Giày
        dr.rectangle([80, 355, 110, 375], fill=(30, 30, 35, 255))
        dr.rectangle([165, 355, 195, 375], fill=(30, 30, 35, 255))

    elif view in ("front", "rear"):
        dr.ellipse([105, 45, 155, 95], fill=skin if view == "front" else (40, 30, 25, 255))
        dr.polygon([(90, 95), (170, 95), (160, 225), (100, 225)], fill=shirt)
        dr.line([(110, 225), (110, 365)], fill=pants, width=18)
        dr.line([(150, 225), (150, 365)], fill=pants, width=18)
        dr.rectangle([95, 360, 125, 378], fill=(30, 30, 35, 255))
        dr.rectangle([135, 360, 165, 378], fill=(30, 30, 35, 255))

    else:
        dr.ellipse([110, 45, 150, 95], fill=skin)
        dr.polygon([(100, 95), (165, 95), (155, 220), (105, 220)], fill=shirt)
        dr.line([(115, 220), (105, 365)], fill=pants, width=16)
        dr.line([(145, 220), (160, 365)], fill=pants, width=16)

    return img


def draw_bicycle(view: str) -> Image.Image:
    w, h = 380, 320
    img, dr = _create_canvas((w, h))
    frame_col = (30, 140, 210)

    if view in ("side", "oblique_front", "oblique_rear"):
        _add_wheel(dr, 100, 230, 42, rim_color=(200, 205, 210))
        _add_wheel(dr, 280, 230, 42, rim_color=(200, 205, 210))
        # Khung tam giác kim cương
        dr.line([(100, 230), (175, 230)], fill=frame_col, width=5)
        dr.line([(175, 230), (150, 140)], fill=frame_col, width=5)
        dr.line([(150, 140), (240, 140)], fill=frame_col, width=5)
        dr.line([(240, 140), (280, 230)], fill=frame_col, width=5)
        dr.line([(175, 230), (240, 140)], fill=frame_col, width=5)
        # Yên xe & Ghi-đông
        dr.rectangle([135, 130, 165, 140], fill=(20, 20, 25, 255))
        dr.line([(240, 140), (250, 110)], fill=(120, 125, 130, 255), width=5)
        dr.line([(235, 110), (265, 110)], fill=(20, 20, 25, 255), width=5)
    else:
        _add_wheel(dr, 190, 235, 40)
        dr.line([(190, 235), (190, 125)], fill=frame_col, width=6)
        dr.line([(160, 125), (220, 125)], fill=(20, 20, 25, 255), width=6)

    return img


def draw_animal(category: str, view: str) -> Image.Image:
    """Vẽ chó, bò vàng hoặc xe bò."""
    w, h = 440, 320
    img, dr = _create_canvas((w, h))

    is_cow = ("cow" in category or "ox" in category)
    fur_col = (185, 125, 55) if is_cow else (195, 145, 80) # Vàng bò hoặc nâu chó

    if view == "side":
        # Thân mình
        dr.ellipse([110, 110, 330, 210], fill=fur_col)
        # Đầu & tai
        head_r = 45 if is_cow else 30
        dr.ellipse([70, 75, 70 + head_r * 2, 75 + head_r * 2], fill=fur_col)
        if is_cow:
            # Sừng cong
            dr.arc([60, 45, 100, 95], start=180, end=320, fill=(240, 235, 220, 255), width=6)
        # 4 chân
        leg_w = 14 if is_cow else 8
        for x in (130, 170, 260, 300):
            dr.line([(x, 190), (x, 285)], fill=fur_col, width=leg_w)
            dr.rectangle([x - leg_w // 2, 280, x + leg_w // 2, 290], fill=(40, 35, 30, 255)) # Móng
        # Đuôi
        dr.line([(325, 130), (370, 210)], fill=fur_col, width=6 if is_cow else 4)

        if "ox_cart" in category:
            # Cỗ xe bò phía sau
            dr.rectangle([340, 110, 435, 220], fill=(135, 95, 55, 255))
            _add_wheel(dr, 395, 235, 45, rim_color=(160, 120, 80))
            dr.line([(240, 140), (340, 180)], fill=(70, 50, 30, 255), width=8) # Càng xe

    elif view in ("front", "rear"):
        dr.ellipse([160, 110, 280, 220], fill=fur_col)
        dr.ellipse([175, 70, 265, 150], fill=fur_col)
        if is_cow and view == "front":
            dr.arc([160, 45, 205, 95], start=180, end=320, fill=(240, 235, 220, 255), width=5)
            dr.arc([235, 45, 280, 95], start=220, end=360, fill=(240, 235, 220, 255), width=5)
        dr.line([(185, 210), (185, 290)], fill=fur_col, width=12)
        dr.line([(255, 210), (255, 290)], fill=fur_col, width=12)

    else:
        dr.ellipse([120, 110, 320, 210], fill=fur_col)
        dr.ellipse([80, 75, 160, 150], fill=fur_col)
        dr.line([(140, 190), (140, 285)], fill=fur_col, width=12)
        dr.line([(280, 190), (280, 285)], fill=fur_col, width=12)

    return img


def draw_barrier_cone(category: str, view: str) -> Image.Image:
    w, h = 320, 280
    img, dr = _create_canvas((w, h))

    if "cone" in category:
        # Cọc tiêu giao thông (Traffic Cone)
        dr.polygon([(160, 40), (105, 235), (215, 235)], fill=(245, 100, 25, 255)) # Cam
        dr.polygon([(145, 105), (175, 105), (185, 145), (135, 145)], fill=(250, 250, 250, 255)) # Vạch phản quang
        dr.rectangle([80, 235, 240, 260], fill=(30, 30, 35, 255)) # Đế đen

    else:
        # Rào chắn công trình (Barrier)
        dr.rounded_rectangle([40, 70, 280, 210], radius=10, fill=(240, 240, 240, 255), outline=(60, 60, 65, 255), width=2)
        # Sọc chéo đỏ phản quang
        for x in range(30, 280, 40):
            dr.polygon([(x, 70), (x + 25, 70), (x + 5, 210), (x - 20, 210)], fill=(225, 35, 30, 255))
        # Chân đế
        dr.line([(70, 210), (70, 255)], fill=(60, 65, 70, 255), width=8)
        dr.line([(250, 210), (250, 255)], fill=(60, 65, 70, 255), width=8)
        dr.line([(45, 255), (95, 255)], fill=(40, 45, 50, 255), width=6)
        dr.line([(225, 255), (275, 255)], fill=(40, 45, 50, 255), width=6)

    return img


def generate_sprite(category: str, view: str) -> Image.Image:
    """Hàm dispatch sinh ảnh RGBA cho từng category và view."""
    cat = category.lower()
    if cat == "car":
        return draw_car(view)
    if cat == "truck":
        return draw_truck(view)
    if cat == "bus":
        return draw_bus(view)
    if cat in ("motorcycle", "moto_cargo", "moto_multi", "moto_tow"):
        return draw_motorcycle_family(cat, view)
    if cat == "ba_gac":
        return draw_ba_gac(view)
    if cat == "vendor_cart":
        return draw_vendor_cart(view)
    if cat == "pedestrian":
        return draw_pedestrian(view)
    if cat == "bicycle":
        return draw_bicycle(view)
    if cat in ("dog", "cow", "ox_cart"):
        return draw_animal(cat, view)
    if cat in ("barrier", "traffic_cone"):
        return draw_barrier_cone(cat, view)
    return draw_car(view)


ALL_CATEGORIES = (
    "ba_gac", "moto_cargo", "moto_multi", "moto_tow", "vendor_cart",
    "cow", "dog", "ox_cart", "car", "truck", "bus",
    "motorcycle", "bicycle", "pedestrian", "barrier", "traffic_cone"
)

ALL_VIEWS = ("front", "rear", "side", "oblique_front", "oblique_rear")


def bootstrap_all_assets(target_dir: Optional[Path] = None, overwrite: bool = False) -> int:
    """Tạo toàn bộ thư mục và file PNG RGBA mẫu nếu chưa tồn tại."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parents[2] / "assets" / "vn_objects"
    target_dir.mkdir(parents=True, exist_ok=True)

    n_created = 0
    for cat in ALL_CATEGORIES:
        cat_dir = target_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        for v in ALL_VIEWS:
            fp = cat_dir / f"{v}.png"
            if not fp.is_file() or overwrite:
                sprite = generate_sprite(cat, v)
                sprite.save(fp, format="PNG")
                n_created += 1

    return n_created


if __name__ == "__main__":
    count = bootstrap_all_assets(overwrite=True)
    print(f"Đã tạo thành công {count} ảnh PNG RGBA tại assets/vn_objects/")
