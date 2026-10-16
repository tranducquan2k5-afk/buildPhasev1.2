# ====================================================================
# STAGE 1: WEB DEMO N3-08C (CPU Nhanh, Siêu nhẹ cho Demo & Khởi chạy nhanh)
# ====================================================================
FROM python:3.10-slim AS demo

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOST=0.0.0.0 PORT=8050 NO_BROWSER=1 \
    NUSCENES_DATAROOT=/app/datasets/nuscenes N308C_OUTPUT_DIR=/app/output

# Cài đặt thư viện hệ thống cần thiết cho OpenCV và hiển thị hình ảnh
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r webapp/requirements.txt

COPY n3-08c/ n3-08c/
COPY webapp/ webapp/
COPY docs/ docs/
COPY run_app.py .
RUN mkdir -p webapp/temp_outputs output datasets

EXPOSE 8050
CMD ["python", "run_app.py"]

# ====================================================================
# STAGE 2: GPU ACCELERATED ENGINE (CUDA 11.8 + cuDNN 8 + PyTorch + OpenPCDet)
# ====================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS gpu

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 PORT=8050 NO_BROWSER=1 \
    NUSCENES_DATAROOT=/app/datasets/nuscenes N308C_OUTPUT_DIR=/app/output \
    PCDET_ROOT=/app/OpenPCDet

# Cài đặt môi trường build C++/CUDA, Python 3.10 và git
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git build-essential \
    libgl1 libglib2.0-0 libgomp1 cmake ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && ln -s /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Cài đặt PyTorch tương thích CUDA 11.8
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir --retries 10 --timeout 120 -r webapp/requirements.txt

# Cài đặt OpenPCDet từ mã nguồn
RUN git clone https://github.com/open-mmlab/OpenPCDet.git /app/OpenPCDet \
    && cd /app/OpenPCDet && pip install --no-cache-dir -r requirements.txt \
    && python setup.py develop

COPY n3-08c/ n3-08c/
COPY webapp/ webapp/
COPY docs/ docs/
COPY run_app.py .
RUN mkdir -p webapp/temp_outputs output datasets

EXPOSE 8050
CMD ["python", "run_app.py"]
