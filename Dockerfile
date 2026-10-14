# Multi-Format Synthetic LiDAR SaaS Platform (Dual-Mode 3D Engine)
FROM python:3.11-slim

# Prevent Python from writing .pyc and buffer stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Install Linux system dependencies required by Open3D, OpenGL, and GLib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libgomp1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY webapp/requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy ONLY Application Source Code (Data directories are strictly mounted via volumes)
COPY n3-08c/ ./n3-08c/
COPY webapp/ ./webapp/

# Create runtime temporary output directory
RUN mkdir -p webapp/temp_outputs output

EXPOSE 8000

# Launch FastAPI via Uvicorn Production ASGI Server
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
