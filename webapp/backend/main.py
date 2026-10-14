import os
import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Add paths to sys.path
BACKEND_DIR = Path(__file__).resolve().parent
WEBAPP_DIR = BACKEND_DIR.parent
PROJECT_ROOT = WEBAPP_DIR.parent
N3_08C_DIR = PROJECT_ROOT / "n3-08c"

for p in (PROJECT_ROOT, N3_08C_DIR, BACKEND_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from routers import augment, kpi, presets, templates

app = FastAPI(
    title="NovaTeam Synthetic LiDAR SaaS API",
    description="Dual-Mode 3D Point Cloud Augmentation Engine (nuScenes Sandbox & Custom Upload)",
    version="2.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Files
FRONTEND_DIR = WEBAPP_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Mount output/vis for direct image access by slides.html
VIS_DIR = PROJECT_ROOT / "output" / "vis"
if VIS_DIR.exists():
    app.mount("/output/vis", StaticFiles(directory=str(VIS_DIR)), name="output_vis")

# Include API Routers
app.include_router(templates.router)
app.include_router(kpi.router)
app.include_router(augment.router)
app.include_router(presets.router)


@app.get("/")
def serve_index():
    """Serves the main single-page web application."""
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return {"status": "error", "message": "Frontend index.html not found."}
    return FileResponse(index_file)


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Synthetic LiDAR SaaS", "version": "2.0.0"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"\n==================================================================")
    print(f"🚀 FASTAPI SYNTHETIC LIDAR SAAS SERVER (DUAL-MODE ENGINE)")
    print(f"🌐 Server running at: http://localhost:{port}")
    print(f"📖 Interactive Swagger Docs: http://localhost:{port}/docs")
    print(f"==================================================================\n")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
