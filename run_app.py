"""Chạy web app demo N3-08C: python run_app.py  ->  http://localhost:8050

Windows (nuscenes-env): sinh kịch bản + 3D + 6 camera + tải file; kết quả detector lấy từ số đã đo sẵn.
WSL env pcdet (có GPU + OpenPCDet): thêm nút "Chạy detector" trực tiếp trên cảnh vừa sinh.
"""
import os
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "webapp" / "backend"))
sys.path.insert(0, str(ROOT / "n3-08c"))

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

if "NUSCENES_DATAROOT" not in os.environ:
    for cand in [
        ROOT / "datasets" / "nuscenes",
        ROOT.parent.parent / "nuscenes-project" / "datasets" / "nuscenes",
        ROOT.parent / "nuscenes-project" / "datasets" / "nuscenes",
        Path(r"D:\VinAI\Build phase\nuscenes-project\datasets\nuscenes"),
    ]:
        if (cand / "v1.0-mini" / "sample.json").is_file():
            os.environ["NUSCENES_DATAROOT"] = str(cand)
            break

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8050))
    print(f"N3-08C web demo: http://localhost:{port}   (API docs: /docs, slide: /slides)")
    if os.environ.get("NO_BROWSER") != "1":
        try:
            webbrowser.open(f"http://localhost:{port}")
        except Exception:
            pass
    os.chdir(ROOT / "webapp" / "backend")
    uvicorn.run("main:app", host=os.environ.get("HOST", "127.0.0.1"), port=port, reload=False)
