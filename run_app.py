"""
Launcher Script for NovaTeam Synthetic LiDAR SaaS Platform
===========================================================
Run this script to launch the local FastAPI web server:
    python run_app.py
Then open your web browser at:
    http://localhost:8000
Interactive Swagger API Docs:
    http://localhost:8000/docs
"""

import os
import sys
import webbrowser
from pathlib import Path

# Add project root and n3-08c to sys.path
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "n3-08c"))
sys.path.insert(0, str(ROOT / "webapp" / "backend"))

import socket
import uvicorn

def find_available_port(preferred_port: int = 8050) -> int:
    def is_free(p: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', p))
                return True
            except OSError:
                return False

    if is_free(preferred_port):
        return preferred_port
    for p in range(preferred_port + 1, preferred_port + 50):
        if is_free(p):
            return p
    return preferred_port

if __name__ == "__main__":
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass

    preferred = int(os.environ.get("PORT", 8050))
    port = find_available_port(preferred)
    url = f"http://localhost:{port}"

    print("\n" + "=" * 70)
    print("  [+] NOVATEAM - SYNTHETIC LIDAR SAAS PLATFORM (DUAL-MODE ENGINE)")
    print(f"  [>] Web Dashboard:  {url}")
    print(f"  [>] API Swagger:    {url}/docs")
    print(f"  [>] Slide Deck:     {url}/slides")
    print("  [*] Supported formats: .bin, .pcd, .ply, .las, .laz, .xyz, .csv, .npy")
    if port != preferred:
        print(f"  [!] Port {preferred} was occupied, auto-routed to port {port}")
    print("  [*] Press Ctrl + C to stop the server.")
    print("=" * 70 + "\n")

    # Automatically try opening the browser
    try:
        webbrowser.open(url)
    except Exception:
        pass

    uvicorn.run("webapp.backend.main:app", host="0.0.0.0", port=port, reload=False)

