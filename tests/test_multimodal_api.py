"""
Pytest suite for Multi-modal Camera-LiDAR API and Projection Modules.
"""

import pytest
import requests

BASE_URL = "http://localhost:8050"

def test_kpi_metrics():
    res = requests.get(f"{BASE_URL}/api/kpi", timeout=10)
    assert res.status_code == 200
    data = res.json()
    assert data["highlights"]["valid_rate"] >= 0.8
    assert data["highlights"]["collision_rate"] == 0.0
    assert data["highlights"]["total_benchmark_objects"] == 48

def test_vis_images():
    res = requests.get(f"{BASE_URL}/api/vis/list", timeout=10)
    assert res.status_code == 200
    data = res.json()
    assert len(data["images"]) >= 4

def test_surround_rain_preset():
    url = f"{BASE_URL}/api/cameras/surround?dataset=vietnam_monsoon_dataset&filename=syn_0000_ca9a282c_rain.bin"
    res = requests.get(url, timeout=15)
    assert res.status_code == 200
    data = res.json()
    assert data["total_cameras"] == 6
    assert data["cameras_with_objects"] >= 1

def test_surround_fog_preset():
    url = f"{BASE_URL}/api/cameras/surround?dataset=fog_dense_dataset&filename=syn_0000_ca9a282c_fog.bin"
    res = requests.get(url, timeout=15)
    assert res.status_code == 200
    data = res.json()
    assert data["weather_type"] == "fog"
    assert len(data["cameras"]) == 6

def test_pillar3_ai_benchmark():
    res = requests.get(f"{BASE_URL}/api/kpi", timeout=10)
    assert res.status_code == 200
    data = res.json()
    assert "ai_benchmark" in data
    ai_b = data["ai_benchmark"]
    assert ai_b["highlights"]["monsoon_map_gain"] == 23.4
    assert ai_b["highlights"]["monsoon_baseline_map"] == 35.2
    assert ai_b["highlights"]["monsoon_ours_map"] == 58.6
    assert len(ai_b["weather_comparison"]) >= 8
    assert len(ai_b["class_comparison"]) >= 6

def test_slides_endpoint():
    res = requests.get(f"{BASE_URL}/slides", timeout=10)
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
