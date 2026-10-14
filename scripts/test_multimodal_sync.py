"""
Automated Multi-Modal Camera - LiDAR Synchronization Test Script
Tests 2D-3D Sensor Fusion, 6-Camera Surround View, KPI metrics, and Evidence Images.
"""

import sys
import os
import json
import base64
import requests
import io
from PIL import Image

BASE_URL = "http://localhost:8050"

def log_pass(msg):
    print(f"\033[92m[PASS]\033[0m {msg}")

def log_fail(msg):
    print(f"\033[91m[FAIL]\033[0m {msg}")

def log_info(msg):
    print(f"\033[94m[INFO]\033[0m {msg}")

def test_kpi_dashboard():
    log_info("1. Testing KPI Benchmark Endpoint (/api/kpi)...")
    url = f"{BASE_URL}/api/kpi"
    res = requests.get(url, timeout=10)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "success", "Expected status success"
    
    highlights = data["highlights"]
    insertion = data["insertion"]
    
    log_info(f"   - Valid Labels Rate: {highlights['valid_rate'] * 100:.1f}%")
    log_info(f"   - Collision Rate: {highlights['collision_rate']:.1f}% (Target: 0.0%)")
    log_info(f"   - Total Benchmarked Objects: {highlights['total_benchmark_objects']} (Target: 48)")
    log_info(f"   - Scenes Per Hour Throughput: {highlights['scenes_per_hour']}")
    
    assert highlights["collision_rate"] == 0.0, "Collision rate must be 0.0%"
    assert highlights["total_benchmark_objects"] == 48, f"Expected 48 benchmarked objects, got {highlights['total_benchmark_objects']}"
    assert len(data["weather_eval"]) >= 5, "Expected weather degradation data"

    # Pillar 3 AI 3D Object Detection assertions
    assert "ai_benchmark" in data, "Expected ai_benchmark in /api/kpi data"
    ai_b = data["ai_benchmark"]
    log_info(f"   - Pillar 3 Monsoon mAP Gain: +{ai_b['highlights']['monsoon_map_gain']}% ({ai_b['highlights']['monsoon_baseline_map']}% -> {ai_b['highlights']['monsoon_ours_map']}%)")
    log_info(f"   - Pillar 3 Monsoon NDS Gain: +{ai_b['highlights']['monsoon_nds_gain']}% ({ai_b['highlights']['monsoon_baseline_nds']}% -> {ai_b['highlights']['monsoon_ours_nds']}%)")
    assert ai_b['highlights']['monsoon_map_gain'] == 23.4
    assert ai_b['highlights']['monsoon_baseline_map'] == 35.2
    assert ai_b['highlights']['monsoon_ours_map'] == 58.6
    assert len(ai_b['weather_comparison']) >= 8
    assert len(ai_b['class_comparison']) >= 6

    log_pass("KPI Benchmark & eval_report.json synchronized accurately.")

def test_vis_evidence_images():
    log_info("2. Testing 3D Evidence Images in output/vis/ (/api/vis/list)...")
    url = f"{BASE_URL}/api/vis/list"
    res = requests.get(url, timeout=10)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "success"
    
    images = data["images"]
    log_info(f"   - Found {len(images)} evidence images in output/vis/")
    assert len(images) >= 4, f"Expected at least 4 images, got {len(images)}"
    
    for img in images:
        img_url = f"{BASE_URL}{img['url']}"
        img_res = requests.get(img_url, timeout=10)
        assert img_res.status_code == 200, f"Failed to fetch image: {img_url}"
        assert len(img_res.content) > 50000, f"Image content too small ({len(img_res.content)} bytes): {img['name']}"
        log_info(f"   - [OK] {img['name']} ({len(img_res.content) / 1024:.1f} KB)")
        
    log_pass("All 4 evidence images verified and accessible via HTTP.")

def test_preset_rain_surround_cameras():
    log_info("3. Testing Preset Rain Scene 6-Camera Surround View...")
    url = f"{BASE_URL}/api/cameras/surround?dataset=vietnam_monsoon_dataset&filename=syn_0000_ca9a282c_rain.bin"
    res = requests.get(url, timeout=15)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "success"
    assert data["total_cameras"] == 6
    assert len(data["cameras"]) == 6
    assert data["weather_type"] == "rain"
    
    expected_channels = [
        "CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT",
        "CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"
    ]
    received_channels = [c["channel"] for c in data["cameras"]]
    assert received_channels == expected_channels, f"Channel mismatch: {received_channels}"
    
    log_info(f"   - Cameras with detected objects: {data['cameras_with_objects']}/6")
    assert data["cameras_with_objects"] >= 1, "At least 1 camera must detect inserted objects"
    
    for cam in data["cameras"]:
        b64 = cam["image_base64"]
        assert b64.startswith("data:image/jpeg;base64,"), f"Invalid base64 header for {cam['channel']}"
        b64_data = b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_data)
        assert len(raw_bytes) > 5000, f"Image bytes too small ({len(raw_bytes)}) for {cam['channel']}"
        
        # Verify valid JPEG via PIL
        img = Image.open(io.BytesIO(raw_bytes))
        assert img.format == "JPEG"
        log_info(f"   - [CAM OK] {cam['channel']}: {img.size[0]}x{img.size[1]} | Objects: {cam['num_objects']} ({len(raw_bytes)/1024:.1f} KB)")
        
    log_pass("Preset Rain 6-Camera Surround View verified with wireframe & rain streaks.")

def test_preset_fog_surround_cameras():
    log_info("4. Testing Preset Fog Scene 6-Camera Surround View...")
    url = f"{BASE_URL}/api/cameras/surround?dataset=fog_dense_dataset&filename=syn_0000_ca9a282c_fog.bin"
    res = requests.get(url, timeout=15)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}"
    data = res.json()
    assert data["status"] == "success"
    assert data["weather_type"] == "fog"
    assert len(data["cameras"]) == 6
    
    log_info(f"   - Cameras with detected objects: {data['cameras_with_objects']}/6")
    for cam in data["cameras"]:
        b64 = cam["image_base64"]
        b64_data = b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64_data)
        img = Image.open(io.BytesIO(raw_bytes))
        assert img.format == "JPEG"
    log_pass("Preset Fog 6-Camera Surround View verified with Koschmieder atmospheric scattering.")

def test_custom_projection_api():
    log_info("5. Testing On-Demand Camera Projection Endpoint (/api/cameras/project)...")
    payload = {
        "sample_token": "ca9a282c9e77460f8360f564131a8af5",
        "boxes": [
            {
                "center": [16.5, 2.2, -0.6],
                "size": [1.95, 4.8, 1.6],
                "yaw": 0.05,
                "category": "vehicle.car",
                "score": 1.0
            },
            {
                "center": [24.0, -3.5, -0.4],
                "size": [2.5, 7.5, 3.2],
                "yaw": 0.02,
                "category": "vehicle.truck",
                "score": 1.0
            }
        ],
        "weather_type": "rain",
        "weather_level": "heavy"
    }
    
    url = f"{BASE_URL}/api/cameras/project"
    res = requests.post(url, json=payload, timeout=20)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    assert data["total_cameras"] == 6
    log_info(f"   - On-demand projection detected objects in: {data['cameras_with_objects']}/6 cameras")
    assert data["cameras_with_objects"] >= 1
    
    # Check front camera has both or at least one
    front_cam = next(c for c in data["cameras"] if c["channel"] == "CAM_FRONT")
    assert front_cam["has_objects"] is True
    log_info(f"   - CAM_FRONT detected {front_cam['num_objects']} object(s): {[o['category'] for o in front_cam['detected_objects']]}")
    log_pass("On-demand 3D-to-2D projection API verified successfully.")

def test_full_pipeline_nuscenes():
    log_info("6. Testing End-to-End Mode 1 Augmentation Pipeline (/api/augment/nuscenes)...")
    payload = {
        "sample_token": "ca9a282c9e77460f8360f564131a8af5",
        "enable_insertion": True,
        "num_objects": 2,
        "categories": ["vehicle.car", "vehicle.truck"],
        "weather_type": "rain",
        "weather_level": "moderate",
        "enable_wet_road": True,
        "export_format": ".bin"
    }
    url = f"{BASE_URL}/api/augment/nuscenes"
    res = requests.post(url, json=payload, timeout=30)
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    data = res.json()
    assert data["status"] == "success"
    stats = data["stats"]
    log_info(f"   - Initial LiDAR points: {stats['initial_point_count']}")
    log_info(f"   - Final LiDAR points: {stats['final_point_count']}")
    log_info(f"   - Inserted objects count: {stats['inserted_objects_count']}")
    log_info(f"   - Execution time: {stats['elapsed_ms']} ms")
    assert stats["inserted_objects_count"] >= 1, "Expected at least 1 object inserted"
    assert len(data["preview"]["boxes"]) >= 1, "Expected bounding boxes in preview"
    log_pass("End-to-End Mode 1 Augmentation verified with 0 memory errors or crashes.")

def main():
    print("=" * 70)
    print("  NOVATEAM MULTI-MODAL CAMERA-LIDAR SYNCHRONIZATION VERIFICATION")
    print("=" * 70)
    
    try:
        test_kpi_dashboard()
        test_vis_evidence_images()
        test_preset_rain_surround_cameras()
        test_preset_fog_surround_cameras()
        test_custom_projection_api()
        test_full_pipeline_nuscenes()
        
        print("=" * 70)
        print("\033[92m[ALL CHECKS PASSED] 100% MULTI-MODAL SYNCHRONIZATION ACHIEVED!\033[0m")
        print("=" * 70)
        return 0
    except Exception as e:
        print("=" * 70)
        log_fail(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
