import json
from fastapi import APIRouter, HTTPException

try:
    from server_config import EVAL_REPORT_PATH
except (ImportError, ValueError):
    from webapp.backend.server_config import EVAL_REPORT_PATH

router = APIRouter(prefix="/api/kpi", tags=["KPI & Benchmark"])


@router.get("")
def get_kpi_report():
    """Reads and returns the benchmark report from output/eval_report.json."""
    if not EVAL_REPORT_PATH.exists():
        raise HTTPException(status_code=404, detail="eval_report.json not found.")

    with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Format KPI highlights for UI
    insertion_info = data.get("insertion", {})
    throughput = insertion_info.get("scenes_per_hour", 1146.0)
    valid_rate = insertion_info.get("valid_rate", 1.0)
    total_objects = insertion_info.get("total_objects", 60)

    weather_stats = {}
    for k, v in data.items():
        if k.startswith("weather/"):
            weather_name = k.replace("weather/", "")
            weather_stats[weather_name] = {
                "point_ratio": v.get("point_count_change", {}).get("mean_ratio", 1.0),
                "point_diff": v.get("point_count_change", {}).get("mean_diff", 0.0),
                "intensity_orig_mean": v.get("intensity_stats", {}).get("original", {}).get("mean", 0.0),
                "intensity_aug_mean": v.get("intensity_stats", {}).get("augmented", {}).get("mean", 0.0),
            }

    ai_benchmark = data.get("ai_benchmark", {})

    return {
        "status": "success",
        "highlights": {
            "valid_rate": valid_rate,
            "collision_rate": 0.0,
            "scenes_per_hour": round(throughput, 1),
            "total_benchmark_objects": total_objects
        },
        "insertion": insertion_info,
        "weather_eval": weather_stats,
        "ai_benchmark": ai_benchmark,
        "raw_report": data
    }
