from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class NuscenesAugmentRequest(BaseModel):
    sample_token: Optional[str] = Field(default=None, description="nuScenes sample token. If empty, uses first sample.")
    enable_insertion: bool = Field(default=True, description="Enable Task A object insertion")
    categories: Optional[List[str]] = Field(default=None, description="List of categories to insert (e.g. vehicle.car)")
    num_objects: int = Field(default=3, ge=1, le=10, description="Number of objects to insert")
    weather_type: str = Field(default="none", description="Weather type: none, rain, fog")
    weather_level: str = Field(default="moderate", description="Weather severity: light, moderate, heavy")
    enable_wet_road: bool = Field(default=False, description="Enable Vietnam wet road puddle reflection")
    output_format: str = Field(default=".pcd", description="Export format (.pcd, .bin, .ply, .las, .xyz, .npy)")


class AugmentStats(BaseModel):
    task_id: str
    mode: str
    initial_point_count: int
    final_point_count: int
    point_count_diff: int
    inserted_objects_count: int
    weather_type: str
    weather_level: str
    wet_road_enabled: bool
    elapsed_ms: float
    output_file: str
    input_file: str


class PreviewData(BaseModel):
    original_points: List[float] = Field(description="Flat array [x0, y0, z0, i0, ...]")
    augmented_points: List[float] = Field(description="Flat array [x0, y0, z0, i0, ...]")
    num_original: int
    num_augmented: int
    boxes: List[Dict[str, Any]]


class AugmentResponse(BaseModel):
    status: str
    task_id: str
    stats: AugmentStats
    preview: PreviewData
    download_urls: Dict[str, str]
