"""Typed configuration for the panorama pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "model" / "vggt_omega_1b_512.pt"


@dataclass(frozen=True)
class ModelConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    image_resolution: int = 512
    preprocess_mode: str = "balanced"
    device: str = "auto"

    def validate(self) -> None:
        if not self.checkpoint.is_file():
            raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {self.checkpoint}")
        if self.image_resolution <= 0 or self.image_resolution % 16:
            raise ValueError("image_resolution must be a positive multiple of 16")
        if self.preprocess_mode not in {"balanced", "max_size"}:
            raise ValueError(f"Unsupported preprocess mode: {self.preprocess_mode}")
        if self.device not in {"auto", "cuda", "cpu"}:
            raise ValueError(f"Unsupported device: {self.device}")


@dataclass(frozen=True)
class BundleAdjustmentConfig:
    enabled: bool = False
    extractor: str = "aliked"
    max_query_points: int = 2048
    max_iterations: int = 200

    def validate(self) -> None:
        if self.extractor not in {"aliked", "superpoint", "sift"}:
            raise ValueError(f"Unsupported feature extractor: {self.extractor}")
        if self.max_query_points < 4:
            raise ValueError("max_query_points must be at least 4")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")


@dataclass(frozen=True)
class SeamConfig:
    method: str = "torch_dp"
    scale: float = 0.5
    megapixels: float | None = None

    def validate(self) -> None:
        if self.method not in {"no", "torch_dp", "torch_soft"}:
            raise ValueError(f"Unsupported seam finder: {self.method}")
        if self.scale <= 0:
            raise ValueError("seam scale must be positive")
        if self.megapixels is not None and self.megapixels <= 0:
            raise ValueError("seam megapixels must be positive")


@dataclass(frozen=True)
class BlendConfig:
    method: str = "multiband"
    strength: float = 5.0

    def validate(self) -> None:
        if self.method not in {"no", "feather", "multiband"}:
            raise ValueError(f"Unsupported blend method: {self.method}")
        if self.strength <= 0:
            raise ValueError("blend strength must be positive")


@dataclass(frozen=True)
class ProjectionConfig:
    mode: str = "auto"
    panini_distance: float = 1.0
    panini_squeeze: float = 1.0
    erp_width: int = 8192

    def validate(self) -> None:
        supported = {
            "auto",
            "plane",
            "cylindrical",
            "spherical",
            "mercator",
            "panini",
            "erp",
            "equirectangular",
        }
        if self.mode not in supported:
            raise ValueError(f"Unsupported projection: {self.mode}")
        if self.panini_distance <= 0:
            raise ValueError("panini distance must be positive")
        if self.panini_squeeze <= 0:
            raise ValueError("panini squeeze must be positive")
        if self.erp_width < 512 or self.erp_width % 2:
            raise ValueError("ERP width must be an even integer of at least 512")


@dataclass(frozen=True)
class ViewConfig:
    mode: str = "normal"
    size: int = 2048
    rotation_degrees: float = 0.0
    zoom: float = 0.65
    fisheye_fov_degrees: float = 180.0
    cubemap_face_size: int = 1024

    def validate(self) -> None:
        if self.mode not in {"normal", "little_planet", "rabbit_hole", "cubemap", "fisheye"}:
            raise ValueError(f"Unsupported panorama view: {self.mode}")
        if self.size <= 0:
            raise ValueError("view size must be positive")
        if self.zoom <= 0:
            raise ValueError("view zoom must be positive")
        if not 0 < self.fisheye_fov_degrees <= 360:
            raise ValueError("fisheye FOV must be in (0, 360]")
        if self.cubemap_face_size <= 0:
            raise ValueError("cubemap face size must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    image_path: Path
    output_path: Path
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    view: ViewConfig = field(default_factory=ViewConfig)
    share_intrinsics: bool = False
    model: ModelConfig = field(default_factory=ModelConfig)
    bundle_adjustment: BundleAdjustmentConfig = field(default_factory=BundleAdjustmentConfig)
    seam: SeamConfig = field(default_factory=SeamConfig)
    blend: BlendConfig = field(default_factory=BlendConfig)

    def validate(self) -> None:
        self.model.validate()
        self.bundle_adjustment.validate()
        self.projection.validate()
        self.view.validate()
        self.seam.validate()
        self.blend.validate()
        if not self.image_path.exists():
            raise FileNotFoundError(f"Input image path not found: {self.image_path}")
