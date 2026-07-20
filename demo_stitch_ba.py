"""VGGT-Omega panorama demo with LightGlue bundle adjustment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from pano360.config import (
    DEFAULT_CHECKPOINT,
    PROJECT_ROOT,
    BlendConfig,
    BundleAdjustmentConfig,
    ModelConfig,
    PipelineConfig,
    ProjectionConfig,
    SeamConfig,
    ViewConfig,
)
from pano360.pipeline import PanoramaPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pano360 stitching with VGGT-Omega (with LightGlue bundle adjustment)"
    )
    parser.add_argument(
        "--image-folder",
        "--image_folder",
        dest="image_path",
        type=Path,
        default=PROJECT_ROOT / "example/night",
    )
    parser.add_argument(
        "--output-path",
        "--output_path",
        dest="output_path",
        type=Path,
        default=PROJECT_ROOT / "result/result_ba.jpg",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument(
        "--image-resolution",
        "--image_resolution",
        dest="image_resolution",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--preprocess-mode",
        "--preprocess_mode",
        dest="preprocess_mode",
        choices=("balanced", "max_size"),
        default="balanced",
    )
    parser.add_argument(
        "--projection",
        "--warp_type",
        dest="projection",
        choices=(
            "auto",
            "plane",
            "cylindrical",
            "spherical",
            "mercator",
            "panini",
            "erp",
            "equirectangular",
        ),
        default="auto",
    )
    parser.add_argument("--panini-distance", type=float, default=1.0)
    parser.add_argument("--panini-squeeze", type=float, default=1.0)
    parser.add_argument("--erp-width", type=int, default=8192)
    parser.add_argument(
        "--view",
        choices=("normal", "little_planet", "rabbit_hole", "cubemap", "fisheye"),
        default="normal",
    )
    parser.add_argument("--view-size", type=int, default=2048)
    parser.add_argument("--view-rotation", type=float, default=0.0)
    parser.add_argument("--view-zoom", type=float, default=0.65)
    parser.add_argument("--fisheye-fov", type=float, default=180.0)
    parser.add_argument("--cubemap-face-size", type=int, default=1024)
    parser.add_argument(
        "--share-intrinsics",
        "--share_cameras",
        dest="share_intrinsics",
        action="store_true",
    )
    parser.add_argument("--extractor", choices=("aliked", "superpoint", "sift"), default="aliked")
    parser.add_argument(
        "--max-query-points",
        "--max_query_pts",
        dest="max_query_points",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--ba-iterations",
        "--ba_iter",
        dest="ba_iterations",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--seam-method",
        "--seam_find_type",
        dest="seam_method",
        choices=("no", "torch_dp", "torch_soft"),
        default="torch_dp",
    )
    parser.add_argument(
        "--seam-scale",
        "--seam_scale",
        dest="seam_scale",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--seam-megapixels",
        "--seam_megapix",
        dest="seam_megapixels",
        type=float,
    )
    parser.add_argument(
        "--blend-method",
        "--blend_type",
        dest="blend_method",
        choices=("no", "feather", "multiband"),
        default="multiband",
    )
    parser.add_argument("--blend-strength", type=float, default=5.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        image_path=args.image_path,
        output_path=args.output_path,
        projection=ProjectionConfig(
            mode=args.projection,
            panini_distance=args.panini_distance,
            panini_squeeze=args.panini_squeeze,
            erp_width=args.erp_width,
        ),
        view=ViewConfig(
            mode=args.view,
            size=args.view_size,
            rotation_degrees=args.view_rotation,
            zoom=args.view_zoom,
            fisheye_fov_degrees=args.fisheye_fov,
            cubemap_face_size=args.cubemap_face_size,
        ),
        share_intrinsics=args.share_intrinsics,
        model=ModelConfig(
            checkpoint=args.model_path,
            image_resolution=args.image_resolution,
            preprocess_mode=args.preprocess_mode,
            device=args.device,
        ),
        bundle_adjustment=BundleAdjustmentConfig(
            enabled=True,
            extractor=args.extractor,
            max_query_points=args.max_query_points,
            max_iterations=args.ba_iterations,
        ),
        seam=SeamConfig(
            method=args.seam_method,
            scale=args.seam_scale,
            megapixels=args.seam_megapixels,
        ),
        blend=BlendConfig(method=args.blend_method, strength=args.blend_strength),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    PanoramaPipeline(config_from_args(args)).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
