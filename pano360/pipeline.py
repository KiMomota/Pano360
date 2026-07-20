from __future__ import annotations

import gc
import logging

import torch

from .camera import cameras_from_bundle_adjustment, decode_cameras
from .config import PipelineConfig
from .io import discover_images, load_rgb_images, preprocess_images, save_rgb_image
from .model import load_camera_model, predict_camera_poses, release_device_memory, resolve_device
from .stitching import stitch_images


LOGGER = logging.getLogger(__name__)


class PanoramaPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        config.validate()
        self.config = config

    def run(self):
        image_paths = discover_images(self.config.image_path)
        LOGGER.info("Found %d input images", len(image_paths))
        original_images = load_rgb_images(image_paths)
        model_batch = preprocess_images(
            original_images,
            mode=self.config.model.preprocess_mode,
            image_resolution=self.config.model.image_resolution,
        )
        LOGGER.info("VGGT-Omega input shape: %s", tuple(model_batch.images.shape))

        device = resolve_device(self.config.model.device)
        model = load_camera_model(self.config.model.checkpoint, device)
        predictions = predict_camera_poses(model, model_batch.images, device)
        cameras = decode_cameras(
            predictions,
            model_size_hw=tuple(model_batch.images.shape[-2:]),
            transforms=model_batch.transforms,
            share_intrinsics=self.config.share_intrinsics,
            for_bundle_adjustment=self.config.bundle_adjustment.enabled,
        )

        del predictions, model_batch, model
        release_device_memory(device)

        if self.config.bundle_adjustment.enabled:
            cameras = self._run_bundle_adjustment(original_images, cameras, device)

        LOGGER.info("PyTorch stitching device: %s", device)
        panorama = stitch_images(
            original_images,
            cameras,
            projection=self.config.projection.mode,
            panini_distance=self.config.projection.panini_distance,
            panini_squeeze=self.config.projection.panini_squeeze,
            erp_width=self.config.projection.erp_width,
            seam_method=self.config.seam.method,
            seam_scale=self.config.seam.scale,
            seam_megapixels=self.config.seam.megapixels,
            blend_method=self.config.blend.method,
            blend_strength=self.config.blend.strength,
            view_mode=self.config.view.mode,
            view_size=self.config.view.size,
            view_rotation_degrees=self.config.view.rotation_degrees,
            view_zoom=self.config.view.zoom,
            fisheye_fov_degrees=self.config.view.fisheye_fov_degrees,
            cubemap_face_size=self.config.view.cubemap_face_size,
            device=device,
        )
        save_rgb_image(self.config.output_path, panorama)
        LOGGER.info("Panorama saved to %s", self.config.output_path)
        return panorama

    def _run_bundle_adjustment(self, images, cameras, device):
        from .stitching.bundle_adjustment import bundle_adjust
        from .stitching.features import match_image_pairs

        LOGGER.info("Running LightGlue matching and bundle adjustment")
        image_tensors = [
            torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0).to(device)
            for image in images
        ]
        matches = match_image_pairs(
            images,
            image_tensors,
            max_query_points=self.config.bundle_adjustment.max_query_points,
            extractor_name=self.config.bundle_adjustment.extractor,
        )
        adjusted = bundle_adjust(
            images,
            matches,
            cameras,
            max_iterations=self.config.bundle_adjustment.max_iterations,
        )
        del image_tensors
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        return cameras_from_bundle_adjustment(adjusted, [image.shape[:2] for image in images])
