from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import torch

from .blending import blend_tensor, image_to_numpy
from .projection import warp_images
from .seams import find_seams
from .views import render_view


LOGGER = logging.getLogger(__name__)


def stitch_images(
    images: list[np.ndarray],
    cameras: list[cv2.detail.CameraParams],
    projection: str,
    panini_distance: float,
    panini_squeeze: float,
    erp_width: int,
    seam_method: str,
    seam_scale: float,
    seam_megapixels: float | None,
    blend_method: str,
    blend_strength: float,
    view_mode: str,
    view_size: int,
    view_rotation_degrees: float,
    view_zoom: float,
    fisheye_fov_degrees: float,
    cubemap_face_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run projection, seam estimation and blending on one torch device."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = True

    effective_projection = projection
    if view_mode != "normal" and projection not in {"erp", "equirectangular"}:
        LOGGER.info("View %s requires ERP; overriding projection %s", view_mode, projection)
        effective_projection = "erp"

    started = time.perf_counter()
    warped = warp_images(
        images,
        cameras,
        effective_projection,
        device,
        panini_distance=panini_distance,
        panini_squeeze=panini_squeeze,
        erp_width=erp_width,
    )
    _log_stage("projection", started, device)

    started = time.perf_counter()
    warped = find_seams(
        warped,
        method=seam_method,
        scale=seam_scale,
        megapixels=seam_megapixels,
    )
    _log_stage("seam", started, device)

    started = time.perf_counter()
    panorama_tensor = blend_tensor(warped, method=blend_method, strength=blend_strength)
    _log_stage("exposure + blending", started, device)

    started = time.perf_counter()
    panorama_tensor = render_view(
        panorama_tensor,
        mode=view_mode,
        size=view_size,
        rotation_degrees=view_rotation_degrees,
        zoom=view_zoom,
        fisheye_fov_degrees=fisheye_fov_degrees,
        cubemap_face_size=cubemap_face_size,
    )
    panorama = image_to_numpy(panorama_tensor)
    _log_stage(f"{view_mode} view + download", started, device)
    if device.type == "cuda":
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        LOGGER.info("Torch stitching peak allocated CUDA memory: %.2f GiB", peak_gib)
    return panorama


def _log_stage(name: str, started: float, device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    LOGGER.info("Torch %s time: %.3f s", name, time.perf_counter() - started)
