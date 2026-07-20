"""Feather and streaming multi-band panorama blending on CUDA."""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as functional

from .exposure import estimate_exposure_gains
from .types import WarpedImages


LOGGER = logging.getLogger(__name__)


def blend_images(
    warped: WarpedImages,
    method: str = "multiband",
    strength: float = 5.0,
) -> np.ndarray:
    """Blend CUDA image ROIs and download one final RGB image."""
    return image_to_numpy(blend_tensor(warped, method, strength))


def blend_tensor(
    warped: WarpedImages,
    method: str = "multiband",
    strength: float = 5.0,
) -> torch.Tensor:
    """Blend CUDA image ROIs while retaining the result on its torch device."""
    gains = estimate_exposure_gains(warped)
    if method in {"no", "feather"}:
        result = _weighted_blend(warped, gains)
    elif method == "multiband":
        result = _multiband_blend(warped, gains, strength)
    else:
        raise ValueError(f"Unsupported torch blend method: {method}")
    return result


def image_to_numpy(image: torch.Tensor) -> np.ndarray:
    """Convert one CHW [0, 1] tensor to an RGB uint8 NumPy image."""
    return (
        image.clamp_(0.0, 1.0)
        .mul_(255.0)
        .round_()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )


def _weighted_blend(warped: WarpedImages, gains: torch.Tensor) -> torch.Tensor:
    left, top, width, height = warped.canvas_roi
    numerator = torch.zeros((3, height, width), device=warped.device, dtype=torch.float32)
    denominator = torch.zeros((1, height, width), device=warped.device, dtype=torch.float32)
    for image, mask, gain, corner in zip(
        warped.images, warped.masks, gains, warped.corners
    ):
        offset_x = corner[0] - left
        offset_y = corner[1] - top
        image_height, image_width = image.shape[-2:]
        destination = (
            slice(offset_y, offset_y + image_height),
            slice(offset_x, offset_x + image_width),
        )
        weight = mask.float()
        numerator[:, destination[0], destination[1]].add_(image.float() * gain * weight)
        denominator[:, destination[0], destination[1]].add_(weight)
    return numerator / denominator.clamp_min_(1e-6)


def _multiband_blend(
    warped: WarpedImages,
    gains: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    left, top, width, height = warped.canvas_roi
    blend_width = max(1.0, math.sqrt(width * height) * strength / 100.0)
    requested_bands = max(1, int(math.ceil(math.log2(blend_width))) - 1)
    maximum_bands = max(1, int(math.floor(math.log2(max(2, min(width, height))))) - 1)
    band_count = min(requested_bands, maximum_bands)
    level_shapes = _level_shapes(height, width, band_count)
    LOGGER.info(
        "Torch multiband canvas: %dx%d; pyramid bands: %d",
        width,
        height,
        band_count,
    )

    accumulated_images = [
        torch.zeros((1, 3, level_height, level_width), device=warped.device, dtype=torch.float32)
        for level_height, level_width in level_shapes
    ]
    accumulated_weights = [
        torch.zeros((1, 1, level_height, level_width), device=warped.device, dtype=torch.float32)
        for level_height, level_width in level_shapes
    ]
    storage_dtype = torch.float16 if warped.device.type == "cuda" else torch.float32

    for image, mask, valid_mask, gain, corner in zip(
        warped.images,
        warped.masks,
        warped.exposure_masks,
        gains,
        warped.corners,
    ):
        image_canvas = torch.zeros(
            (1, 3, height, width), device=warped.device, dtype=storage_dtype
        )
        mask_canvas = torch.zeros(
            (1, 1, height, width), device=warped.device, dtype=storage_dtype
        )
        valid_canvas = torch.zeros(
            (1, 1, height, width), device=warped.device, dtype=storage_dtype
        )
        offset_x = corner[0] - left
        offset_y = corner[1] - top
        image_height, image_width = image.shape[-2:]
        image_canvas[
            :, :, offset_y : offset_y + image_height, offset_x : offset_x + image_width
        ] = image.to(storage_dtype).unsqueeze(0) * gain.to(storage_dtype)
        mask_canvas[
            :, :, offset_y : offset_y + image_height, offset_x : offset_x + image_width
        ] = mask.to(storage_dtype).unsqueeze(0)
        valid_canvas[
            :, :, offset_y : offset_y + image_height, offset_x : offset_x + image_width
        ] = valid_mask.to(storage_dtype).unsqueeze(0)

        # The source-valid mask is intentionally separate from the seam mask.
        # The seam mask decides which image contributes to the blend, whereas
        # the valid mask prevents the black canvas outside an image ROI from
        # leaking into its Gaussian/Laplacian pyramid.
        current_image = image_canvas
        current_mask = mask_canvas
        current_valid = valid_canvas
        for level in range(band_count):
            next_image, next_valid = _masked_pyr_down(current_image, current_valid)
            next_mask = _pyr_down(current_mask)
            expanded = functional.interpolate(
                next_image,
                size=current_image.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            laplacian = current_image - expanded
            weight = current_mask.float()
            accumulated_images[level].add_(laplacian.float() * weight)
            accumulated_weights[level].add_(weight)
            current_image = next_image
            current_mask = next_mask
            current_valid = next_valid

        weight = current_mask.float()
        accumulated_images[-1].add_(current_image.float() * weight)
        accumulated_weights[-1].add_(weight)
        del image_canvas, mask_canvas, valid_canvas
        del current_image, current_mask, current_valid

    result = accumulated_images[-1] / accumulated_weights[-1].clamp_min(1e-6)
    for level in range(band_count - 1, -1, -1):
        result = functional.interpolate(
            result,
            size=accumulated_images[level].shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        result = result + accumulated_images[level] / accumulated_weights[level].clamp_min(1e-6)
    # Gaussian mask levels deliberately grow beyond the original support.
    # Clip the reconstruction back to the exact full-resolution coverage so
    # normalized edge colors never appear in otherwise uncovered canvas areas.
    coverage = accumulated_weights[0] > 1e-6
    return result.mul_(coverage).squeeze(0)


def _pyr_down(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    one_dimensional = torch.tensor(
        [1.0, 4.0, 6.0, 4.0, 1.0], device=image.device, dtype=image.dtype
    ).div_(16.0)
    kernel = torch.outer(one_dimensional, one_dimensional)[None, None]
    kernel = kernel.expand(channels, 1, 5, 5)
    return functional.conv2d(image, kernel, stride=2, padding=2, groups=channels)


def _masked_pyr_down(
    image: torch.Tensor,
    valid: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample without mixing the black canvas outside an image ROI.

    A normal Gaussian pyramid treats pixels outside each warped image as black.
    Its Laplacian then contains a strong positive/negative edge at every ROI,
    which appears as bright vertical bands after several images are blended.
    Normalized convolution extends valid colors into the shrinking support and
    keeps those artificial canvas edges out of the image pyramid.
    """
    next_valid = _pyr_down(valid)
    weighted = _pyr_down(image * valid)
    epsilon = torch.finfo(image.dtype).eps
    next_image = weighted / next_valid.clamp_min(epsilon)
    next_image = torch.where(next_valid > epsilon, next_image, 0.0)
    return next_image, next_valid


def _level_shapes(height: int, width: int, band_count: int) -> list[tuple[int, int]]:
    shapes = [(height, width)]
    for _ in range(band_count):
        height = (height + 1) // 2
        width = (width + 1) // 2
        shapes.append((height, width))
    return shapes
