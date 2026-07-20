"""CUDA rotation warping for rectilinear and panoramic projections."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as functional

from .types import WarpedImages


LOGGER = logging.getLogger(__name__)
SUPPORTED_PROJECTIONS = {
    "plane",
    "cylindrical",
    "spherical",
    "mercator",
    "panini",
    "erp",
    "equirectangular",
}


def correct_camera_wave(
    cameras: list[cv2.detail.CameraParams], mode: str = "horizontal"
) -> None:
    """Straighten the small set of camera rotations before CUDA warping."""
    correction_modes = {
        "horizontal": cv2.detail.WAVE_CORRECT_HORIZ,
        "vertical": cv2.detail.WAVE_CORRECT_VERT,
        "none": None,
    }
    if mode not in correction_modes:
        raise ValueError(f"Unsupported wave correction: {mode}")
    correction = correction_modes[mode]
    if correction is None:
        return
    rotations = [camera.R.astype(np.float32, copy=True) for camera in cameras]
    cv2.detail.waveCorrect(rotations, correction)
    for camera, rotation in zip(cameras, rotations):
        camera.R = rotation


def choose_projection(cameras: list[cv2.detail.CameraParams]) -> str:
    """Select a projection from the angular span of the predicted cameras."""
    horizontal_span, vertical_span, pitch_std = _camera_span(cameras)
    single_row = pitch_std < 5.0
    if horizontal_span < 60 and vertical_span < 45:
        return "plane"
    if single_row and horizontal_span >= 120 and vertical_span <= 60:
        return "panini"
    if single_row and horizontal_span < 170 and vertical_span < 80:
        return "cylindrical"
    if horizontal_span >= 170 or vertical_span >= 80:
        return "mercator" if vertical_span > 100 else "spherical"
    return "cylindrical"


def warp_images(
    images: list[np.ndarray],
    cameras: list[cv2.detail.CameraParams],
    projection: str,
    device: torch.device,
    panini_distance: float = 1.0,
    panini_squeeze: float = 1.0,
    erp_width: int = 8192,
) -> WarpedImages:
    """Inverse-warp RGB images and masks without leaving the torch device."""
    if len(images) != len(cameras):
        raise ValueError("Image count does not match camera count")
    if not images:
        raise ValueError("At least one image is required")

    focals = np.asarray([camera.focal for camera in cameras], dtype=np.float64)
    if not np.all(np.isfinite(focals)) or np.any(focals <= 0):
        raise ValueError(f"Invalid predicted focal lengths: {focals.tolist()}")
    selected = choose_projection(cameras) if projection == "auto" else projection
    if selected == "equirectangular":
        selected = "erp"
    if selected not in SUPPORTED_PROJECTIONS:
        raise ValueError(
            f"PyTorch stitcher does not support projection {selected!r}; "
            f"choose one of {sorted(SUPPORTED_PROJECTIONS)}"
        )

    warp_scale = (
        erp_width / (2.0 * np.pi) if selected == "erp" else float(np.median(focals))
    )
    coordinate_projection = "spherical" if selected == "erp" else selected
    correct_camera_wave(cameras, mode="horizontal")
    roi_warper = (
        None
        if coordinate_projection == "panini"
        else cv2.PyRotationWarper(coordinate_projection, warp_scale)
    )
    LOGGER.info("Torch projection: %s; warp scale: %.2f", selected, warp_scale)

    tensor_dtype = torch.float16 if device.type == "cuda" else torch.float32
    corners: list[tuple[int, int]] = []
    sizes: list[tuple[int, int]] = []
    warped_images: list[torch.Tensor] = []
    warped_masks: list[torch.Tensor] = []

    for image, camera in zip(images, cameras):
        height, width = image.shape[:2]
        intrinsic = camera.K().astype(np.float32)
        if coordinate_projection == "panini":
            x, y, warped_width, warped_height = _panini_roi(
                (width, height),
                intrinsic,
                camera.R,
                warp_scale,
                panini_distance,
                panini_squeeze,
            )
        else:
            x, y, warped_width, warped_height = roi_warper.warpRoi(
                (width, height), intrinsic, camera.R
            )
        corner = (int(x), int(y))
        size = (int(warped_width), int(warped_height))
        source_x, source_y, positive_depth = _inverse_coordinates(
            corner,
            size,
            intrinsic,
            camera.R,
            warp_scale,
            coordinate_projection,
            device,
            panini_distance,
            panini_squeeze,
        )
        valid = (
            positive_depth
            & (source_x >= 0)
            & (source_x <= width - 1)
            & (source_y >= 0)
            & (source_y <= height - 1)
        )
        grid = torch.stack(
            (
                source_x.mul(2.0 / max(width - 1, 1)).sub(1.0),
                source_y.mul(2.0 / max(height - 1, 1)).sub(1.0),
            ),
            dim=-1,
        ).unsqueeze(0)
        source = (
            torch.from_numpy(np.ascontiguousarray(image))
            .to(device=device, dtype=torch.float32, non_blocking=True)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div_(255.0)
        )
        warped = functional.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="reflection",
            align_corners=True,
        ).squeeze(0)
        mask = valid.unsqueeze(0).to(dtype=tensor_dtype)

        corners.append(corner)
        sizes.append(size)
        warped_images.append(warped.to(dtype=tensor_dtype))
        warped_masks.append(mask)
        del source, grid, source_x, source_y, positive_depth, valid, warped

    return WarpedImages(
        corners=corners,
        sizes=sizes,
        images=warped_images,
        masks=[mask.clone() for mask in warped_masks],
        exposure_masks=warped_masks,
        cameras=cameras,
        projection=selected,
        device=device,
        canvas_override=(
            (-erp_width // 2, 0, erp_width, erp_width // 2)
            if selected == "erp"
            else None
        ),
    )


def _inverse_coordinates(
    corner: tuple[int, int],
    size: tuple[int, int],
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    projection: str,
    device: torch.device,
    panini_distance: float = 1.0,
    panini_squeeze: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x0, y0 = corner
    width, height = size
    destination_y, destination_x = torch.meshgrid(
        torch.arange(y0, y0 + height, device=device, dtype=torch.float32),
        torch.arange(x0, x0 + width, device=device, dtype=torch.float32),
        indexing="ij",
    )
    horizontal = destination_x / scale
    vertical = destination_y / scale
    if projection == "plane":
        world_ray = torch.stack((horizontal, vertical, torch.ones_like(horizontal)))
    elif projection == "cylindrical":
        world_ray = torch.stack(
            (torch.sin(horizontal), vertical, torch.cos(horizontal))
        )
    elif projection == "spherical":
        sin_vertical = torch.sin(vertical)
        world_ray = torch.stack(
            (
                torch.sin(horizontal) * sin_vertical,
                -torch.cos(vertical),
                torch.cos(horizontal) * sin_vertical,
            )
        )
    elif projection == "mercator":
        latitude = 2.0 * torch.atan(torch.exp(vertical)) - torch.pi / 2
        cos_latitude = torch.cos(latitude)
        world_ray = torch.stack(
            (
                torch.sin(horizontal) * cos_latitude,
                torch.sin(latitude),
                torch.cos(horizontal) * cos_latitude,
            )
        )
    elif projection == "panini":
        distance = float(panini_distance)
        squeeze = float(panini_squeeze)
        horizontal_scale = distance + 1.0
        radius = torch.sqrt(horizontal_scale**2 + horizontal**2)
        longitude = torch.atan2(horizontal, torch.full_like(horizontal, horizontal_scale))
        longitude = longitude + torch.asin(
            (horizontal * distance / radius).clamp(-1.0, 1.0)
        )
        tangent_latitude = (
            vertical
            / squeeze
            * (distance + torch.cos(longitude))
            / horizontal_scale
        )
        cos_latitude = torch.rsqrt(1.0 + tangent_latitude.square())
        sin_latitude = tangent_latitude * cos_latitude
        world_ray = torch.stack(
            (
                torch.sin(longitude) * cos_latitude,
                sin_latitude,
                torch.cos(longitude) * cos_latitude,
            )
        )
    else:  # guarded by the public function
        raise ValueError(f"Unsupported torch projection: {projection}")

    camera_matrix = torch.as_tensor(
        intrinsic @ rotation.T,
        device=device,
        dtype=torch.float32,
    )
    camera_ray = camera_matrix @ world_ray.reshape(3, -1)
    depth = camera_ray[2].reshape(height, width)
    safe_depth = torch.where(depth.abs() < 1e-8, torch.full_like(depth, 1e-8), depth)
    source_x = camera_ray[0].reshape(height, width) / safe_depth
    source_y = camera_ray[1].reshape(height, width) / safe_depth
    return source_x, source_y, depth > 0


def _panini_roi(
    image_size: tuple[int, int],
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    distance: float,
    squeeze: float,
) -> tuple[int, int, int, int]:
    """Project the source border to obtain a tight custom Panini ROI."""
    width, height = image_size
    horizontal = np.arange(width, dtype=np.float64)
    vertical = np.arange(height, dtype=np.float64)
    border_x = np.concatenate(
        (horizontal, horizontal, np.zeros_like(vertical), np.full_like(vertical, width - 1))
    )
    border_y = np.concatenate(
        (np.zeros_like(horizontal), np.full_like(horizontal, height - 1), vertical, vertical)
    )
    pixels = np.stack((border_x, border_y, np.ones_like(border_x)))
    rays = rotation.astype(np.float64) @ np.linalg.inv(intrinsic.astype(np.float64)) @ pixels
    norms = np.linalg.norm(rays, axis=0).clip(1e-12)
    longitude = np.arctan2(rays[0], rays[2])
    latitude = np.arcsin(np.clip(rays[1] / norms, -1.0, 1.0))
    denominator = distance + np.cos(longitude)
    valid = denominator > 1e-6
    projected_x = scale * (distance + 1.0) * np.sin(longitude[valid]) / denominator[valid]
    projected_y = (
        scale
        * squeeze
        * (distance + 1.0)
        * np.tan(latitude[valid])
        / denominator[valid]
    )
    if not len(projected_x):
        raise ValueError("Panini projection has no valid source border")
    left = int(np.floor(projected_x.min()))
    top = int(np.floor(projected_y.min()))
    right = int(np.ceil(projected_x.max()))
    bottom = int(np.ceil(projected_y.max()))
    return left, top, right - left + 1, bottom - top + 1


def _camera_span(cameras: list[cv2.detail.CameraParams]) -> tuple[float, float, float]:
    yaws = []
    pitches = []
    for camera in cameras:
        forward = camera.R @ np.array([0.0, 0.0, 1.0])
        yaws.append(np.degrees(np.arctan2(forward[0], forward[2])))
        pitches.append(
            np.degrees(np.arctan2(-forward[1], np.hypot(forward[0], forward[2])))
        )
    wrapped_yaws = np.mod(yaws, 360.0)
    sorted_yaws = np.sort(wrapped_yaws)
    gaps = np.diff(np.concatenate([sorted_yaws, sorted_yaws[:1] + 360.0]))
    horizontal_span = 360.0 - float(np.max(gaps))
    return horizontal_span, float(np.ptp(pitches)), float(np.std(pitches))
