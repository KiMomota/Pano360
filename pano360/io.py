from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image


SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class ImageTransform:
    """Pixel transform from an original image into the padded model tensor."""

    original_height: int
    original_width: int
    crop_left: int
    crop_top: int
    crop_height: int
    crop_width: int
    resized_height: int
    resized_width: int
    pad_top: int = 0
    pad_left: int = 0

    @property
    def scale_x(self) -> float:
        return self.resized_width / self.crop_width

    @property
    def scale_y(self) -> float:
        return self.resized_height / self.crop_height


@dataclass(frozen=True)
class ModelBatch:
    images: torch.Tensor
    transforms: tuple[ImageTransform, ...]


def discover_images(path: Path) -> list[Path]:
    """Discover supported images in deterministic filename order."""
    if path.is_dir():
        image_paths = sorted(
            (item for item in path.iterdir() if item.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES),
            key=lambda item: item.name.lower(),
        )
    elif path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES:
        image_paths = [path]
    else:
        raise ValueError(f"Unsupported image path: {path}")

    if not image_paths:
        raise ValueError(f"No supported images found in: {path}")
    return image_paths


def load_rgb_images(image_paths: list[Path]) -> list[np.ndarray]:
    """Load original images as independent RGB uint8 arrays."""
    images = []
    for image_path in image_paths:
        with Image.open(image_path) as image:
            if image.mode == "RGBA":
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, image)
            images.append(np.array(image.convert("RGB"), copy=True))
    return images


def preprocess_images(
    images: list[np.ndarray],
    mode: str = "balanced",
    image_resolution: int = 512,
    patch_size: int = 16,
) -> ModelBatch:
    """Apply VGGT-Omega preprocessing while retaining pixel transforms."""
    if not images:
        raise ValueError("At least one image is required")
    if mode not in {"balanced", "max_size"}:
        raise ValueError(f"Unsupported preprocess mode: {mode}")
    if image_resolution <= 0 or image_resolution % patch_size:
        raise ValueError("image_resolution must be a positive multiple of patch_size")

    tensors: list[torch.Tensor] = []
    transforms: list[ImageTransform] = []
    for array in images:
        image = Image.fromarray(array)
        original_width, original_height = image.size
        image, crop_left, crop_top = _crop_to_supported_aspect_ratio(image)
        crop_width, crop_height = image.size
        aspect_ratio = crop_height / crop_width

        if mode == "balanced":
            resized_height, resized_width = _balanced_target_shape(
                aspect_ratio, image_resolution, patch_size
            )
        else:
            resized_height, resized_width = _max_size_target_shape(
                aspect_ratio, image_resolution, patch_size
            )

        resized = image.resize((resized_width, resized_height), Image.Resampling.BICUBIC)
        tensor = torch.from_numpy(np.array(resized, copy=True)).permute(2, 0, 1).float().div_(255.0)
        tensors.append(tensor)
        transforms.append(
            ImageTransform(
                original_height=original_height,
                original_width=original_width,
                crop_left=crop_left,
                crop_top=crop_top,
                crop_height=crop_height,
                crop_width=crop_width,
                resized_height=resized_height,
                resized_width=resized_width,
            )
        )

    max_height = max(tensor.shape[1] for tensor in tensors)
    max_width = max(tensor.shape[2] for tensor in tensors)
    padded_tensors = []
    padded_transforms = []
    for tensor, transform in zip(tensors, transforms):
        height_padding = max_height - tensor.shape[1]
        width_padding = max_width - tensor.shape[2]
        pad_top = height_padding // 2
        pad_bottom = height_padding - pad_top
        pad_left = width_padding // 2
        pad_right = width_padding - pad_left
        if height_padding or width_padding:
            tensor = functional.pad(
                tensor,
                (pad_left, pad_right, pad_top, pad_bottom),
                mode="constant",
                value=1.0,
            )
        padded_tensors.append(tensor)
        padded_transforms.append(replace(transform, pad_top=pad_top, pad_left=pad_left))

    return ModelBatch(torch.stack(padded_tensors), tuple(padded_transforms))


def save_rgb_image(path: Path, image: np.ndarray) -> None:
    """Create the destination directory and write an RGB image with OpenCV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not success:
        raise OSError(f"Failed to write panorama: {path}")


def _crop_to_supported_aspect_ratio(
    image: Image.Image,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
) -> tuple[Image.Image, int, int]:
    width, height = image.size
    aspect_ratio = height / width
    if aspect_ratio < min_aspect_ratio:
        crop_width = min(width, max(1, int(round(height / min_aspect_ratio))))
        left = max((width - crop_width) // 2, 0)
        return image.crop((left, 0, left + crop_width, height)), left, 0
    if aspect_ratio > max_aspect_ratio:
        crop_height = min(height, max(1, int(round(width * max_aspect_ratio))))
        top = max((height - crop_height) // 2, 0)
        return image.crop((0, top, width, top + crop_height)), 0, top
    return image, 0, 0


def _balanced_target_shape(aspect_ratio: float, resolution: int, patch_size: int) -> tuple[int, int]:
    token_count = (resolution // patch_size) ** 2
    width_patches = max(1, int(np.round(np.sqrt(token_count / aspect_ratio))))
    height_patches = max(1, int(np.round(token_count / width_patches)))
    return height_patches * patch_size, width_patches * patch_size


def _max_size_target_shape(aspect_ratio: float, resolution: int, patch_size: int) -> tuple[int, int]:
    if aspect_ratio >= 1.0:
        height = resolution
        width = _round_to_patch_multiple(resolution / aspect_ratio, patch_size)
    else:
        width = resolution
        height = _round_to_patch_multiple(resolution * aspect_ratio, patch_size)
    return height, width


def _round_to_patch_multiple(value: float, patch_size: int) -> int:
    return max(patch_size, int(np.round(value / patch_size)) * patch_size)
