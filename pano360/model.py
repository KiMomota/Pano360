from __future__ import annotations

import gc
import logging
from pathlib import Path

import torch

from vggt_omega.models import VGGTOmega


LOGGER = logging.getLogger(__name__)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested)
    if device.type == "cpu":
        LOGGER.warning("VGGT-Omega-1B is running on CPU; inference can be very slow")
    return device


def load_camera_model(checkpoint_path: Path, device: torch.device) -> VGGTOmega:
    """Load only VGGT-Omega's camera path without duplicating 1B parameters in RAM."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"VGGT-Omega checkpoint not found: {checkpoint_path}")

    with torch.device("meta"):
        model = VGGTOmega(enable_depth=False).eval()

    state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
    incompatible = model.load_state_dict(state_dict, strict=False, assign=True)
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith("dense_head.")]
    if incompatible.missing_keys or unexpected:
        raise RuntimeError(
            "Incompatible VGGT-Omega checkpoint: "
            f"missing={incompatible.missing_keys}, unexpected={unexpected}"
        )

    del state_dict
    gc.collect()
    return model.to(device).eval()


def predict_camera_poses(model: VGGTOmega, images: torch.Tensor, device: torch.device) -> dict:
    """Run camera inference and keep only values consumed by the stitcher."""
    with torch.inference_mode():
        predictions = model(images.to(device))
    return {"pose_enc": predictions["pose_enc"]}


def release_device_memory(device: torch.device) -> None:
    """Collect released model tensors and clear the CUDA allocator cache."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
