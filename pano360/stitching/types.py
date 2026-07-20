from __future__ import annotations

from dataclasses import dataclass

import cv2
import torch


@dataclass
class WarpedImages:
    """Warped image ROIs kept on one torch device.

    Images use CHW layout and values in [0, 1]. Masks use 1HW layout and
    floating values in [0, 1]. Corners remain in panorama pixel coordinates.
    """

    corners: list[tuple[int, int]]
    sizes: list[tuple[int, int]]
    images: list[torch.Tensor]
    masks: list[torch.Tensor]
    exposure_masks: list[torch.Tensor]
    cameras: list[cv2.detail.CameraParams]
    projection: str
    device: torch.device
    canvas_override: tuple[int, int, int, int] | None = None

    @property
    def canvas_roi(self) -> tuple[int, int, int, int]:
        if self.canvas_override is not None:
            return self.canvas_override
        left = min(corner[0] for corner in self.corners)
        top = min(corner[1] for corner in self.corners)
        right = max(corner[0] + size[0] for corner, size in zip(self.corners, self.sizes))
        bottom = max(corner[1] + size[1] for corner, size in zip(self.corners, self.sizes))
        return left, top, right - left, bottom - top


def overlap_slices(
    first_corner: tuple[int, int],
    first_size: tuple[int, int],
    second_corner: tuple[int, int],
    second_size: tuple[int, int],
) -> tuple[tuple[slice, slice], tuple[slice, slice]] | None:
    """Return matching YX slices for the intersection of two panorama ROIs."""
    left = max(first_corner[0], second_corner[0])
    top = max(first_corner[1], second_corner[1])
    right = min(first_corner[0] + first_size[0], second_corner[0] + second_size[0])
    bottom = min(first_corner[1] + first_size[1], second_corner[1] + second_size[1])
    if left >= right or top >= bottom:
        return None
    first = (
        slice(top - first_corner[1], bottom - first_corner[1]),
        slice(left - first_corner[0], right - first_corner[0]),
    )
    second = (
        slice(top - second_corner[1], bottom - second_corner[1]),
        slice(left - second_corner[0], right - second_corner[0]),
    )
    return first, second
