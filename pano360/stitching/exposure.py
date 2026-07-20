"""Small global exposure solve with overlap statistics computed on CUDA."""

from __future__ import annotations

import logging

import torch

from .types import WarpedImages, overlap_slices


LOGGER = logging.getLogger(__name__)


def estimate_exposure_gains(warped: WarpedImages) -> torch.Tensor:
    """Estimate one robust luminance gain per image without downloading pixels."""
    image_count = len(warped.images)
    device = warped.device
    rows: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []

    for first in range(image_count):
        for second in range(first + 1, image_count):
            overlap = overlap_slices(
                warped.corners[first],
                warped.sizes[first],
                warped.corners[second],
                warped.sizes[second],
            )
            if overlap is None:
                continue
            first_slice, second_slice = overlap
            first_mask = warped.exposure_masks[first][
                :, first_slice[0], first_slice[1]
            ] > 0.5
            second_mask = warped.exposure_masks[second][
                :, second_slice[0], second_slice[1]
            ] > 0.5
            valid = first_mask & second_mask
            count = valid.sum()
            if int(count.item()) < 256:
                continue

            first_rgb = warped.images[first][:, first_slice[0], first_slice[1]].float()
            second_rgb = warped.images[second][:, second_slice[0], second_slice[1]].float()
            first_luminance = (
                0.299 * first_rgb[0] + 0.587 * first_rgb[1] + 0.114 * first_rgb[2]
            )
            second_luminance = (
                0.299 * second_rgb[0] + 0.587 * second_rgb[1] + 0.114 * second_rgb[2]
            )
            valid_2d = valid[0]
            first_mean = first_luminance[valid_2d].mean().clamp_min(1e-4)
            second_mean = second_luminance[valid_2d].mean().clamp_min(1e-4)
            row = torch.zeros(image_count, device=device, dtype=torch.float32)
            row[first] = 1.0
            row[second] = -1.0
            rows.append(row)
            values.append(torch.log(second_mean) - torch.log(first_mean))
            weights.append(count.float())

    if not rows:
        return torch.ones(image_count, device=device, dtype=torch.float32)

    maximum_weight = torch.stack(weights).max().clamp_min(1.0)
    weighted_rows = []
    weighted_values = []
    for row, value, weight in zip(rows, values, weights):
        factor = torch.sqrt(weight / maximum_weight)
        weighted_rows.append(row * factor)
        weighted_values.append(value * factor)

    # Fix the otherwise free global exposure scale to image zero.
    anchor = torch.zeros(image_count, device=device, dtype=torch.float32)
    anchor[0] = 10.0
    weighted_rows.append(anchor)
    weighted_values.append(torch.zeros((), device=device, dtype=torch.float32))
    matrix = torch.stack(weighted_rows)
    target = torch.stack(weighted_values)
    normal = matrix.T @ matrix + torch.eye(image_count, device=device) * 1e-5
    solution = torch.linalg.solve(normal, matrix.T @ target)
    gains = torch.exp(solution).clamp(0.5, 2.0)
    LOGGER.info("Torch exposure gains: %s", [round(float(value), 4) for value in gains])
    return gains
