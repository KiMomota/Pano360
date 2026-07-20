from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn.functional as functional

from .types import WarpedImages


LOGGER = logging.getLogger(__name__)
DEFAULT_SEAM_CANVAS_MEGAPIXELS = 1.0


def find_seams(
    warped: WarpedImages,
    method: str = "torch_dp",
    scale: float = 0.5,
    megapixels: float | None = None,
) -> WarpedImages:
    """Estimate seam masks on a compact CUDA canvas and recover full resolution."""
    if method == "no":
        return warped
    if method == "torch_soft":
        warped.masks = _soft_center_masks(warped)
        return warped
    if method != "torch_dp":
        raise ValueError(f"Unsupported PyTorch seam method: {method}")

    left, top, canvas_width, canvas_height = warped.canvas_roi
    seam_scale = _seam_scale(canvas_width, canvas_height, scale, megapixels)
    small_width = max(1, int(math.ceil(canvas_width * seam_scale)))
    small_height = max(1, int(math.ceil(canvas_height * seam_scale)))
    LOGGER.info(
        "Torch DP seam canvas: %dx%d (scale %.4f)",
        small_width,
        small_height,
        seam_scale,
    )

    small_images: list[torch.Tensor] = []
    small_masks: list[torch.Tensor] = []
    small_rois: list[tuple[int, int, int, int]] = []
    centers: list[tuple[float, float]] = []
    for image, mask, corner, size in zip(
        warped.images, warped.exposure_masks, warped.corners, warped.sizes
    ):
        roi_width = max(1, int(round(size[0] * seam_scale)))
        roi_height = max(1, int(round(size[1] * seam_scale)))
        offset_x = int(round((corner[0] - left) * seam_scale))
        offset_y = int(round((corner[1] - top) * seam_scale))
        roi_width = min(roi_width, small_width - offset_x)
        roi_height = min(roi_height, small_height - offset_y)
        resized_image = functional.interpolate(
            image.float().unsqueeze(0),
            size=(roi_height, roi_width),
            mode="area",
        ).squeeze(0)
        resized_mask = functional.interpolate(
            mask.float().unsqueeze(0),
            size=(roi_height, roi_width),
            mode="nearest",
        ).squeeze(0)
        canvas_image = torch.zeros(
            (3, small_height, small_width), device=warped.device, dtype=torch.float32
        )
        canvas_mask = torch.zeros(
            (small_height, small_width), device=warped.device, dtype=torch.bool
        )
        canvas_image[
            :, offset_y : offset_y + roi_height, offset_x : offset_x + roi_width
        ] = resized_image
        canvas_mask[
            offset_y : offset_y + roi_height, offset_x : offset_x + roi_width
        ] = resized_mask[0] > 0.5
        small_images.append(canvas_image)
        small_masks.append(canvas_mask)
        small_rois.append((offset_x, offset_y, roi_width, roi_height))
        centers.append((offset_x + roi_width / 2, offset_y + roi_height / 2))

    labels = _assign_labels(small_images, small_masks, centers)
    recovered_masks: list[torch.Tensor] = []
    for index, (roi, original_mask) in enumerate(zip(small_rois, warped.exposure_masks)):
        offset_x, offset_y, roi_width, roi_height = roi
        selected = (
            labels[offset_y : offset_y + roi_height, offset_x : offset_x + roi_width]
            == index
        ).float()[None, None]
        # A one-pixel low-resolution overlap gives the Gaussian pyramid enough
        # support to blend across the otherwise hard DP boundary.
        selected = functional.max_pool2d(selected, kernel_size=3, stride=1, padding=1)
        selected = functional.interpolate(
            selected,
            size=original_mask.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        recovered_masks.append(selected.to(original_mask.dtype).mul_(original_mask))
    warped.masks = recovered_masks
    return warped


def _assign_labels(
    images: list[torch.Tensor],
    masks: list[torch.Tensor],
    centers: list[tuple[float, float]],
) -> torch.Tensor:
    height, width = masks[0].shape
    device = masks[0].device
    labels = torch.full((height, width), -1, device=device, dtype=torch.int16)
    start = max(range(len(masks)), key=lambda index: int(masks[index].sum().item()))
    labels[masks[start]] = start
    coverage = masks[start].clone()
    composite = images[start].clone()
    processed = {start}

    while len(processed) < len(images):
        remaining = [index for index in range(len(images)) if index not in processed]
        overlap_counts = {
            index: int((masks[index] & coverage).sum().item()) for index in remaining
        }
        index = max(remaining, key=lambda candidate: overlap_counts[candidate])
        image_mask = masks[index]
        overlap = image_mask & coverage
        uncovered = image_mask & ~coverage

        if overlap.any():
            existing_labels = labels[overlap].long()
            existing_labels = existing_labels[existing_labels >= 0]
            reference = int(torch.bincount(existing_labels, minlength=len(images)).argmax().item())
            rows, columns = torch.where(overlap)
            top = int(rows.min().item())
            bottom = int(rows.max().item()) + 1
            left = int(columns.min().item())
            right = int(columns.max().item()) + 1

            overlap_crop = overlap[top:bottom, left:right]
            current_crop = composite[:, top:bottom, left:right]
            candidate_crop = images[index][:, top:bottom, left:right]
            cost = _seam_cost(current_crop, candidate_crop, overlap_crop)
            delta_x = centers[index][0] - centers[reference][0]
            delta_y = centers[index][1] - centers[reference][1]
            if abs(delta_x) >= abs(delta_y):
                path = _minimum_vertical_path(cost)
                columns_grid = torch.arange(
                    right - left, device=device, dtype=path.dtype
                )[None]
                choose_candidate = columns_grid >= path[:, None]
                if delta_x < 0:
                    choose_candidate = ~choose_candidate
            else:
                path = _minimum_vertical_path(cost.T)
                rows_grid = torch.arange(
                    bottom - top, device=device, dtype=path.dtype
                )[:, None]
                choose_candidate = rows_grid >= path[None, :]
                if delta_y < 0:
                    choose_candidate = ~choose_candidate

            overlap_selection = torch.zeros_like(overlap)
            overlap_selection[top:bottom, left:right] = choose_candidate & overlap_crop
            selected = uncovered | overlap_selection
        else:
            selected = uncovered

        labels[selected] = index
        composite[:, selected] = images[index][:, selected]
        coverage |= image_mask
        processed.add(index)

    return labels


def _seam_cost(
    first: torch.Tensor,
    second: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    color = (first - second).abs().mean(dim=0)
    first_gray = first.mean(dim=0, keepdim=True).unsqueeze(0)
    second_gray = second.mean(dim=0, keepdim=True).unsqueeze(0)
    first_gradient = _gradient_magnitude(first_gray)
    second_gradient = _gradient_magnitude(second_gray)
    gradient = (first_gradient - second_gradient).abs().squeeze()
    cost = color + 0.25 * gradient
    cost = functional.avg_pool2d(
        cost[None, None], kernel_size=3, stride=1, padding=1
    ).squeeze()
    finite_max = cost[valid].max() if valid.any() else cost.max()
    return torch.where(valid, cost, finite_max + 1000.0)


def _gradient_magnitude(image: torch.Tensor) -> torch.Tensor:
    horizontal = functional.pad(image[..., 1:] - image[..., :-1], (0, 1, 0, 0))
    vertical = functional.pad(image[..., 1:, :] - image[..., :-1, :], (0, 0, 0, 1))
    return horizontal.abs() + vertical.abs()


def _minimum_vertical_path(cost: torch.Tensor) -> torch.Tensor:
    """Run the expensive DP recurrence on GPU and only backtrack its tiny graph on CPU."""
    height, width = cost.shape
    parents = torch.empty((height, width), device=cost.device, dtype=torch.int8)
    accumulated = cost[0]
    parents[0].zero_()
    infinity = torch.finfo(cost.dtype).max / 16
    for row in range(1, height):
        padded = functional.pad(accumulated, (1, 1), value=infinity)
        candidates = torch.stack((padded[:-2], padded[1:-1], padded[2:]))
        minimum, parent = candidates.min(dim=0)
        accumulated = cost[row] + minimum
        parents[row] = parent.to(torch.int8) - 1

    column = int(accumulated.argmin().item())
    parents_cpu = parents.cpu().numpy()
    path = np.empty(height, dtype=np.int64)
    path[-1] = column
    for row in range(height - 1, 0, -1):
        column += int(parents_cpu[row, column])
        column = min(max(column, 0), width - 1)
        path[row - 1] = column
    return torch.from_numpy(path).to(device=cost.device)


def _soft_center_masks(warped: WarpedImages) -> list[torch.Tensor]:
    results = []
    for mask in warped.exposure_masks:
        height, width = mask.shape[-2:]
        y = torch.linspace(0, torch.pi, height, device=mask.device, dtype=torch.float32)
        x = torch.linspace(0, torch.pi, width, device=mask.device, dtype=torch.float32)
        confidence = torch.sin(y)[:, None] * torch.sin(x)[None, :]
        results.append(confidence[None].to(mask.dtype).mul_(mask))
    return results


def _seam_scale(
    width: int,
    height: int,
    requested_scale: float,
    megapixels: float | None,
) -> float:
    target_megapixels = (
        float(megapixels) if megapixels is not None else DEFAULT_SEAM_CANVAS_MEGAPIXELS
    )
    target_scale = math.sqrt(target_megapixels * 1_000_000 / float(width * height))
    return max(1e-3, min(1.0, float(requested_scale), target_scale))
