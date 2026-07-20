from __future__ import annotations

import math

import torch
import torch.nn.functional as functional


PLANET_DISC_RADIUS = 0.90
PLANET_EDGE_FEATHER_PIXELS = 2.0


def render_view(
    panorama: torch.Tensor,
    mode: str = "normal",
    size: int = 2048,
    rotation_degrees: float = 0.0,
    zoom: float = 0.65,
    fisheye_fov_degrees: float = 180.0,
    cubemap_face_size: int = 1024,
) -> torch.Tensor:
    """Render an optional display view while keeping pixels on CUDA."""
    if mode == "normal":
        return panorama
    rotation = math.radians(rotation_degrees)
    if mode in {"little_planet", "rabbit_hole"}:
        return _planet_view(panorama, size, rotation, zoom, mode == "rabbit_hole")
    if mode == "fisheye":
        return _fisheye_view(panorama, size, rotation, fisheye_fov_degrees)
    if mode == "cubemap":
        return _cubemap_atlas(panorama, cubemap_face_size, rotation)
    raise ValueError(f"Unsupported panorama view: {mode}")


def _planet_view(
    panorama: torch.Tensor,
    size: int,
    rotation: float,
    zoom: float,
    rabbit_hole: bool,
) -> torch.Tensor:
    y, x = _square_coordinates(size, panorama.device, panorama.dtype)
    output_radius = torch.sqrt(x.square() + y.square())
    # Fit the old unit-radius planet inside a smaller output disc. This keeps
    # the circular projection fully visible instead of clipping it at the
    # horizontal/vertical canvas edges.
    projection_radius = output_radius / PLANET_DISC_RADIUS
    angular_distance = (2.0 * torch.atan(projection_radius / zoom)).clamp_max(torch.pi)
    polar = angular_distance if rabbit_hole else torch.pi - angular_distance
    longitude = torch.atan2(x, -y) + rotation
    result = _sample_angles(panorama, longitude, polar)

    # Reserve a visible black border on all four sides. A short feather avoids
    # stair-stepping along the circular boundary while keeping the outermost
    # pixels exactly black.
    coordinate_per_pixel = 2.0 / max(size - 1, 1)
    feather = PLANET_EDGE_FEATHER_PIXELS * coordinate_per_pixel
    alpha = ((PLANET_DISC_RADIUS - output_radius) / max(feather, 1e-8)).clamp(0.0, 1.0)
    return result * alpha.unsqueeze(0)


def _fisheye_view(
    panorama: torch.Tensor,
    size: int,
    rotation: float,
    fov_degrees: float,
) -> torch.Tensor:
    y, x = _square_coordinates(size, panorama.device, panorama.dtype)
    radius = torch.sqrt(x.square() + y.square())
    angle = radius * math.radians(fov_degrees) / 2.0
    safe_radius = radius.clamp_min(1e-8)
    sin_angle = torch.sin(angle)
    direction_x = sin_angle * x / safe_radius
    direction_y = sin_angle * y / safe_radius
    direction_z = torch.cos(angle)
    result = _sample_directions(
        panorama,
        torch.stack((direction_x, direction_y, direction_z)),
        rotation,
    )
    return result * (radius <= 1.0).to(result.dtype).unsqueeze(0)


def _cubemap_atlas(
    panorama: torch.Tensor,
    face_size: int,
    rotation: float,
) -> torch.Tensor:
    y, x = _square_coordinates(face_size, panorama.device, panorama.dtype)
    faces = (
        torch.stack((torch.ones_like(x), y, -x)),       # +X
        torch.stack((-torch.ones_like(x), y, x)),       # -X
        torch.stack((x, torch.ones_like(x), -y)),       # +Y (nadir)
        torch.stack((x, -torch.ones_like(x), y)),       # -Y (zenith)
        torch.stack((x, y, torch.ones_like(x))),        # +Z
        torch.stack((-x, y, -torch.ones_like(x))),      # -Z
    )
    rendered = [_sample_directions(panorama, direction, rotation) for direction in faces]
    return torch.cat((torch.cat(rendered[:3], dim=2), torch.cat(rendered[3:], dim=2)), dim=1)


def _sample_directions(
    panorama: torch.Tensor,
    direction: torch.Tensor,
    rotation: float,
) -> torch.Tensor:
    direction = direction / torch.linalg.vector_norm(direction, dim=0, keepdim=True).clamp_min(1e-8)
    longitude = torch.atan2(direction[0], direction[2]) + rotation
    polar = torch.acos((-direction[1]).clamp(-1.0, 1.0))
    return _sample_angles(panorama, longitude, polar)


def _sample_angles(
    panorama: torch.Tensor,
    longitude: torch.Tensor,
    polar: torch.Tensor,
) -> torch.Tensor:
    """Sample ERP angles with a one-column periodic horizontal border."""
    longitude = torch.atan2(torch.sin(longitude), torch.cos(longitude))
    height, width = panorama.shape[-2:]
    periodic = torch.cat((panorama[:, :, -1:], panorama, panorama[:, :, :1]), dim=2)
    pixel_x = (longitude + torch.pi) * (width / (2.0 * torch.pi)) + 1.0
    pixel_y = polar.clamp(0.0, torch.pi) * ((height - 1) / torch.pi)
    grid = torch.stack(
        (
            pixel_x.mul(2.0 / (width + 1)).sub(1.0),
            pixel_y.mul(2.0 / max(height - 1, 1)).sub(1.0),
        ),
        dim=-1,
    ).unsqueeze(0)
    return functional.grid_sample(
        periodic.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).squeeze(0)


def _square_coordinates(
    size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.linspace(-1.0, 1.0, size, device=device, dtype=dtype)
    return torch.meshgrid(axis, axis, indexing="ij")
