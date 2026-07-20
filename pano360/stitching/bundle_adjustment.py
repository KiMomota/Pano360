"""Robust rotation/focal bundle adjustment for panorama cameras."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np


LOGGER = logging.getLogger(__name__)
FOCAL_SCALE_LIMIT = 0.30
ROBUST_LOSS_SCALE_PX = 3.0


@dataclass
class AdjustedCamera:
    """Camera-from-world rotation and centered pixel intrinsics."""

    rotation: np.ndarray
    intrinsic: np.ndarray


def bundle_adjust(
    images: list[np.ndarray],
    matches: dict,
    cameras: list[cv2.detail.CameraParams],
    max_iterations: int = 200,
    straighten_result: bool = True,
) -> list[AdjustedCamera]:
    """Jointly refine camera rotations and focal lengths from pairwise matches.

    The highest-connectivity camera is held fixed to remove the global-rotation
    gauge freedom. LightGlue matches have already passed homography RANSAC, so
    all connected edges are retained and a robust loss limits remaining outliers.
    """
    if len(images) != len(cameras):
        raise ValueError("Image count does not match camera count")
    if len(images) < 2:
        raise ValueError("Bundle adjustment requires at least two images")

    edges = _unique_edges(matches)
    if not edges:
        raise RuntimeError("Bundle adjustment has no valid feature-match edges")
    _ensure_connected(len(images), edges)

    anchor = _select_anchor(len(images), edges)
    reference_rotation = cameras[anchor].R.astype(np.float64)
    adjusted = [
        _opencv_to_adjusted(
            camera,
            image.shape[:2],
            camera.R.astype(np.float64) @ reference_rotation.T,
        )
        for image, camera in zip(images, cameras)
    ]
    adjusted[anchor].rotation = np.eye(3, dtype=np.float64)

    initial_residuals = _residuals(adjusted, edges)
    LOGGER.info(
        "Optimizing %d cameras from %d match edges (%d inliers); initial RMSE %.4f px",
        len(adjusted),
        len(edges),
        sum(len(points) for _, _, points in edges),
        _loss(initial_residuals),
    )

    try:
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation
    except ImportError as error:
        raise RuntimeError("Bundle adjustment requires scipy>=1.10") from error

    variable_indices = [index for index in range(len(adjusted)) if index != anchor]
    initial_focals = np.array([camera.intrinsic[0, 0] for camera in adjusted])
    rotation_parameters = np.concatenate(
        [Rotation.from_matrix(adjusted[index].rotation).as_rotvec() for index in variable_indices]
    )
    initial_parameters = np.concatenate([rotation_parameters, np.zeros(len(adjusted))])

    rotation_parameter_count = len(rotation_parameters)
    lower_bounds = np.full_like(initial_parameters, -np.inf)
    upper_bounds = np.full_like(initial_parameters, np.inf)
    lower_bounds[rotation_parameter_count:] = np.log(1.0 - FOCAL_SCALE_LIMIT)
    upper_bounds[rotation_parameter_count:] = np.log(1.0 + FOCAL_SCALE_LIMIT)

    def cameras_from_parameters(parameters: np.ndarray) -> list[AdjustedCamera]:
        trial = [
            AdjustedCamera(camera.rotation.copy(), camera.intrinsic.copy())
            for camera in adjusted
        ]
        rotation_vectors = parameters[:rotation_parameter_count].reshape(-1, 3)
        for index, vector in zip(variable_indices, rotation_vectors):
            trial[index].rotation = Rotation.from_rotvec(vector).as_matrix()
        for index, log_scale in enumerate(parameters[rotation_parameter_count:]):
            scale = np.exp(log_scale)
            trial[index].intrinsic[0, 0] = initial_focals[index] * scale
            trial[index].intrinsic[1, 1] = adjusted[index].intrinsic[1, 1] * scale
        return trial

    optimization = least_squares(
        lambda parameters: _residuals(cameras_from_parameters(parameters), edges),
        initial_parameters,
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=ROBUST_LOSS_SCALE_PX,
        x_scale="jac",
        max_nfev=max_iterations,
    )
    result = cameras_from_parameters(optimization.x)
    final_residuals = _residuals(result, edges)
    LOGGER.info(
        "Bundle adjustment final RMSE %.4f px after %d evaluations (%s)",
        _loss(final_residuals),
        optimization.nfev,
        optimization.message,
    )

    if straighten_result:
        rotations = _straighten_rotations([camera.rotation for camera in result])
        for camera, rotation in zip(result, rotations):
            camera.rotation = rotation
    return result


def _unique_edges(matches: dict) -> list[tuple[int, int, np.ndarray]]:
    edges = []
    for source, neighbors in matches.items():
        for target, (points, _, _) in neighbors.items():
            if source < target and len(points) >= 4:
                edges.append((source, target, points.astype(np.float64, copy=False)))
    return edges


def _ensure_connected(camera_count: int, edges: list[tuple[int, int, np.ndarray]]) -> None:
    adjacency = [set() for _ in range(camera_count)]
    for source, target, _ in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    reached = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - reached:
            reached.add(neighbor)
            frontier.append(neighbor)
    if len(reached) != camera_count:
        missing = sorted(set(range(camera_count)) - reached)
        raise RuntimeError(f"Feature match graph is disconnected; unmatched image indices: {missing}")


def _select_anchor(camera_count: int, edges: list[tuple[int, int, np.ndarray]]) -> int:
    scores = np.zeros(camera_count, dtype=np.int64)
    for source, target, points in edges:
        scores[source] += len(points)
        scores[target] += len(points)
    return int(np.argmax(scores))


def _opencv_to_adjusted(
    camera: cv2.detail.CameraParams,
    image_shape: tuple[int, int],
    rotation: np.ndarray,
) -> AdjustedCamera:
    height, width = image_shape
    fx = float(camera.focal)
    fy = fx * float(camera.aspect)
    intrinsic = np.array(
        [
            [fx, 0.0, float(camera.ppx) - width / 2],
            [0.0, fy, float(camera.ppy) - height / 2],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return AdjustedCamera(rotation=rotation, intrinsic=intrinsic)


def _homography(first: AdjustedCamera, second: AdjustedCamera) -> np.ndarray:
    """Return the pure-rotation homography mapping second into first."""
    return (
        first.intrinsic
        @ first.rotation
        @ second.rotation.T
        @ np.linalg.inv(second.intrinsic)
    )


def _camera_difference(
    first: AdjustedCamera,
    second: AdjustedCamera,
    matched_points: np.ndarray,
) -> np.ndarray:
    transformed = _homography(first, second) @ matched_points[:, 3:6].T
    safe_depth = np.where(np.abs(transformed[2]) < 1e-12, 1e-12, transformed[2])
    normalized = transformed[:2] / safe_depth
    return (matched_points[:, :2].T - normalized).T.ravel()


def _residuals(
    cameras: list[AdjustedCamera],
    edges: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    return np.concatenate(
        [_camera_difference(cameras[source], cameras[target], points) for source, target, points in edges]
    )


def _loss(residuals: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(residuals))))


def _straighten_rotations(rotations: list[np.ndarray]) -> list[np.ndarray]:
    covariance = np.cov(np.stack([rotation[0] for rotation in rotations], axis=-1))
    _, _, right_vectors = np.linalg.svd(covariance)
    vertical = right_vectors[2]
    forward = np.sum(np.stack([rotation[2] for rotation in rotations]), axis=0)
    horizontal = np.cross(vertical, forward)
    horizontal /= np.linalg.norm(horizontal)
    forward = np.cross(horizontal, vertical)
    if np.sum([horizontal @ rotation[0] for rotation in rotations]) < 0:
        horizontal, vertical = -horizontal, -vertical
    global_rotation = np.stack([horizontal, vertical, forward], axis=-1)
    return [rotation @ global_rotation for rotation in rotations]
