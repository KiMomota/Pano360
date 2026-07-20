from __future__ import annotations

import cv2
import numpy as np

from vggt_omega.utils.geometry import closed_form_inverse_se3
from vggt_omega.utils.pose_enc import encoding_to_camera

from .io import ImageTransform


def decode_cameras(
    predictions: dict,
    model_size_hw: tuple[int, int],
    transforms: tuple[ImageTransform, ...],
    share_intrinsics: bool = False,
    for_bundle_adjustment: bool = False,
) -> list[cv2.detail.CameraParams]:
    """Decode model cameras and map padded model intrinsics to original pixels."""
    extrinsics, intrinsics = encoding_to_camera(predictions["pose_enc"], model_size_hw)
    extrinsics = extrinsics.detach().float().cpu().numpy().squeeze(0)
    intrinsics = intrinsics.detach().float().cpu().numpy().squeeze(0)

    if len(transforms) != len(extrinsics):
        raise ValueError("Camera prediction count does not match the input image count")

    # VGGT-Omega extrinsics are camera-from-world. OpenCV's rotation warper uses
    # camera-to-world rotations, while our BA residuals use camera-from-world.
    camera_matrices = extrinsics if for_bundle_adjustment else closed_form_inverse_se3(extrinsics)

    cameras = []
    for index, transform in enumerate(transforms):
        intrinsic = intrinsics[0 if share_intrinsics else index]
        fx = float(intrinsic[0, 0]) / transform.scale_x
        fy = float(intrinsic[1, 1]) / transform.scale_y
        ppx = (float(intrinsic[0, 2]) - transform.pad_left) / transform.scale_x + transform.crop_left
        ppy = (float(intrinsic[1, 2]) - transform.pad_top) / transform.scale_y + transform.crop_top

        camera = cv2.detail.CameraParams()
        camera.R = camera_matrices[index, :3, :3].astype(np.float32)
        camera.t = camera_matrices[index, :3, 3].astype(np.float32)
        camera.focal = fx
        camera.ppx = ppx
        camera.ppy = ppy
        camera.aspect = fy / fx
        cameras.append(camera)
    return cameras


def cameras_from_bundle_adjustment(
    adjusted_cameras,
    image_shapes: list[tuple[int, int]],
) -> list[cv2.detail.CameraParams]:
    """Convert optimized camera-from-world rotations/intrinsics for OpenCV warping."""
    if len(adjusted_cameras) != len(image_shapes):
        raise ValueError("Bundle-adjusted camera count does not match the input image count")

    cameras = []
    for adjusted, (height, width) in zip(adjusted_cameras, image_shapes):
        intrinsic = adjusted.intrinsic
        fx = float(intrinsic[0, 0])
        fy = float(intrinsic[1, 1])
        camera = cv2.detail.CameraParams()
        camera.R = adjusted.rotation.T.astype(np.float32)
        camera.t = np.zeros(3, dtype=np.float32)
        camera.focal = fx
        camera.ppx = float(intrinsic[0, 2] + width / 2)
        camera.ppy = float(intrinsic[1, 2] + height / 2)
        camera.aspect = fy / fx
        cameras.append(camera)
    return cameras
