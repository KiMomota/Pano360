"""Optional LightGlue feature extraction and robust pairwise matching."""

from __future__ import annotations

from collections import defaultdict
import logging

import cv2
import numpy as np
import torch


LOGGER = logging.getLogger(__name__)


def match_image_pairs(
    images: list[np.ndarray],
    image_tensors: list[torch.Tensor],
    max_query_points: int = 2048,
    extractor_name: str = "aliked",
) -> dict:
    """Match all image pairs, retaining only homography-supported inliers."""
    extractor, matcher = _create_models(extractor_name, max_query_points)
    features = [_extract_features(tensor, extractor) for tensor in image_tensors]
    keypoints = [feature["keypoints"].squeeze(0).cpu().numpy() for feature in features]
    centered_keypoints = [
        points - np.array([image.shape[1], image.shape[0]], dtype=np.float32) / 2
        for points, image in zip(keypoints, images)
    ]

    matches = defaultdict(dict)
    for source in range(len(images)):
        for target in range(source + 1, len(images)):
            with torch.inference_mode():
                prediction = matcher({"image0": features[source], "image1": features[target]})
            pairs = prediction["matches"][0].detach().cpu().numpy().astype(np.int64, copy=False)
            if len(pairs) < 4:
                continue

            source_points = centered_keypoints[source][pairs[:, 0]].astype(np.float32)
            target_points = centered_keypoints[target][pairs[:, 1]].astype(np.float32)
            homography, inlier_mask = cv2.findHomography(source_points, target_points, cv2.RANSAC)
            if homography is None or inlier_mask is None:
                continue
            inlier_pairs = pairs[inlier_mask.ravel().astype(bool)]
            if len(inlier_pairs) < 4 or abs(np.linalg.det(homography)) < 1e-12:
                continue

            forward_points = _pairs_to_homogeneous(
                inlier_pairs, centered_keypoints[source], centered_keypoints[target]
            )
            reverse_pairs = np.fliplr(inlier_pairs)
            reverse_points = _pairs_to_homogeneous(
                reverse_pairs, centered_keypoints[target], centered_keypoints[source]
            )
            score = len(inlier_pairs)
            LOGGER.info(
                "LightGlue pair %d-%d: %d matches, %d homography inliers",
                source,
                target,
                len(pairs),
                score,
            )
            matches[source][target] = (forward_points, homography, score)
            matches[target][source] = (reverse_points, np.linalg.inv(homography), score)

    if not matches:
        raise RuntimeError("LightGlue found no geometrically valid image pairs")
    return {index: dict(neighbors) for index, neighbors in matches.items()}


def _create_models(extractor_name: str, max_query_points: int):
    try:
        from lightglue import ALIKED, SIFT, LightGlue, SuperPoint
    except ImportError as error:
        raise RuntimeError(
            "demo_stitch_ba requires LightGlue. Install requirements-ba.txt first."
        ) from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if extractor_name == "aliked":
        extractor = ALIKED(max_num_keypoints=max_query_points, detection_threshold=0.005)
        feature_type = "aliked"
    elif extractor_name == "superpoint":
        extractor = SuperPoint(max_num_keypoints=max_query_points, detection_threshold=0.005)
        feature_type = "superpoint"
    elif extractor_name == "sift":
        extractor = SIFT(max_num_keypoints=max_query_points)
        feature_type = "sift"
    else:
        raise ValueError(f"Unsupported feature extractor: {extractor_name}")
    return extractor.to(device).eval(), LightGlue(features=feature_type).to(device).eval()


def _extract_features(image: torch.Tensor, extractor):
    if image.ndim != 3 or image.shape[0] not in {1, 3}:
        raise ValueError(f"Expected CHW image tensor, got {tuple(image.shape)}")
    if image.shape[0] == 3:
        red, green, blue = image
        gray = 0.299 * red + 0.587 * green + 0.114 * blue
    else:
        gray = image[0]
    with torch.inference_mode():
        return extractor.extract(gray[None, None])


def _pairs_to_homogeneous(pairs, source_keypoints, target_keypoints) -> np.ndarray:
    source = source_keypoints[pairs[:, 0]]
    target = target_keypoints[pairs[:, 1]]
    ones = np.ones((len(pairs), 1), dtype=np.float64)
    return np.concatenate([source, ones, target, ones], axis=1)
