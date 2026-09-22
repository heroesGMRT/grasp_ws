"""Robust metric-depth estimation for a detected bounding box."""

import numpy as np


def measure_bbox_depth(
    depth_image: np.ndarray,
    bbox: tuple[int, int, int, int],
    intrinsics: tuple[float, float, float, float],
    scale_to_m: float,
    depth_min_m: float,
    depth_max_m: float,
    min_valid_samples: int,
) -> tuple[float, float, float] | None:
    """Return camera-frame XYZ from valid depth in a bbox's inner half."""
    fx, fy, cx, cy = intrinsics
    if fx <= 0.0 or fy <= 0.0:
        raise ValueError("camera focal lengths must be positive")

    image_h, image_w = depth_image.shape[:2]
    x, y, width, height = bbox
    x1 = max(0, min(image_w, x))
    y1 = max(0, min(image_h, y))
    x2 = max(x1, min(image_w, x + width))
    y2 = max(y1, min(image_h, y + height))
    if x2 <= x1 or y2 <= y1:
        return None

    inner_x1 = x1 + int(0.25 * (x2 - x1))
    inner_x2 = x2 - int(0.25 * (x2 - x1))
    inner_y1 = y1 + int(0.25 * (y2 - y1))
    inner_y2 = y2 - int(0.25 * (y2 - y1))
    if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
        inner_x1, inner_x2, inner_y1, inner_y2 = x1, x2, y1, y2

    depth_m = depth_image[inner_y1:inner_y2, inner_x1:inner_x2].astype(np.float32)
    depth_m *= scale_to_m
    valid = depth_m[np.isfinite(depth_m)]
    valid = valid[(valid > depth_min_m) & (valid < depth_max_m)]
    if valid.size < min_valid_samples:
        return None

    z_m = float(np.median(valid))
    u_px = 0.5 * (x1 + x2)
    v_px = 0.5 * (y1 + y2)
    return ((u_px - cx) * z_m / fx, (v_px - cy) * z_m / fy, z_m)
