"""Depth reliability + adaptive RGB-D fusion (Pers. 1-4). Komponen pendukung."""
import numpy as np
import cv2


def depth_reliability(depth_m, lam: float = 0.02, ksize: int = 5):
    """Pers. (1): c_d(u) = v(u) * exp(-var_local / (2*lam^2)).

    depth_m : depth dalam METER (0 = invalid). D455 keluar uint16 mm -> bagi 1000.
    """
    valid = (depth_m > 0).astype(np.float32)                       # v(u)
    mean = cv2.blur(depth_m, (ksize, ksize))
    mean_sq = cv2.blur(depth_m * depth_m, (ksize, ksize))
    var = np.clip(mean_sq - mean * mean, 0, None)                  # varians lokal
    c_d = valid * np.exp(-var / (2.0 * lam * lam))
    return c_d.astype(np.float32)


def fuse(F_rgb, F_depth, c_rgb, c_d, mode: str, fixed_w_depth: float = 0.5, eps: float = 1e-6):
    """Pers. (2-4): gabung fitur RGB & depth sesuai mode.

    F_rgb, F_depth : (H, W, C) peta fitur. c_rgb, c_d : (H, W) confidence [0,1].
    """
    if mode == "rgb_only":
        return F_rgb
    if mode == "fixed":
        w_d = np.full_like(c_d, fixed_w_depth)
    elif mode == "adaptive":
        w_d = c_d / (c_d + c_rgb + eps)                            # Pers. (2)
    else:
        raise ValueError(f"fusion_mode tak dikenal: {mode}")
    w_d = w_d[..., None]                                           # broadcast ke channel
    return (1.0 - w_d) * F_rgb + w_d * F_depth                     # Pers. (4)
