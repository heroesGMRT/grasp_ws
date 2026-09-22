"""Uncertainty-aware grasp scoring (Pers. 7-8). Komponen pendukung."""
import numpy as np


def uncertainty(cand_mask, c_rgb, c_d):
    """Pers. (8): U(g) = 1 - mean_{u in R(g)} max(c_rgb, c_d)."""
    conf = np.maximum(c_rgb, c_d)
    return 1.0 - float(conf[cand_mask].mean()) if cand_mask.any() else 1.0


def score_candidates(cands, c_rgb, c_d, use_uncertainty: bool,
                     alpha=1.0, beta=1.0, gamma=1.0):
    """Pers. (7): S(g) = a*S_geo + b*S_conf - c*U(g).

    cands : list objek grasp; tiap grasp punya .s_geo, .s_conf, .mask (bool HxW).
    """
    scores = np.zeros(len(cands), dtype=np.float32)
    for i, g in enumerate(cands):
        u = uncertainty(g.mask, c_rgb, c_d) if use_uncertainty else 0.0
        scores[i] = alpha * g.s_geo + beta * g.s_conf - gamma * u
    return scores


def argmax_valid(scores, tau: float):
    """Kembalikan index skor tertinggi yang >= tau; None jika tak ada yang valid."""
    if len(scores) == 0:
        return None
    i = int(np.argmax(scores))
    return i if scores[i] >= tau else None
