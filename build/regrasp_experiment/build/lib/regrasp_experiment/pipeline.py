"""Loop satu trial (run_trial). Semua metode lewat sini; beda hanya cfg."""
import time
import numpy as np

from .perception import depth_reliability, fuse
from .scoring import score_candidates, argmax_valid
from .detector import backproject
from .memory import ConfidenceMemory


def region_changed(cand, prev_regions):
    """Retry pindah region? (metrik retry-diversity)."""
    for (u, v) in prev_regions:
        if abs(cand.u - u) < 15 and abs(cand.v - v) < 15:
            return False
    return True


def run_trial(cfg, frame, detector, robot, K_intr):
    """Jalankan satu scene dengan satu metode.

    cfg      : MethodConfig
    frame    : dict {"rgb": HxWx3, "F_rgb": HxWxC, "depth_m": HxW, "c_rgb": HxW}
               (Untuk uji awal, F_rgb boleh = rgb float dan c_rgb dari objectness.)
    return   : list log per-attempt
    """
    logs = []
    prev_regions = []
    mem = None

    for attempt in range(cfg.K):
        t0 = time.time()
        rgb, F_rgb, depth_m, c_rgb = frame["rgb"], frame["F_rgb"], frame["depth_m"], frame["c_rgb"]

        c_d = depth_reliability(depth_m)                                  # Pers. (1)
        F_fused = fuse(F_rgb, _as_feat(depth_m), c_rgb, c_d,
                       cfg.fusion_mode, cfg.fixed_w_depth)                # Pers. (2-4)

        if mem is None:
            mem = ConfidenceMemory(c_rgb.shape, rho=cfg.rho)

        cands = detector.infer(F_fused, depth_m, c_rgb, c_d)             # deteksi
        scores = score_candidates(cands, c_rgb, c_d, cfg.use_uncertainty)  # Pers. (7)

        if cfg.recovery == "confidence_memory":                          # OURS: Pers. (12)
            scores = mem.modulate(scores, [c.mask for c in cands])

        idx = argmax_valid(scores, cfg.tau)
        if idx is None or (mem.exhausted() and cfg.recovery == "confidence_memory"):
            logs.append(_row(attempt, None, False, "no_valid_grasp", t0))
            break

        best = cands[idx]
        if backproject(best, depth_m, K_intr) is None:
            logs.append(_row(attempt, best, False, "invalid_depth", t0))
            _after_fail(cfg, mem, best); prev_regions.append((best.u, best.v))
            if cfg.recovery == "none":
                break
            continue

        moved = region_changed(best, prev_regions)
        outcome = robot.execute(best)                                    # eksekusi (sensor)
        robot.go_home()
        logs.append(_row(attempt, best, outcome == "success", "", t0, moved))
        prev_regions.append((best.u, best.v))

        if outcome == "success":
            break
        # ---- strategi setelah gagal ----
        if cfg.recovery == "none":
            break                                                        # BL1/BL3
        elif cfg.recovery == "naive":
            continue                                                     # BL2: ulang apa adanya
        elif cfg.recovery == "confidence_memory":
            mem.penalize(best.region)                                    # OURS: Pers. (10)

    return logs


def _after_fail(cfg, mem, cand):
    if cfg.recovery == "confidence_memory" and cand.mask is not None:
        mem.penalize(cand.region)


def _as_feat(depth_m):
    """Depth -> peta 'fitur' minimal (H,W,1) untuk uji awal. Ganti dgn cabang depth nyata."""
    return depth_m[..., None].astype(np.float32)


def _row(attempt, cand, success, cause, t0, moved=None):
    return dict(
        attempt=attempt,
        region=(None if cand is None else (cand.u, cand.v)),
        moved=moved,
        outcome=("success" if success else "fail"),
        cause=cause,
        latency_ms=round((time.time() - t0) * 1000, 1),
    )
