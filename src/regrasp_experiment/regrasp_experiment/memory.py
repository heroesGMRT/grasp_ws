"""KONTRIBUSI INTI paper — confidence memory (Pers. 10-12).

Hanya bagian inilah yang benar-benar novel. Selebihnya komponen standar.
"""
import numpy as np


class ConfidenceMemory:
    """Peta memori spasial M_t: down-weight region yang pernah gagal grasp,
    sehingga retry berikutnya dipaksa berbeda (Pers. 10-12)."""

    def __init__(self, shape, rho: float = 0.5):
        self.rho = rho
        self.M = np.ones(shape, dtype=np.float32)   # M_0(u) = 1

    def modulate(self, base_scores, cand_masks):
        """Pers. (12): S~(g_i) = S(g_i) * mean_{u in R(g_i)} M(u)."""
        mem_factor = np.array(
            [float(self.M[m].mean()) if m.any() else 0.0 for m in cand_masks],
            dtype=np.float32,
        )
        return base_scores * mem_factor

    def penalize(self, region_mask):
        """Pers. (10): setelah gagal, turunkan bobot region grasp."""
        self.M[region_mask] *= self.rho

    def exhausted(self, thresh: float = 1e-3) -> bool:
        """Degradasi anggun: bila hampir semua region tak dipercaya, hentikan loop."""
        return bool(self.M.max() < thresh)
