"""Antarmuka grasp detector + candidate. GANTI bagian TODO dengan detector nyata.

Rekomendasi: bungkus GR-ConvNet atau GG-CNN (kode PyTorch publik).
Detector menerima peta fitur/gambar -> menghasilkan kandidat grasp 2D+depth.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class GraspCandidate:
    u: int; v: int            # piksel titik grasp
    angle: float              # sudut grasp (rad)
    width: float              # lebar bukaan (px atau m)
    s_geo: float = 0.0        # kualitas geometri [0,1]
    s_conf: float = 0.0       # confidence detektor [0,1]
    mask: np.ndarray = field(default=None)   # region pendukung R(g), bool HxW
    pose6d: np.ndarray = field(default=None) # 4x4 di frame kamera (diisi saat back-project)

    @property
    def region(self):
        return self.mask


class GraspDetector:
    """Bungkus model deteksi grasp. Implementasikan .infer()."""

    def __init__(self, weights_path: str = None, device: str = "cuda"):
        self.device = device
        # TODO: muat model, mis. GR-ConvNet:
        #   self.net = GenerativeResnet(); self.net.load_state_dict(torch.load(weights_path))
        self.net = None

    def infer(self, F_input, depth_m, c_rgb, c_d, top_k: int = 20):
        """Kembalikan list[GraspCandidate].

        F_input : peta fitur/citra hasil fusion. depth_m : depth meter.
        """
        if self.net is None:
            # ---- STUB: untuk uji plumbing sebelum detector nyata siap ----
            return self._stub(F_input, c_rgb, top_k)
        # TODO nyata:
        #   q_img, angle_img, width_img = self.net.predict(F_input)
        #   peaks = detect_grasps(q_img, angle_img, width_img, no_grasps=top_k)
        #   -> bangun GraspCandidate + mask (disk radius width) + s_geo/s_conf
        raise NotImplementedError

    def _stub(self, F_input, c_rgb, top_k):
        H, W = c_rgb.shape
        rng = np.random.default_rng(0)
        cands = []
        for _ in range(top_k):
            u, v = int(rng.integers(0, W)), int(rng.integers(0, H))
            m = np.zeros((H, W), dtype=bool)
            r = 12
            m[max(0, v-r):v+r, max(0, u-r):u+r] = True
            cands.append(GraspCandidate(
                u=u, v=v, angle=float(rng.uniform(0, np.pi)), width=30.0,
                s_geo=float(rng.uniform(0, 1)), s_conf=float(c_rgb[v, u]), mask=m))
        return cands


def backproject(cand, depth_m, K):
    """Pers. (6): titik grasp -> 3D kamera. K : intrinsik 3x3 dari camera_info."""
    z = float(depth_m[cand.v, cand.u])
    if z <= 0:
        return None
    x = (cand.u - K[0, 2]) * z / K[0, 0]
    y = (cand.v - K[1, 2]) * z / K[1, 1]
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    # TODO: isi orientasi R dari PCA point cloud di dalam mask (Section 3.4)
    cand.pose6d = T
    return T
