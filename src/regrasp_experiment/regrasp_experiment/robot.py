"""Antarmuka eksekusi robot. GANTI stub dengan driver arm Anda (MoveIt2).

Outcome grasp WAJIB dari sensor (gaya gripper / tekanan vacuum), bukan mata.
"""
import numpy as np


class RobotExecutor:
    def __init__(self, node, T_base_cam: np.ndarray):
        """T_base_cam : 4x4 hasil hand-eye calibration (Pers. 9)."""
        self.node = node
        self.T_base_cam = T_base_cam
        # TODO: init MoveIt2 / driver arm + antarmuka gripper.

    def execute(self, cand) -> str:
        """Pindah ke pose grasp, tutup gripper, angkat, cek sukses. -> "success"|"fail"."""
        pose_base = self.T_base_cam @ cand.pose6d       # Pers. (9): kamera -> base
        # TODO nyata:
        #   1. plan & move ke pre-grasp lalu grasp pose (MoveIt2)
        #   2. tutup gripper / aktifkan vacuum
        #   3. angkat objek
        #   4. return "success" jika sensor gaya/vacuum mendeteksi objek, else "fail"
        return self._stub_outcome(pose_base)

    def go_home(self):
        pass  # TODO: kembali ke pose awal antar-attempt

    def _stub_outcome(self, pose_base) -> str:
        # STUB acak untuk uji plumbing; hapus saat driver nyata siap.
        return "success" if np.random.default_rng().random() > 0.5 else "fail"
