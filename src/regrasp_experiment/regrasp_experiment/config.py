"""Konfigurasi 5 metode. Semua metode = pipeline yang sama, beda flag.

Ini yang mewujudkan prinsip 'satu codebase, banyak metode' untuk perbandingan adil.
"""
from dataclasses import dataclass


@dataclass
class MethodConfig:
    name: str
    fusion_mode: str        # "adaptive" | "fixed" | "rgb_only"
    use_uncertainty: bool   # aktifkan suku uncertainty pada scoring (Pers. 7)
    recovery: str           # "none" | "naive" | "confidence_memory"
    K: int = 4              # maksimum attempt per scene
    rho: float = 0.5        # decay confidence-memory (Pers. 10)
    tau: float = 0.30       # ambang grasp valid
    fixed_w_depth: float = 0.5  # bobot depth utk fusion_mode == "fixed"


# BL4 (SOTA) tidak di sini: dijalankan dari repo eksternal, hasilnya di-log manual.
METHODS = {
    "BL1":  MethodConfig("BL1_oneshot",  fusion_mode="adaptive", use_uncertainty=True,  recovery="none"),
    "BL2":  MethodConfig("BL2_naive",    fusion_mode="adaptive", use_uncertainty=True,  recovery="naive"),
    "BL3":  MethodConfig("BL3_rgbonly",  fusion_mode="rgb_only", use_uncertainty=False, recovery="none"),
    "OURS": MethodConfig("OURS",         fusion_mode="adaptive", use_uncertainty=True,  recovery="confidence_memory"),
}


# Kondisi workspace (dicatat di log, tidak mengubah kode)
CONDITIONS = ["C1_structured", "C2_semi", "C3_cluttered"]
