# regrasp_experiment — Langkah Implementasi (ROS2 + RealSense D455)

Paket ROS2 untuk eksperimen *uncertainty-guided closed-loop re-grasp*. Kelima metode
(BL1, BL2, BL3, OURS + BL4 eksternal) berjalan pada **satu codebase**, beda `config`.

## 0. Peta file
| File | Isi | Novel? |
|---|---|---|
| `config.py` | Definisi 5 metode (flag) | — |
| `memory.py` | **Confidence memory (Pers. 10-12)** | ✅ inti |
| `perception.py` | Depth reliability + adaptive fusion (Pers. 1-4) | pendukung |
| `scoring.py` | Uncertainty-aware scoring (Pers. 7-8) | pendukung |
| `detector.py` | Bungkus grasp detector (GR-ConvNet/GG-CNN) + stub | ganti TODO |
| `robot.py` | Eksekusi arm via MoveIt2 + stub | ganti TODO |
| `pipeline.py` | `run_trial()` — loop 1 trial | — |
| `experiment_node.py` | Node ROS2: sinkron D455 → trial → log | — |
| `logger.py` | CSV per-attempt (sesuai SOP Bagian 8) | — |

---

## 1. Prasyarat (sekali)
```bash
# ROS2 (Humble/Jazzy) sudah terpasang. Lalu:
sudo apt install ros-$ROS_DISTRO-realsense2-camera \
                 ros-$ROS_DISTRO-cv-bridge \
                 ros-$ROS_DISTRO-image-transport
pip install numpy opencv-python
# Untuk detector nyata (GR-ConvNet): pip install torch torchvision
```
Colokkan **D455 ke port USB 3.0** (kabel/port biru). Cek: `realsense-viewer`.

## 2. Build workspace
```bash
cd "D:/Paper Q2 Robot/grasp_ws"     # di Linux: ~/grasp_ws
colcon build --packages-select regrasp_experiment
source install/setup.bash
```

## 3. Uji plumbing DULU (pakai stub, tanpa detector/robot nyata)
Tujuan: pastikan D455 → pipeline → CSV mengalir sebelum integrasi berat.

**Terminal 1 — kamera D455:**
```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true rgb_camera.color_profile:=848x480x30 \
  depth_module.depth_profile:=848x480x30 enable_sync:=true
```
Cek topik hidup: `ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw`

**Terminal 2 — node eksperimen (interaktif):**
```bash
source install/setup.bash
ros2 run regrasp_experiment experiment
```
> ⚠️ Jalankan node lewat `ros2 run` di terminal sendiri (BUKAN via `ros2 launch`),
> karena mode operator butuh input keyboard. Launch file disediakan hanya untuk
> mode non-interaktif nanti.

Ikuti prompt (trial_id, metode, kondisi, object_id). Stub akan mengembalikan outcome
acak → satu baris per attempt tertulis di `experiment_log.csv`. Jika CSV terisi, plumbing OK.

## 4. Integrasi bertahap (ganti stub → nyata)
Kerjakan berurutan; uji tiap langkah sebelum lanjut:

1. **Kalibrasi hand-eye** → simpan `config/hand_eye.npy` (4×4). Muat di `experiment_node.py`
   (ganti `T_base_cam = np.eye(4)`). Ukur & catat residual (mm).
2. **Detector nyata** (`detector.py`): muat GR-ConvNet/GG-CNN di `__init__`, isi `infer()`
   (hasilkan `q_img/angle/width` → `GraspCandidate` + `mask` + `s_geo/s_conf`).
   Ini otomatis mengaktifkan BL1, BL2, BL3, OURS.
3. **Backbone fitur** (`experiment_node.on_frame`): ganti `F_rgb`/`c_rgb` placeholder
   dengan fitur & objectness dari backbone detektor.
4. **Robot nyata** (`robot.py`): isi `execute()` dengan MoveIt2 (plan→grasp→lift) dan
   baca outcome dari **sensor gripper/vacuum** (bukan mata). Isi `go_home()`.
5. **Orientasi 6D** (`detector.backproject`): isi `R` dari PCA point cloud dalam mask.

## 5. Menjalankan metode
- **BL1/BL2/BL3/OURS:** cukup pilih kode metode saat prompt. Tak ada perubahan kode.
- **BL4 (SOTA):** clone repo publik (mis. Contact-GraspNet), jalankan **pada scene yang sama**,
  catat hasil + *resource footprint* (GPU, ukuran model) ke `experiment_log.csv` secara manual.

## 6. Parameter (laporkan di paper — Section 4.2)
Semua di `config/params.yaml`: `fusion_lambda, alpha/beta/gamma, tau, rho, K`.
Untuk ablation, ubah `config.py` (mis. `fusion_mode`, `use_uncertainty`) atau `rho/K/tau`.

## 7. Output
`experiment_log.csv` — satu baris per attempt, kolom sesuai SOP Bagian 8
(termasuk `moved` untuk **retry-diversity** dan `cause` untuk **failure analysis**).
Analisis lanjutan (mean±std, uji signifikansi) di SOP Bagian 9.

---
### Catatan realistas
- Kode ini **scaffold** — bagian bertanda `TODO`/`STUB` harus Anda isi. Yang **novel & sudah jadi**
  ada di `memory.py` (≈30 baris) — itulah kontribusi utama paper.
- Path Windows di atas untuk referensi; **ROS2 sebaiknya di Ubuntu** (native atau WSL2/Docker).
  Di WSL2, akses USB D455 perlu `usbipd-win`.
