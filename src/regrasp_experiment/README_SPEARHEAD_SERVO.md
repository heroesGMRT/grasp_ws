# Spearhead Visual Servo

Node ROS 2 ini dipakai untuk tahap pengambilan spearhead dengan kamera RealSense, perintah kecepatan `/cmd_vel`, dan heading hold dari `/teensy_odom_raw`.

Fokus node ini adalah membuat robot bergerak sampai bounding box target sama dengan pose teach, lalu menjalankan command gripper. Outcome `SUCCESS` / `FAIL` dibaca dari `/proximity/obstacle` setelah objek diangkat dan robot retreat.

## TL;DR Commands

Jalankan build sekali setiap kali source code berubah:

```bash
cd ~/Workspace/eksperimen_R2/grasp_ws
colcon build --packages-select regrasp_experiment
source install/setup.bash
```

Terminal 1: nyalakan RealSense.

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  rgb_camera.color_profile:=848x480x30 \
  depth_module.depth_profile:=848x480x30 \
  enable_sync:=true
```

Terminal 2: lakukan teach. Posisikan robot tepat di depan rack, fokuskan jendela `Spearhead Teach`, lalu tekan `s` untuk menyimpan bbox goal.

```bash
source ~/Workspace/eksperimen_R2/grasp_ws/install/setup.bash
ros2 launch regrasp_experiment spearhead_servo.launch.py \
  teach:=true \
  servo_mode:=bbox_ibvs
```

Terminal 2: jalankan eksperimen. Command pertama memulai proses pengambilan; command kedua menghentikan robot bila diperlukan.

```bash
source ~/Workspace/eksperimen_R2/grasp_ws/install/setup.bash
ros2 launch regrasp_experiment spearhead_servo.launch.py \
  teach:=false \
  servo_mode:=bbox_ibvs

ros2 service call /spearhead_servo/start std_srvs/srv/Trigger {}
ros2 service call /spearhead_servo/abort std_srvs/srv/Trigger {}
```

Untuk menjalankan dashboard operator lokal, tambahkan `dashboard:=true` saat
launch (butuh desktop display dan paket `python3-tk`):

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py \
  teach:=false servo_mode:=bbox_ibvs dashboard:=true
```

Dashboard menyediakan tombol **Start run**, **Continue step**, **Abort / stop**,
**Save current position as start**, dan **Return base to start**. Reset memakai
pose `/teensy_odom_raw` yang disimpan saat start pertama atau melalui tombol
simpan. Ia membatalkan urutan aktif, lalu mengembalikan base ke pose tersebut
dengan kecepatan terbatas. Uji dahulu dengan roda terangkat: wheel odometry
bukan pengganti sensor keselamatan dan dapat drift di lantai.

Untuk `step_mode`, tambahkan `step_mode:=true`; setelah robot mencapai posisi awal, kirim command berikut untuk mulai servo bbox:

```bash
ros2 service call /spearhead_servo/next_step std_srvs/srv/Trigger {}
```

## Cara Kerja Singkat

Alur utamanya:

```text
RealSense RGB
-> YOLO best.pt mendeteksi bounding box spearhead
-> teach menyimpan pusat bbox [u, v] dan ukuran [w, h] sebagai goal
-> pusat bbox mengoreksi strafe, depth median di dalam bbox mengoreksi maju/mundur ke 0.21 m
-> yaw dijaga pada heading saat start; target <0.20 m memaksa robot mundur
-> saat bbox, depth, dan yaw masuk toleransi, robot berhenti melakukan servo
-> gripper close
-> arm up
-> robot retreat
```

Node punya dua mode visual servo:

- `bbox_ibvs`: mode default. Bbox YOLO memilih dan memusatkan spearhead di antara tongs; depth RealSense di dalam bbox mengatur standoff 0.21 m.
- `depth_3d`: mode lama. Menggunakan posisi 3D RGB-D dan `final_approach_m`.

Untuk pengambilan di lapangan dengan rack spearhead, gunakan `bbox_ibvs` dan `detector_mode: "yolo_bbox"`.

## Dependency

### ROS 2

Install ROS 2 dan package berikut di Ubuntu/WSL2:

```bash
sudo apt install ros-$ROS_DISTRO-realsense2-camera \
                 ros-$ROS_DISTRO-cv-bridge \
                 ros-$ROS_DISTRO-image-transport
```

Package ROS yang dipakai node:

- `rclpy`
- `sensor_msgs`
- `geometry_msgs`
- `nav_msgs`
- `std_msgs`
- `std_srvs`
- `cv_bridge`
- `realsense2_camera`

### Python

Minimal:

```bash
pip install numpy opencv-python pyyaml
```

Untuk mode YOLO `best.pt`:

```bash
pip install ultralytics
```

Catatan: `best.pt` harus berasal dari model YOLO yang kompatibel dengan Ultralytics.

## Build

Dari root workspace ROS:

```bash
cd grasp_ws
colcon build --packages-select regrasp_experiment
source install/setup.bash
```

Artefak hasil `colcon build` di `grasp_ws/build/`, `grasp_ws/install/`, dan
`grasp_ws/log/` diabaikan oleh Git. Folder tersebut dibuat ulang secara lokal
saat build dan tidak perlu dimasukkan ke commit.

## Jalankan RealSense

Jalankan kamera dengan depth aligned ke color:

```bash
ros2 launch realsense2_camera rs_launch.py \
  align_depth.enable:=true \
  rgb_camera.color_profile:=848x480x30 \
  depth_module.depth_profile:=848x480x30 \
  enable_sync:=true
```

Pastikan topik ini hidup:

```bash
ros2 topic hz /camera/camera/aligned_depth_to_color/image_raw
ros2 topic hz /camera/camera/color/image_raw
```

## Konfigurasi YOLO best.pt

Untuk tahap pengembangan sekarang, taruh file model di source package:

```text
grasp_ws/src/regrasp_experiment/config/models/best.pt
```

Edit:

```text
grasp_ws/src/regrasp_experiment/config/spearhead_servo.yaml
```

Contoh:

```yaml
spearhead_servo:
  ros__parameters:
    detector_mode: "yolo_bbox"
    yolo_weights: "/home/heroes/grasp_ws/src/regrasp_experiment/config/models/best.pt"
    yolo_conf: 0.35
    yolo_iou: 0.45
    yolo_imgsz: 640
    yolo_class_id: -1
```

Jika workspace berada di path lain, sesuaikan `yolo_weights` ke path absolut file `best.pt` di komputer robot.

Contoh saat masih di Windows:

```text
D:\Coding\eksperimen_R2\grasp_ws\src\regrasp_experiment\config\models\best.pt
```

Jika model hanya punya satu class, biarkan:

```yaml
yolo_class_id: -1
```

Jika model punya banyak class, isi dengan class ID target spearhead.

## BBox Goal

Dalam `bbox_ibvs`, goal teach adalah posisi bbox di gambar saat robot sudah berada di posisi pengambilan yang tepat di depan rack. Node menyimpan empat nilai:

```text
u, v = titik tengah bbox dalam pixel
w, h = lebar dan tinggi bbox dalam pixel
```

Saat eksperimen berjalan, error `u` mengoreksi strafe. Median depth valid dari separuh bagian dalam bbox mengoreksi maju/mundur ke `bbox_target_depth_m: 0.21`. Jika depth spearhead berada di bawah `bbox_overshoot_depth_m: 0.20`, node hanya memerintahkan backup dan tidak akan menutup gripper.

Node berhenti dan mengirim `grip close -> arm up -> retreat` ketika pusat bbox, depth, dan yaw masuk toleransi. Node mengambil yaw saat `start` dari `/teensy_odom_raw` dan mengoreksi drift melalui `Twist.angular.z`; yaw yang belum selaras memblokir gripper. Jika error vertikal bbox masih besar setelah error lain benar, node masuk `FAULT` karena robot ini tidak mengoreksi gerak vertikal.

File setpoint hasil teach menyimpan goal bbox. Teach ulang setiap kali dudukan kamera, gripper, resolusi kamera, ROI, atau posisi rack berubah.

## Teach Mode

Jalankan node dalam mode teach:

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py teach:=true
```

Di mode ini robot tidak bergerak otomatis. Sebuah jendela kamera `Spearhead Teach` akan muncul dan menampilkan target yang terdeteksi secara real-time:

- Kotak hijau: bounding box YOLO saat ini.
- Tanda silang kuning: pusat bbox saat ini.
- Kotak biru: bbox goal hasil teach, setelah goal tersimpan.
- `C` dan `size`: pusat serta ukuran bbox saat ini.
- `dU`, `dV`, dan `scale`: error bbox terhadap goal.

Saat jendela kamera fokus, tekan `s` untuk menyimpan bbox goal langsung ke `setpoint_file`; terminal atau service call tidak diperlukan. UI menampilkan status simpan dan akan menolak penyimpanan sampai minimal 15 frame bbox telah diterima. Tekan `q` untuk menyembunyikan preview tanpa menghentikan node. Tombol simpan dapat diubah lewat `teach_save_key`.

Untuk menjalankan di komputer tanpa desktop/monitor, nonaktifkan UI saat launch:

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py \
  teach:=true \
  teach_show_ui:=false
```

Gerakkan robot secara manual sampai gripper berada pada posisi pengambilan yang tepat di depan rack. Tahan robot dan rack tetap stabil sampai UI menerima minimal 15 frame bbox, lalu tekan `s` pada jendela kamera.

Goal bbox akan disimpan ke file:

```yaml
setpoint_file: "/home/heroes/spearhead_setpoint.yaml"
```

Path ini bisa diubah di YAML atau saat launch:

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py \
  teach:=true \
  setpoint_file:=/home/heroes/spearhead_setpoint.yaml
```

## Run Mode

## Safety and Calibration Gate

Before enabling motion, confirm the emergency-stop works, depth is aligned to
color, and `/teensy_odom_raw` publishes a valid `nav_msgs/Odometry` message.
Hand-eye calibration must be repeated whenever the camera or gripper mount
moves and at the start of each experiment day; record the calibration residual
and keep it below the experiment's accepted threshold (for example, 5 mm).

Perform the first `/cmd_vel` test with the wheels raised. Confirm the signs of
`linear.x`, `linear.y`, and `angular.z`, then call `abort` and confirm the base
stops. Run low-speed floor validation before enabling an autonomous grasp.
Grasp success/failure must come from the gripper or vacuum sensor, not visual
inspection.

## Grip Outcome and Attempt Log

After the gripper-close command (`grip_close_code: 41`), the node waits for a
fresh `std_msgs/Bool` reading from `/proximity/obstacle`. `true` means the
grip succeeded; `false` means it failed. No new sensor reading before
After closing, the node completes its configured lift and retreat, then waits
for `grip_outcome_timeout_s` (default: four seconds) before reading the most
recent fresh sensor result. If no message is received during that window, the
attempt is recorded as failed with cause `proximity_timeout`.

Each close command creates one row in `attempt_log_file` (default
`/home/heroes/spearhead_attempt_log.csv`) with timestamp, attempt number,
outcome, cause, raw sensor value, and `attempt_duration_s`. This duration is
measured from the gripper-close command until the post-lift, post-retreat
proximity result (or timeout). The result is also published as a Bool
on `/spearhead_servo/grip_success`.

Jalankan node:

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py teach:=false
```

Mulai pengambilan:

```bash
ros2 service call /spearhead_servo/start std_srvs/srv/Trigger {}
```

Alur `bbox_ibvs` saat run:

```text
pre-grab -> ukur 5 bbox -> koreksi strafe/maju -> settle -> ukur ulang
-> goal masuk toleransi -> grip close -> arm up -> retreat -> DONE
```

Kecepatan koreksi dibatasi oleh `max_forward_mps`, `max_strafe_mps`, `max_yaw_radps`, dan `max_cycles`. Jika odometry tidak tersedia atau stale, node menolak start atau berhenti untuk menjaga yaw safety.

Abort jika perlu:

```bash
ros2 service call /spearhead_servo/abort std_srvs/srv/Trigger {}
```

## Step Mode

`step_mode` dipakai saat eksperimen membutuhkan operator untuk memverifikasi pose awal sebelum visual servo dimulai. Saat node menerima command `start`, robot menjalankan urutan `pre_grab_codes`, lalu menjalankan `initial_move_x` dan `initial_move_y` (bila nilainya minimal `min_move_m`). Setelah itu robot berhenti di `WAIT_STEP`; tidak ada koreksi visual atau gerak pengambilan yang berjalan otomatis.

Posisi awal bersifat relatif terhadap posisi robot saat `start` dipanggil. Perintah dikirim sebagai kecepatan terbatas pada `/cmd_vel`; verifikasi arah dan jarak dengan roda terangkat sebelum digunakan di lantai.

Atur perpindahan awal di YAML, misalnya:

```yaml
step_mode: true
initial_move_x: 0.30
initial_move_y: 0.00
```

Lalu jalankan:

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py step_mode:=true
ros2 service call /spearhead_servo/start std_srvs/srv/Trigger {}
```

Setelah robot mencapai pose awal dan Anda sudah memeriksa kondisi eksperimen, lanjutkan visual servo dari terminal:

```bash
ros2 service call /spearhead_servo/next_step std_srvs/srv/Trigger {}
```

Service `next_step` hanya diterima ketika state node adalah `WAIT_STEP`. Gunakan `abort` untuk menghentikan proses pada tahap apa pun.

Untuk langsung start otomatis:

```bash
ros2 launch regrasp_experiment spearhead_servo.launch.py \
  teach:=false \
  auto_start:=true
```

## Topic I/O

Subscribe:

- `/camera/camera/aligned_depth_to_color/image_raw`
- `/camera/camera/color/image_raw`
- `/camera/camera/color/camera_info`
- `/camera/camera/imu`
- `/teensy_odom_raw` (`nav_msgs/Odometry`)
- `/proximity/obstacle` (`std_msgs/Bool`)

Publish:

- `/cmd_vel` (`geometry_msgs/Twist`)
- `/fsm_command` (`std_msgs/Int32`)
- `/spearhead_servo/grip_success` (`std_msgs/Bool`)

Service:

- `/spearhead_servo/record_setpoint`
- `/spearhead_servo/start`
- `/spearhead_servo/next_step`
- `/spearhead_servo/abort`
- `/spearhead_servo/set_start_pose`
- `/spearhead_servo/reset_to_start`

## FSM Command Default

Default command:

```yaml
pre_grab_codes: [43, 40]  # arm down, grip open
grab_codes: [41, 42]      # grip close, arm up
```

Kode:

```text
40 = grip open
41 = grip close
42 = arm spear up
43 = arm spear down
```

Cek urutan ini dengan mekanisme robot sebelum run pertama.

## Parameter Penting

```yaml
servo_mode: "bbox_ibvs"
bbox_tol_u_px: 12.0
bbox_tol_v_px: 20.0
bbox_tol_scale_ratio: 0.08
bbox_lateral_gain_m_per_px: 0.001
bbox_forward_gain_m: 0.30
max_step_m: 0.15
min_move_m: 0.10
max_cycles: 12
retreat_m: 0.15
```

`bbox_tol_u_px`, `bbox_tol_v_px`, dan `bbox_tol_scale_ratio` adalah batas berhenti. Naikkan toleransi bila bbox bergetar agar robot berhenti lebih cepat; turunkan hanya setelah kontrol geraknya stabil. `min_move_m` adalah batas minimum firmware. Jika koreksi lebih kecil dari nilai ini, node membuat back-off pair agar hasil netto tetap kecil tetapi setiap command tetap memenuhi batas firmware.

## Catatan SUCCESS / FAIL

Saat ini node hanya menjalankan proses pengambilan sampai command gripper dan retreat. Deteksi `SUCCESS` / `FAIL` sebaiknya ditambahkan dari sensor sendiri, misalnya:

- pressure sensor untuk vacuum
- limit/gap sensor gripper
- force/current sensor

Tempat integrasi paling alami adalah setelah command `grip close` dan `arm up`, sebelum node menyatakan `DONE`.
