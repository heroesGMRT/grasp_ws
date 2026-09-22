"""Node ROS2 utama: sinkron RGB-D D455 -> jalankan trial -> log.

Alur lab: operator menata scene, tekan Enter, node menjalankan 1 trial 1 metode.
"""
import threading
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import message_filters

from .config import METHODS, CONDITIONS
from .detector import GraspDetector
from .robot import RobotExecutor
from .pipeline import run_trial
from .logger import TrialLogger

# Topik default realsense2_camera (D455). Sesuaikan bila namespace berbeda.
TOPIC_COLOR = "/camera/camera/color/image_raw"
TOPIC_DEPTH = "/camera/camera/aligned_depth_to_color/image_raw"
TOPIC_INFO  = "/camera/camera/color/camera_info"


class ExperimentNode(Node):
    def __init__(self):
        super().__init__("regrasp_experiment")
        self.bridge = CvBridge()
        self.latest = None          # frame RGB-D terbaru
        self.K = None               # intrinsik 3x3
        self.lock = threading.Lock()

        color_sub = message_filters.Subscriber(self, Image, TOPIC_COLOR)
        depth_sub = message_filters.Subscriber(self, Image, TOPIC_DEPTH)
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [color_sub, depth_sub], queue_size=10, slop=0.05)
        self.sync.registerCallback(self.on_frame)
        self.create_subscription(CameraInfo, TOPIC_INFO, self.on_info, 10)

        # Hand-eye calibration: MUAT dari file (Pers. 9). Placeholder = identitas.
        T_base_cam = np.eye(4)   # TODO: np.load("config/hand_eye.npy")
        self.detector = GraspDetector(weights_path=None)   # TODO: isi weights
        self.robot = RobotExecutor(self, T_base_cam)
        self.logger = TrialLogger("experiment_log.csv")
        self.get_logger().info("ExperimentNode siap. Menunggu frame D455...")

    def on_info(self, msg: CameraInfo):
        self.K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def on_frame(self, color_msg: Image, depth_msg: Image):
        rgb = self.bridge.imgmsg_to_cv2(color_msg, "rgb8")
        depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")  # uint16 mm
        depth_m = depth_raw.astype(np.float32) / 1000.0                  # -> meter
        # F_rgb & c_rgb minimal utk uji awal; ganti dgn backbone nyata.
        F_rgb = rgb.astype(np.float32) / 255.0
        c_rgb = F_rgb.mean(axis=2)                                       # placeholder objectness
        with self.lock:
            self.latest = dict(rgb=rgb, F_rgb=F_rgb, depth_m=depth_m, c_rgb=c_rgb)

    def get_frame(self):
        with self.lock:
            if self.latest is None or self.K is None:
                return None
            return dict(self.latest)

    def run_one(self, trial_id, run, method_key, condition, object_id):
        frame = self.get_frame()
        if frame is None:
            self.get_logger().warn("Belum ada frame/camera_info.")
            return
        cfg = METHODS[method_key]
        rows = run_trial(cfg, frame, self.detector, self.robot, self.K)
        ok = self.logger.log_trial(trial_id, run, cfg.name, condition, object_id, rows)
        self.get_logger().info(
            f"[{trial_id}] {cfg.name} {condition} {object_id}: "
            f"{'SUKSES' if ok else 'GAGAL'} dalam {len(rows)} attempt")


def operator_loop(node: ExperimentNode):
    """Loop interaktif di thread utama."""
    print("\n=== Mode operator. Ketik 'q' untuk keluar. ===")
    while rclpy.ok():
        try:
            trial_id  = input("\ntrial_id (mis. T-0001): ").strip()
            if trial_id.lower() == "q":
                break
            method    = input(f"metode {list(METHODS)}: ").strip() or "OURS"
            condition = input(f"kondisi {CONDITIONS}: ").strip() or CONDITIONS[0]
            object_id = input("object_id (mis. OBJ-07): ").strip()
            run       = input("run (mis. run-1): ").strip() or "run-1"
            input(">> Tata scene, lalu tekan Enter untuk eksekusi...")
            node.run_one(trial_id, run, method, condition, object_id)
        except (EOFError, KeyboardInterrupt):
            break


def main():
    rclpy.init()
    node = ExperimentNode()
    # Spin ROS di background; operator prompt di thread utama.
    spin = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin.start()
    try:
        operator_loop(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
