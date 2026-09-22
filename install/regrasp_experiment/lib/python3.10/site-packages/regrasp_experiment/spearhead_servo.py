#!/usr/bin/env python3
"""Spearhead visual servo node for ABU Robocon 2026 R2.

This node is intentionally integrated as a separate executable from the
uncertainty-guided re-grasp experiment. It controls a rear-mounted spearhead
gripper with look-then-move, position-based visual servoing using ``/cmd_vel``
and gripper FSM commands.
"""

import math
import os
import time
from enum import Enum, auto

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image, Imu
from std_msgs.msg import Bool, Int32
from std_srvs.srv import Trigger

from .motion_control import (
    depth_standoff_command,
    heading_command,
    return_to_start_command,
    wrap_angle,
)
from .grip_outcome import GraspAttemptLogger, proximity_outcome


class State(Enum):
    IDLE = auto()
    SEQUENCE = auto()
    WAIT_STEP = auto()
    MEASURE = auto()
    SETTLE = auto()
    FINAL_MEASURE = auto()
    DONE = auto()
    VERIFY_GRIP = auto()
    RESETTING = auto()
    FAULT = auto()


class SpearheadServo(Node):
    """Look-then-move visual servo controller for a spearhead gripper."""

    def __init__(self):
        super().__init__("spearhead_servo")

        p = self.declare_parameter

        p("depth_topic", "/camera/camera/aligned_depth_to_color/image_raw")
        p("color_topic", "/camera/camera/color/image_raw")
        p("info_topic", "/camera/camera/color/camera_info")
        p("imu_topic", "/camera/camera/imu")
        p("cmd_vel_topic", "/cmd_vel")
        p("odom_topic", "/teensy_odom_raw")
        p("proximity_topic", "/proximity/obstacle")
        p("grip_outcome_topic", "/spearhead_servo/grip_success")
        p("grip_outcome_timeout_s", 4.0)
        p("require_proximity_outcome", True)
        p("attempt_log_file", "spearhead_attempt_log.csv")
        p("grip_close_code", 41)
        p("ignore_imu", False)

        p("teach", False)
        p("auto_start", False)
        p("teach_show_ui", True)
        p("teach_window_name", "Spearhead Teach")
        p("teach_save_key", "s")
        p("step_mode", False)
        p("initial_move_x", 0.0)
        p("initial_move_y", 0.0)
        p("initial_move_wait_s", 1.5)
        p("servo_mode", "bbox_ibvs")

        p("setpoint_file", "/home/heroes/spearhead_setpoint.yaml")
        p("setpoint_x", 0.102)
        p("setpoint_y", -0.105)
        p("setpoint_z", 0.588)
        p("bbox_goal_u", 0.0)
        p("bbox_goal_v", 0.0)
        p("bbox_goal_w", 0.0)
        p("bbox_goal_h", 0.0)
        p("bbox_teach_samples", 15)
        p("bbox_tol_u_px", 12.0)
        p("bbox_tol_v_px", 20.0)
        p("bbox_tol_scale_ratio", 0.08)
        p("bbox_use_depth_z", True)
        p("bbox_target_depth_m", 0.21)
        p("bbox_overshoot_depth_m", 0.20)
        p("bbox_lateral_gain_m_per_px", 0.001)
        p("bbox_forward_gain_m", 0.30)

        p("roi", [0, 0, 0, 0])
        p("detector_mode", "depth_blob")
        p("yolo_weights", "")
        p("yolo_conf", 0.35)
        p("yolo_iou", 0.45)
        p("yolo_imgsz", 640)
        p("yolo_class_id", -1)
        p("yolo_fallback_depth_blob", True)
        p("depth_min_m", 0.10)
        p("depth_max_m", 1.20)
        p("min_width_m", 0.025)
        p("max_width_m", 0.30)
        p("min_blob_px", 40)
        p("use_hsv_reject", False)
        p("hsv_lo", [0, 0, 0])
        p("hsv_hi", [179, 255, 40])
        p("depth_scale_mm", True)

        p("frames_per_measurement", 5)
        p("measure_timeout_s", 2.0)
        p("max_measure_failures", 4)

        p("gyro_settle_rad_s", 0.03)
        p("settle_dwell_s", 0.4)
        p("settle_timeout_s", 3.0)

        p("gain", 0.75)
        p("max_step_m", 0.15)
        p("max_strafe_m", 0.05)
        p("min_move_m", 0.10)
        p("tol_lateral_m", 0.010)
        p("tol_depth_m", 0.015)
        p("max_cycles", 12)
        p("move_wait_s", 0.5)
        p("max_forward_mps", 0.10)
        p("max_strafe_mps", 0.08)
        p("max_yaw_radps", 0.40)
        p("heading_hold_gain", 1.5)
        p("heading_tolerance_rad", 0.035)
        p("odom_timeout_s", 0.75)
        p("reset_position_tolerance_m", 0.03)
        p("reset_yaw_tolerance_rad", 0.035)
        p("reset_linear_gain", 0.8)
        p("reset_max_linear_mps", 0.08)
        p("reset_timeout_s", 20.0)

        p("fwd_from_z_sign", -1.0)
        p("strafe_from_x_sign", -1.0)

        p("pre_grab_codes", [43, 40])
        p("pre_grab_waits", [2.0, 1.0])
        p("grab_codes", [41, 42])
        p("grab_waits", [1.5, 2.0])
        p("wiggle_repeats", 3)
        p("wiggle_back_m", 0.10)
        p("wiggle_fwd_m", 0.05)
        p("wiggle_wait_s", 1.0)

        p("final_approach_m", 0.40)
        p("blind_wait_s", 2.0)
        p("retreat_m", 0.15)
        p("retreat_wait_s", 1.5)

        self.gp = lambda name: self.get_parameter(name).value
        self._check_config()

        self.setpoint = np.array(
            [
                self.gp("setpoint_x"),
                self.gp("setpoint_y"),
                self.gp("setpoint_z"),
            ],
            dtype=np.float64,
        )
        self.bbox_goal = np.array(
            [
                self.gp("bbox_goal_u"),
                self.gp("bbox_goal_v"),
                self.gp("bbox_goal_w"),
                self.gp("bbox_goal_h"),
            ],
            dtype=np.float64,
        )
        self._load_setpoint()
        self.yolo_model = None
        self._load_yolo_model()

        self.bridge = CvBridge()
        self.K = None
        self.depth_img = None
        self.color_img = None
        self.color_seq = 0
        self.last_detection_ui = None
        self.teach_bbox_buf = []
        self.last_teach_bbox_seq = -1
        self.teach_status = ""
        self.teach_status_until = 0.0
        self.teach_ui_visible = self.gp("teach_show_ui")
        self.last_gyro_quiet_since = None
        self.gyro_quiet = self.gp("ignore_imu")

        self.state = State.IDLE
        self.measure_buf = []
        self.measure_start = 0.0
        self.measure_failures = 0
        self.cycles = 0
        self.move_queue = []
        self.move_sent_at = 0.0
        self.seq = []
        self.seq_next_at = 0.0
        self.seq_last_was_move = False
        self.seq_end_state = State.IDLE
        self.odom_yaw = None
        self.odom_x = None
        self.odom_y = None
        self.odom_received_at = 0.0
        self.held_yaw = None
        self.start_pose = None
        self.reset_started_at = 0.0
        self.active_motion = None
        self.proximity_value = None
        self.proximity_seq = 0
        self.proximity_seq_at_verify_start = 0
        self.grip_verify_started_at = 0.0
        self.grip_attempt_started_at = None
        self.trial_started_at = None
        self.grip_attempt = 0
        self.depth_seq = 0
        self.last_measure_seq = -1
        self.measure_depth_seq0 = 0
        self.yolo_cache_key = None
        self.yolo_cache = []
        self.attempt_logger = GraspAttemptLogger(self.gp("attempt_log_file"))

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(Image, self.gp("depth_topic"), self.on_depth, sensor_qos)
        self.create_subscription(Image, self.gp("color_topic"), self.on_color, sensor_qos)
        self.create_subscription(CameraInfo, self.gp("info_topic"), self.on_info, sensor_qos)
        self.create_subscription(Imu, self.gp("imu_topic"), self.on_imu, sensor_qos)
        self.create_subscription(Odometry, self.gp("odom_topic"), self.on_odom, 10)
        self.create_subscription(Bool, self.gp("proximity_topic"), self.on_proximity, 10)

        self.pub_cmd_vel = self.create_publisher(Twist, self.gp("cmd_vel_topic"), 10)
        self.pub_fsm = self.create_publisher(Int32, "/fsm_command", 1)
        self.pub_grip_outcome = self.create_publisher(
            Bool, self.gp("grip_outcome_topic"), 10
        )

        self.create_service(Trigger, "spearhead_servo/record_setpoint", self.srv_record)
        self.create_service(Trigger, "spearhead_servo/start", self.srv_start)
        self.create_service(Trigger, "spearhead_servo/next_step", self.srv_next_step)
        self.create_service(Trigger, "spearhead_servo/abort", self.srv_abort)
        self.create_service(Trigger, "spearhead_servo/set_start_pose", self.srv_set_start_pose)
        self.create_service(Trigger, "spearhead_servo/reset_to_start", self.srv_reset_to_start)

        self.timer = self.create_timer(0.05, self.tick)
        self.get_logger().info(
            f"spearhead_servo ready. teach={self.gp('teach')} "
            f"cmd_vel={self.gp('cmd_vel_topic')} odom={self.gp('odom_topic')} "
            f"state={self.state.name}"
        )
        if self.gp("auto_start") and not self.gp("teach"):
            self._begin()

    def _check_config(self):
        if self.gp("servo_mode") not in ("depth_3d", "bbox_ibvs"):
            self.get_logger().error(
                "CONFIG: servo_mode must be depth_3d or bbox_ibvs."
            )
        if not self.gp("require_proximity_outcome"):
            self.get_logger().warn(
                "CONFIG: require_proximity_outcome=false; grasp attempts will be "
                "logged as 'unverified' and never as 'success'."
            )
        if self._is_bbox_ibvs() and self.gp("detector_mode") != "yolo_bbox":
            self.get_logger().error(
                "CONFIG: bbox_ibvs requires detector_mode=yolo_bbox."
            )
        if self.gp("bbox_target_depth_m") <= self.gp("bbox_overshoot_depth_m"):
            self.get_logger().error(
                "CONFIG: bbox_target_depth_m must be greater than "
                "bbox_overshoot_depth_m."
            )
        if self.gp("depth_min_m") >= self.gp("bbox_overshoot_depth_m"):
            self.get_logger().error(
                "CONFIG: depth_min_m must be below bbox_overshoot_depth_m "
                "so close-target backup can be detected."
            )
        if (
            not self._is_bbox_ibvs()
            and self.gp("final_approach_m") < self.gp("min_move_m")
        ):
            self.get_logger().error(
                "CONFIG: final_approach_m is below min_move_m; blind approach "
                "will not execute. Increase final_approach_m before running."
            )
        if int(self.gp("grip_close_code")) not in [
            int(code) for code in self.gp("grab_codes")
        ]:
            self.get_logger().error(
                "CONFIG: grip_close_code is not in grab_codes; grasp attempts "
                "will not be counted."
            )
        for name in (
            "reset_position_tolerance_m",
            "reset_yaw_tolerance_rad",
            "reset_linear_gain",
            "reset_max_linear_mps",
            "reset_timeout_s",
        ):
            if self.gp(name) <= 0.0:
                self.get_logger().error(f"CONFIG: {name} must be positive.")
        for name in ("initial_move_x", "initial_move_y"):
            distance = abs(float(self.gp(name)))
            if 0.0 < distance < self.gp("min_move_m"):
                self.get_logger().warn(
                    f"CONFIG: {name} is below min_move_m and will be ignored."
                )
        if (
            not self._is_bbox_ibvs()
            and self.gp("setpoint_z") - self.gp("depth_min_m") < 0.03
        ):
            self.get_logger().warn(
                "CONFIG: standoff is within 3 cm of the depth range floor; "
                "final measurement may drop out."
            )

    def _load_setpoint(self):
        try:
            with open(self.gp("setpoint_file"), "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if self._is_bbox_ibvs():
                self.bbox_goal = np.array(
                    [data["u"], data["v"], data["w"], data["h"]],
                    dtype=np.float64,
                )
                self.get_logger().info(
                    f"Loaded bbox goal from file: {self.bbox_goal}"
                    + f"  target depth={self.gp('bbox_target_depth_m'):.3f} m"
                )
            else:
                self.setpoint = np.array(
                    [data["x"], data["y"], data["z"]],
                    dtype=np.float64,
                )
                self.get_logger().info(
                    f"Loaded 3D setpoint from file: {self.setpoint}"
                )
        except Exception as exc:
            self.get_logger().warn(
                "No compatible setpoint file loaded; using parameters. Run "
                f"teach mode before running. ({exc})"
            )

    def _is_bbox_ibvs(self):
        return self.gp("servo_mode") == "bbox_ibvs"

    def _bbox_goal_ready(self):
        return self.bbox_goal[2] > 1.0 and self.bbox_goal[3] > 1.0

    def _load_yolo_model(self):
        if self.gp("detector_mode") != "yolo_bbox" and not self._is_bbox_ibvs():
            return

        weights = str(self.gp("yolo_weights")).strip()
        if not weights:
            self.get_logger().warn(
                "detector_mode=yolo_bbox but yolo_weights is empty; "
                "detection will fall back to depth_blob if enabled."
            )
            return
        if not os.path.exists(weights):
            self.get_logger().warn(
                f"YOLO weights not found: {weights}; detection will fall back "
                "to depth_blob if enabled."
            )
            return

        try:
            from ultralytics import YOLO

            self.yolo_model = YOLO(weights)
            self.get_logger().info(f"Loaded YOLO detector: {weights}")
        except Exception as exc:
            self.get_logger().error(
                f"Failed to load YOLO detector from {weights}: {exc}"
            )

    def on_info(self, msg: CameraInfo):
        if self.K is None:
            self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])
            self.get_logger().info(
                f"Intrinsics: fx={self.K[0]:.1f} fy={self.K[1]:.1f} "
                f"cx={self.K[2]:.1f} cy={self.K[3]:.1f}"
            )

    def on_depth(self, msg: Image):
        self.depth_img = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="passthrough",
        )
        self.depth_seq += 1

    def on_color(self, msg: Image):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.color_seq += 1

    def on_imu(self, msg: Imu):
        g = msg.angular_velocity
        mag = math.sqrt(g.x * g.x + g.y * g.y + g.z * g.z)
        now = time.monotonic()
        if mag < self.gp("gyro_settle_rad_s"):
            if self.last_gyro_quiet_since is None:
                self.last_gyro_quiet_since = now
            self.gyro_quiet = (
                now - self.last_gyro_quiet_since
            ) >= self.gp("settle_dwell_s")
        else:
            self.last_gyro_quiet_since = None
            self.gyro_quiet = False

    def on_odom(self, msg: Odometry):
        """Store the base pose from the Teensy odometry message."""
        q = msg.pose.pose.orientation
        sin_yaw = 2.0 * (q.w * q.z + q.x * q.y)
        cos_yaw = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.odom_yaw = math.atan2(sin_yaw, cos_yaw)
        self.odom_x = float(msg.pose.pose.position.x)
        self.odom_y = float(msg.pose.pose.position.y)
        self.odom_received_at = time.monotonic()

    def on_proximity(self, msg: Bool):
        """Keep the latest proximity reading and mark it as a fresh sample."""
        self.proximity_value = bool(msg.data)
        self.proximity_seq += 1

    def srv_record(self, _req, res):
        if self._is_bbox_ibvs():
            res.success, res.message = self._record_bbox_goal()
            return res

        buf = []
        t0 = time.monotonic()
        while len(buf) < 15 and time.monotonic() - t0 < 3.0:
            measurement = self.detect()
            if measurement is not None:
                buf.append(measurement)
            time.sleep(0.05)

        if not buf:
            res.success = False
            res.message = "No valid detection; cannot record setpoint."
            return res

        setpoint = np.median(np.array(buf), axis=0)
        self.setpoint = setpoint
        try:
            with open(self.gp("setpoint_file"), "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    {
                        "x": float(setpoint[0]),
                        "y": float(setpoint[1]),
                        "z": float(setpoint[2]),
                    },
                    f,
                )
            res.success = True
            res.message = f"Setpoint recorded ({len(buf)} frames): {setpoint}"
        except Exception as exc:
            res.success = False
            res.message = f"Detection OK ({setpoint}) but file write failed: {exc}"

        self.get_logger().info(res.message)
        return res

    def _record_bbox_goal(self):
        required = int(self.gp("bbox_teach_samples"))
        if len(self.teach_bbox_buf) < required:
            return False, (
                f"Need {required} fresh bbox frames "
                f"({len(self.teach_bbox_buf)}/{required})."
            )

        samples = np.array(self.teach_bbox_buf[-required:])
        goal = np.median(samples, axis=0)  # [u,v,w,h] or [u,v,w,h,z]
        self.bbox_goal = goal[:4]
        data = {
            "mode": "bbox_ibvs",
            "u": float(goal[0]),
            "v": float(goal[1]),
            "w": float(goal[2]),
            "h": float(goal[3]),
        }
        z_note = ""
        if samples.shape[1] == 5:
            z_note = (
                f"  measured depth z={float(goal[4]):.3f} m"
                f"; target remains {self.gp('bbox_target_depth_m'):.3f} m"
            )
        try:
            with open(self.gp("setpoint_file"), "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f)
            message = f"BBox goal recorded ({required} fresh frames): {goal[:4]}{z_note}"
        except Exception as exc:
            return False, f"BBox goal OK ({goal[:4]}) but file write failed: {exc}"

        self.get_logger().info(message)
        return True, message

    def srv_start(self, _req, res):
        if self.gp("teach"):
            res.success = False
            res.message = "Node is in teach mode; restart with teach:=false."
            return res
        if self._is_bbox_ibvs() and not self._bbox_goal_ready():
            res.success = False
            res.message = "No bbox goal recorded. Run teach mode and record it first."
            return res
        if not self._have_fresh_odom():
            res.success = False
            res.message = "No fresh odometry; refusing to start without yaw feedback."
            return res

        self._begin()
        res.success = True
        res.message = "Servo started."
        return res

    def srv_next_step(self, _req, res):
        if not self.gp("step_mode"):
            res.success = False
            res.message = "step_mode=false; this service is unavailable."
            return res
        if self.state != State.WAIT_STEP:
            res.success = False
            res.message = f"Not waiting for a step (state={self.state.name})."
            return res

        self._enter_measure()
        res.success = True
        res.message = "Step accepted: visual servo measurement started."
        self.get_logger().info(res.message)
        return res

    def srv_abort(self, _req, res):
        if self.state not in (State.IDLE, State.DONE, State.FAULT, State.RESETTING):
            # An aborted trial still gets a row so it stays in the denominator.
            self._log_attempt("aborted", "operator_abort", None, time.monotonic())
        self.state = State.IDLE
        self.seq = []
        self.move_queue = []
        self._stop_robot()
        res.success = True
        res.message = "Aborted; state=IDLE."
        return res

    def _store_start_pose(self):
        """Record the current odometry pose as the session return target."""
        if not self._have_fresh_odom():
            return False, "No fresh odometry; cannot save start pose."
        self.start_pose = (self.odom_x, self.odom_y, self.odom_yaw)
        return True, (
            "Start pose saved from odometry "
            f"x={self.odom_x:+.3f} y={self.odom_y:+.3f} yaw={self.odom_yaw:+.3f}."
        )

    def srv_set_start_pose(self, _req, res):
        res.success, res.message = self._store_start_pose()
        if res.success:
            self.get_logger().info(res.message)
        else:
            self.get_logger().warn(res.message)
        return res

    def srv_reset_to_start(self, _req, res):
        if self.start_pose is None:
            res.success = False
            res.message = "No start pose saved. Start a run or save the current pose first."
            return res
        if not self._have_fresh_odom():
            res.success = False
            res.message = "No fresh odometry; refusing to reset position."
            return res

        self.seq = []
        self.move_queue = []
        self.active_motion = None
        self.held_yaw = self.start_pose[2]
        self.reset_started_at = time.monotonic()
        self.state = State.RESETTING
        self._stop_robot()
        res.success = True
        res.message = "Returning base to saved start pose using odometry."
        self.get_logger().warn(res.message)
        return res

    def _begin(self):
        # New trial: restart the clocks and the attempt counter first, so any
        # fault raised below is logged against this trial, not the previous one.
        self.trial_started_at = time.monotonic()
        self.grip_attempt_started_at = self.trial_started_at
        self.grip_attempt = 0
        if self._is_bbox_ibvs() and self.yolo_model is None:
            self._fault("no_yolo_model")
            self.get_logger().error(
                "FAULT: bbox_ibvs needs a loaded YOLO model (check yolo_weights)."
            )
            return
        if self._is_bbox_ibvs() and not self._bbox_goal_ready():
            self._fault("no_bbox_goal")
            self.get_logger().error("FAULT: no bbox goal is available.")
            return
        if not self._have_fresh_odom():
            self._fault("no_fresh_odom")
            self.get_logger().error("FAULT: no fresh odometry for yaw hold.")
            return
        if self.start_pose is None:
            saved, message = self._store_start_pose()
            if not saved:
                self._fault("store_start_pose_failed")
                self.get_logger().error(f"FAULT: {message}")
                return
            self.get_logger().info(message)
        self.held_yaw = self.odom_yaw
        self.cycles = 0
        self.measure_failures = 0
        self.move_queue = []
        self.grip_attempt = 0
        codes = self.gp("pre_grab_codes")
        waits = self.gp("pre_grab_waits")
        self.seq = [
            {"type": "fsm", "code": int(code), "wait": float(wait)}
            for code, wait in zip(codes, waits)
        ]
        initial_x = float(self.gp("initial_move_x"))
        initial_y = float(self.gp("initial_move_y"))
        min_m = self.gp("min_move_m")
        if abs(initial_x) >= min_m or abs(initial_y) >= min_m:
            self.seq.append(
                {
                    "type": "move",
                    "x": initial_x if abs(initial_x) >= min_m else 0.0,
                    "y": initial_y if abs(initial_y) >= min_m else 0.0,
                    "wait": float(self.gp("initial_move_wait_s")),
                }
            )
        self.seq_end_state = State.WAIT_STEP if self.gp("step_mode") else State.MEASURE
        self.seq_next_at = 0.0
        self.seq_last_was_move = False
        self.state = State.SEQUENCE
        if self.gp("step_mode"):
            self.get_logger().info(
                "START: moving to defined initial pose, then waiting for "
                "/spearhead_servo/next_step."
            )
        else:
            self.get_logger().info(f"START: pre-grab sequence {codes}")

    def detect(self):
        """Return spearhead centroid [X, Y, Z] in camera frame, or None."""
        self.last_detection_ui = None
        if self.gp("detector_mode") == "yolo_bbox":
            measurement = self._detect_yolo_bbox()
            if measurement is not None:
                return measurement
            if not self.gp("yolo_fallback_depth_blob"):
                return None
        return self._detect_depth_blob()

    def _detect_depth_blob(self):
        if self.depth_img is None or self.K is None:
            return None

        scale = 0.001 if self.gp("depth_scale_mm") else 1.0
        z = self.depth_img.astype(np.float32) * scale

        roi = [int(v) for v in self.gp("roi")]
        rx, ry, rw, rh = roi
        x0, y0 = (rx, ry) if rw > 0 and rh > 0 else (0, 0)
        if rw > 0 and rh > 0:
            z = z[ry : ry + rh, rx : rx + rw]

        mask = (
            (z > self.gp("depth_min_m")) & (z < self.gp("depth_max_m"))
        ).astype(np.uint8)

        if self.gp("use_hsv_reject") and self.color_img is not None:
            color = self.color_img
            if rw > 0 and rh > 0:
                color = color[ry : ry + rh, rx : rx + rw]
            if color.shape[:2] == mask.shape:
                hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
                lo = np.array(self.gp("hsv_lo"), dtype=np.uint8)
                hi = np.array(self.gp("hsv_hi"), dtype=np.uint8)
                reject = cv2.inRange(hsv, lo, hi)
                mask[reject > 0] = 0

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
        if n_labels <= 1:
            return None

        fx, fy, cx, cy = self.K
        best = None
        best_ui = None
        best_score = float("inf")
        for i in range(1, n_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < self.gp("min_blob_px"):
                continue

            ys, xs = np.where(labels == i)
            zi = np.median(z[ys, xs])
            if not np.isfinite(zi) or zi <= 0:
                continue

            width_m = stats[i, cv2.CC_STAT_WIDTH] * zi / fx
            if not (self.gp("min_width_m") <= width_m <= self.gp("max_width_m")):
                continue

            u = centroids[i][0] + x0
            v = centroids[i][1] + y0
            x = (u - cx) * zi / fx
            y = (v - cy) * zi / fy
            score = abs(x - self.setpoint[0]) + 0.5 * abs(zi - self.setpoint[2])
            if score < best_score:
                best_score = score
                best = np.array([x, y, zi])
                best_ui = {
                    "source": "depth_blob",
                    "u": float(u),
                    "v": float(v),
                    "bbox": (
                        int(stats[i, cv2.CC_STAT_LEFT] + x0),
                        int(stats[i, cv2.CC_STAT_TOP] + y0),
                        int(stats[i, cv2.CC_STAT_WIDTH]),
                        int(stats[i, cv2.CC_STAT_HEIGHT]),
                    ),
                    "confidence": None,
                }

        if best is not None:
            self.last_detection_ui = best_ui
        return best

    def _predict_yolo_targets(self):
        """Return YOLO target boxes in full-image pixel coordinates.

        The result is cached per color frame, so callers in the same tick
        (preview, measurement) share one inference and a frame is never run
        through the network twice.
        """
        if self.yolo_model is None or self.color_img is None:
            return []

        roi = [int(v) for v in self.gp("roi")]
        wanted_class = int(self.gp("yolo_class_id"))
        key = (
            self.color_seq,
            tuple(roi),
            float(self.gp("yolo_conf")),
            float(self.gp("yolo_iou")),
            int(self.gp("yolo_imgsz")),
            wanted_class,
        )
        if key == self.yolo_cache_key:
            return list(self.yolo_cache)

        rx, ry, rw, rh = roi
        x0, y0 = (rx, ry) if rw > 0 and rh > 0 else (0, 0)
        color = self.color_img
        if rw > 0 and rh > 0:
            color = color[ry : ry + rh, rx : rx + rw]

        try:
            result = self.yolo_model.predict(
                source=color,
                conf=float(self.gp("yolo_conf")),
                iou=float(self.gp("yolo_iou")),
                imgsz=int(self.gp("yolo_imgsz")),
                verbose=False,
            )[0]
        except Exception as exc:
            self.get_logger().warn(f"YOLO prediction failed: {exc}")
            return []

        targets = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            height, width = self.color_img.shape[:2]
            for box in boxes:
                cls_id = int(box.cls[0]) if box.cls is not None else -1
                if wanted_class >= 0 and cls_id != wanted_class:
                    continue

                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                x1, x2 = sorted((x1 + x0, x2 + x0))
                y1, y2 = sorted((y1 + y0, y2 + y0))
                x1 = max(0, min(x1, width - 1))
                x2 = max(0, min(x2, width))
                y1 = max(0, min(y1, height - 1))
                y2 = max(0, min(y2, height))
                if x2 <= x1 or y2 <= y1:
                    continue

                targets.append(
                    {
                        "source": "yolo_bbox",
                        "u": 0.5 * (x1 + x2),
                        "v": 0.5 * (y1 + y2),
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                        "confidence": (
                            float(box.conf[0]) if box.conf is not None else 0.0
                        ),
                        "class_id": cls_id,
                    }
                )

        self.yolo_cache_key = key
        self.yolo_cache = targets
        return list(targets)

    def _detect_bbox_target(self):
        """Choose a YOLO box without reading depth data."""
        targets = self._predict_yolo_targets()
        if not targets:
            self.last_detection_ui = None
            return None

        if self.gp("teach") or not self._bbox_goal_ready():
            best = max(targets, key=lambda target: target["confidence"])
        else:
            goal_u, goal_v, goal_w, goal_h = self.bbox_goal

            def score(target):
                width, height = target["bbox"][2:]
                target_scale = math.sqrt((width * height) / (goal_w * goal_h))
                center_error = math.hypot(
                    (target["u"] - goal_u) / goal_w,
                    (target["v"] - goal_v) / goal_h,
                )
                return center_error + abs(target_scale - 1.0) - 0.05 * target["confidence"]

            best = min(targets, key=score)

        self.last_detection_ui = best
        return best

    def _detect_bbox_target_with_depth(self):
        """Choose a YOLO box (same as _detect_bbox_target) and sample
        RealSense depth at the inner centre of that box.

        Returns
        -------
        (target, z_m) where z_m is the median depth in metres, or None if
        depth is unavailable or out of the configured range.
        """
        target = self._detect_bbox_target()
        if target is None:
            return None, None
        if self.depth_img is None or self.K is None:
            return target, None

        scale = 0.001 if self.gp("depth_scale_mm") else 1.0
        z_full = self.depth_img.astype(np.float32) * scale
        x1, y1, bw, bh = target["bbox"]
        x2, y2 = x1 + bw, y1 + bh
        # Sample the inner 50 % of the box to avoid background creep
        cx1 = x1 + int(0.25 * bw)
        cx2 = x2 - int(0.25 * bw)
        cy1 = y1 + int(0.25 * bh)
        cy2 = y2 - int(0.25 * bh)
        if cx2 <= cx1 or cy2 <= cy1:
            cx1, cx2, cy1, cy2 = x1, x2, y1, y2
        z_crop = z_full[cy1:cy2, cx1:cx2]
        valid = (z_crop > self.gp("depth_min_m")) & (z_crop < self.gp("depth_max_m"))
        if not valid.any():
            return target, None
        z_m = float(np.median(z_crop[valid]))
        return target, z_m

    def _detect_yolo_bbox(self):
        if self.depth_img is None or self.K is None:
            return None

        scale = 0.001 if self.gp("depth_scale_mm") else 1.0
        z_full = self.depth_img.astype(np.float32) * scale
        fx, fy, cx, cy = self.K
        best = None
        best_ui = None
        best_score = float("inf")
        for target in self._predict_yolo_targets():
            x1, y1, bw, bh = target["bbox"]
            x2, y2 = x1 + bw, y1 + bh

            # Use the center half of the box so background inside loose boxes
            # does not dominate the depth median.
            cx1 = x1 + int(0.25 * bw)
            cx2 = x2 - int(0.25 * bw)
            cy1 = y1 + int(0.25 * bh)
            cy2 = y2 - int(0.25 * bh)
            if cx2 <= cx1 or cy2 <= cy1:
                cx1, cx2, cy1, cy2 = x1, x2, y1, y2

            z_crop = z_full[cy1:cy2, cx1:cx2]
            valid = (z_crop > self.gp("depth_min_m")) & (
                z_crop < self.gp("depth_max_m")
            )
            if not valid.any():
                continue

            zi = float(np.median(z_crop[valid]))
            u, v = target["u"], target["v"]
            x = (u - cx) * zi / fx
            y = (v - cy) * zi / fy
            measurement = np.array([x, y, zi])
            setpoint_distance = (
                abs(x - self.setpoint[0]) + 0.5 * abs(zi - self.setpoint[2])
            )
            score = setpoint_distance - 0.05 * target["confidence"]
            if score < best_score:
                best_score = score
                best = measurement
                best_ui = target

        if best is not None:
            self.last_detection_ui = best_ui
        return best

    def _show_teach_ui(self, measurement=None, bbox_target=None):
        """Show live measurements and the current target while teaching or running."""
        if not self.teach_ui_visible or self.color_img is None:
            return

        frame = self.color_img.copy()
        height = frame.shape[0]
        ui = bbox_target or self.last_detection_ui

        roi = [int(v) for v in self.gp("roi")]
        if roi[2] > 0 and roi[3] > 0:
            cv2.rectangle(
                frame,
                (roi[0], roi[1]),
                (roi[0] + roi[2], roi[1] + roi[3]),
                (80, 180, 255),
                1,
            )

        if ui is None:
            lines = ["TARGET: NOT DETECTED"]
            color = (0, 0, 255)
        else:
            u, v = int(ui["u"]), int(ui["v"])
            bx, by, bw, bh = ui["bbox"]
            cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            cv2.drawMarker(
                frame,
                (u, v),
                (0, 255, 255),
                cv2.MARKER_CROSS,
                18,
                2,
            )
            color = (255, 255, 255)

            if self._is_bbox_ibvs():
                lines = [
                    f"bbox C: ({u}, {v})  size: {bw} x {bh}",
                    f"conf: {ui['confidence']:.2f}",
                ]
                if self._bbox_goal_ready():
                    goal_u, goal_v, goal_w, goal_h = self.bbox_goal
                    gx = int(goal_u - 0.5 * goal_w)
                    gy = int(goal_v - 0.5 * goal_h)
                    cv2.rectangle(
                        frame,
                        (gx, gy),
                        (gx + int(goal_w), gy + int(goal_h)),
                        (255, 120, 0),
                        2,
                    )
                    scale = math.sqrt((bw * bh) / (goal_w * goal_h))
                    lines.append(
                        f"goal dU: {u - goal_u:+.1f} px  "
                        f"dV: {v - goal_v:+.1f} px  scale: {scale:.3f}"
                    )
            elif measurement is not None:
                x, y, z = measurement
                range_m = float(np.linalg.norm(measurement))
                lines = [
                    f"{ui['source']}  Z: {z:.3f} m  R: {range_m:.3f} m",
                    f"X: {x:+.3f} m  Y: {y:+.3f} m",
                ]
                if ui["confidence"] is not None:
                    lines[0] += f"  conf: {ui['confidence']:.2f}"
            else:
                lines = ["TARGET: NO VALID DEPTH"]
                color = (0, 0, 255)

        for index, text in enumerate(lines):
            y_text = 30 + index * 28
            cv2.putText(
                frame,
                text,
                (16, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0),
                4,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                text,
                (16, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                1,
                cv2.LINE_AA,
            )

        mode_label = "TEACH MODE" if self.gp("teach") else "RUN MODE"
        save_key = str(self.gp("teach_save_key") or "s")[0].lower()
        footer = (
            f"{mode_label} - {save_key}: save goal, q: hide preview"
            if self.gp("teach")
            else f"{mode_label} - q: hide preview"
        )
        cv2.putText(
            frame,
            footer,
            (16, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        if time.monotonic() < self.teach_status_until:
            cv2.putText(
                frame,
                self.teach_status,
                (16, height - 42),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0) if self.teach_status.startswith("Saved") else (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        try:
            cv2.imshow(self.gp("teach_window_name"), frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.teach_ui_visible = False
                cv2.destroyWindow(self.gp("teach_window_name"))
                self.get_logger().info("Teach camera preview hidden.")
            elif (
                self.gp("teach")
                and self._is_bbox_ibvs()
                and key in (ord(save_key), ord(save_key.upper()))
            ):
                success, message = self._record_bbox_goal()
                self.teach_status = "Saved bbox goal." if success else message
                self.teach_status_until = time.monotonic() + 4.0
                if success:
                    self.get_logger().info(message)
                else:
                    self.get_logger().warn(message)
        except cv2.error as exc:
            self.get_logger().warn(
                f"Teach camera preview is unavailable: {exc}",
                throttle_duration_sec=5.0,
            )

    def _capture_teach_bbox(self, target, z_m=None):
        """Keep only detections from distinct color frames for bbox teach.

        When bbox_use_depth_z is True, z_m must be valid to accept the frame.
        """
        if target is None or self.color_seq == self.last_teach_bbox_seq:
            return
        if self.gp("bbox_use_depth_z") and z_m is None:
            return  # skip frames with no valid depth

        _, _, width, height = target["bbox"]
        if self.gp("bbox_use_depth_z"):
            vec = np.array(
                [target["u"], target["v"], width, height, z_m], dtype=np.float64
            )
        else:
            vec = np.array(
                [target["u"], target["v"], width, height], dtype=np.float64
            )
        self.teach_bbox_buf.append(vec)
        keep = max(2 * int(self.gp("bbox_teach_samples")), 30)
        self.teach_bbox_buf = self.teach_bbox_buf[-keep:]
        self.last_teach_bbox_seq = self.color_seq

    def _plan_moves(self, fwd, strafe):
        """Return one continuous command; cmd_vel has no firmware move floor."""
        if abs(fwd) <= 1e-4 and abs(strafe) <= 1e-4:
            return []
        return [(fwd, strafe)]

    def _have_fresh_odom(self):
        return (
            self.odom_yaw is not None
            and time.monotonic() - self.odom_received_at <= self.gp("odom_timeout_s")
        )

    def _yaw_command(self):
        if self.held_yaw is None or not self._have_fresh_odom():
            return 0.0
        return heading_command(
            self.held_yaw,
            self.odom_yaw,
            self.gp("heading_hold_gain"),
            self.gp("max_yaw_radps"),
        )

    def _yaw_aligned(self):
        if self.held_yaw is None or not self._have_fresh_odom():
            return False  # cannot be evaluated, so do not pass the grip gate
        return abs(self._yaw_command()) <= self.gp("heading_tolerance_rad") * self.gp("heading_hold_gain")

    def _stop_robot(self):
        self.active_motion = None
        self.pub_cmd_vel.publish(Twist())

    @staticmethod
    def _elapsed(start, now):
        return None if start is None else max(0.0, now - start)

    def _log_attempt(self, outcome, cause, sensor_value, now=None):
        """Write one CSV row; return seconds since the last close command.

        attempt_duration_s runs from the close command (or from trial start
        for anything that happens before a grip); trial_duration_s always runs
        from the start of the trial.
        """
        now = time.monotonic() if now is None else now
        attempt_s = self._elapsed(self.grip_attempt_started_at, now) or 0.0
        trial_s = self._elapsed(self.trial_started_at, now)
        self.attempt_logger.log(
            self.grip_attempt, outcome, cause, sensor_value, attempt_s, trial_s
        )
        return attempt_s

    def _cycle_limit_hit(self):
        """Count one corrective servo cycle; fault and return True past max_cycles."""
        self.cycles += 1
        if self.cycles > self.gp("max_cycles"):
            self._fault("max_servo_cycles_exceeded")
            self.get_logger().error("FAULT: max servo cycles exceeded.")
            return True
        return False

    def _fault(self, cause: str):
        """Transition to FAULT, stop the robot, and log a fault row to the attempt CSV."""
        self.state = State.FAULT
        self._stop_robot()
        self._log_attempt("fault", cause, None)
        self.get_logger().error(f"FAULT logged to CSV: {cause}")

    def _publish_motion(self, x, y, duration_s):
        """Publish a bounded body-frame Twist while correcting held yaw."""
        if not self._have_fresh_odom():
            self._fault("odom_stale")
            self.get_logger().error("FAULT: odometry became stale; stopping robot.")
            return
        msg = Twist()
        msg.linear.x = float(
            np.clip(x / duration_s, -self.gp("max_forward_mps"), self.gp("max_forward_mps"))
        )
        msg.linear.y = float(
            np.clip(y / duration_s, -self.gp("max_strafe_mps"), self.gp("max_strafe_mps"))
        )
        msg.angular.z = float(self._yaw_command())
        self.pub_cmd_vel.publish(msg)
        self.get_logger().info(
            f"/cmd_vel x={msg.linear.x:+.4f} y={msg.linear.y:+.4f} "
            f"yaw={msg.angular.z:+.4f}",
            throttle_duration_sec=0.5,
        )

    def _send_move(self, x, y, duration_s=None):
        """Start a bounded move and keep it active until its scheduled stop."""
        duration_s = max(float(duration_s or self.gp("move_wait_s")), 0.05)
        self.active_motion = (float(x), float(y), duration_s)
        self.move_sent_at = time.monotonic()
        self._publish_motion(*self.active_motion)

    def _refresh_motion(self):
        if self.active_motion is not None:
            self._publish_motion(*self.active_motion)

    def _enter_measure(self, final=False):
        self.state = State.FINAL_MEASURE if final else State.MEASURE
        self.measure_buf = []
        self.measure_start = time.monotonic()
        # Only frames that arrive after this point may enter the measurement.
        self.last_measure_seq = self.color_seq
        self.measure_depth_seq0 = self.depth_seq

    def _measure_failed(self, why):
        self.measure_failures += 1
        self.get_logger().warn(
            f"Measurement failed ({why}) "
            f"[{self.measure_failures}/{self.gp('max_measure_failures')}]"
        )
        if self.measure_failures >= self.gp("max_measure_failures"):
            self._fault("repeated_measurement_failures")
            self.get_logger().error("FAULT: repeated measurement failures. Robot holding.")
        else:
            self._enter_measure(final=(self.state == State.FINAL_MEASURE))

    def tick(self):
        now = time.monotonic()

        if self.gp("teach"):
            if self._is_bbox_ibvs():
                if self.gp("bbox_use_depth_z"):
                    target, z_m = self._detect_bbox_target_with_depth()
                    self._capture_teach_bbox(target, z_m)
                else:
                    target = self._detect_bbox_target()
                    self._capture_teach_bbox(target)
                self._show_teach_ui(bbox_target=target)
                if target is not None:
                    _, _, width, height = target["bbox"]
                    z_log = f" z={z_m:.3f}m" if self.gp("bbox_use_depth_z") and z_m is not None else ""
                    self.get_logger().info(
                        f"[TEACH BBOX] u={target['u']:.1f} "
                        f"v={target['v']:.1f} w={width} h={height}{z_log} "
                        f"samples={len(self.teach_bbox_buf)}/"
                        f"{self.gp('bbox_teach_samples')}",
                        throttle_duration_sec=0.5,
                    )
                return

            measurement = self.detect()
            self._show_teach_ui(measurement)
            if measurement is not None:
                self.get_logger().info(
                    f"[TEACH] X={measurement[0]:+.4f} "
                    f"Y={measurement[1]:+.4f} Z={measurement[2]:.4f} m "
                    f"(dX={measurement[0] - self.setpoint[0]:+.4f} "
                    f"dZ={measurement[2] - self.setpoint[2]:+.4f})",
                    throttle_duration_sec=0.5,
                )
            return

        # The preview is the only consumer of this detection (measuring states
        # detect for themselves), so skip the inference when no window is shown.
        if self.teach_ui_visible:
            if self._is_bbox_ibvs():
                target = self._detect_bbox_target()
                self._show_teach_ui(bbox_target=target)
            else:
                measurement = self.detect()
                self._show_teach_ui(measurement)

        state = self.state
        if state in (State.IDLE, State.WAIT_STEP, State.DONE):
            return

        if state == State.FAULT:
            self._stop_robot()
            return

        if state == State.RESETTING:
            self._tick_reset_to_start(now)
            return

        if state in (State.SEQUENCE, State.SETTLE):
            self._refresh_motion()

        if state == State.SEQUENCE:
            self._tick_sequence(now)
            return

        if state in (State.MEASURE, State.FINAL_MEASURE):
            self._tick_measure(now, final=(state == State.FINAL_MEASURE))
            return

        if state == State.SETTLE:
            self._tick_settle(now)
            return

        if state == State.VERIFY_GRIP:
            self._tick_grip_verification(now)

    def _tick_reset_to_start(self, now):
        """Drive the base back to the recorded odometry pose with bounded speed."""
        if not self._have_fresh_odom():
            self._fault("odom_stale_during_reset")
            self.get_logger().error("FAULT: odometry became stale during reset.")
            return
        if now - self.reset_started_at >= self.gp("reset_timeout_s"):
            self._fault("reset_timeout")
            self.get_logger().error("FAULT: reset-to-start timed out; stopping robot.")
            return

        start_x, start_y, start_yaw = self.start_pose
        distance_m = math.hypot(start_x - self.odom_x, start_y - self.odom_y)
        yaw_error = abs(wrap_angle(start_yaw - self.odom_yaw))
        if (
            distance_m <= self.gp("reset_position_tolerance_m")
            and yaw_error <= self.gp("reset_yaw_tolerance_rad")
        ):
            self.state = State.IDLE
            self._stop_robot()
            self.get_logger().info(
                f"RESET COMPLETE: position error={distance_m:.3f} m, "
                f"yaw error={yaw_error:.3f} rad."
            )
            return

        command = return_to_start_command(
            start_x,
            start_y,
            start_yaw,
            self.odom_x,
            self.odom_y,
            self.odom_yaw,
            self.gp("reset_linear_gain"),
            self.gp("reset_max_linear_mps"),
            self.gp("heading_hold_gain"),
            self.gp("max_yaw_radps"),
        )
        msg = Twist()
        msg.linear.x = command.x_mps
        msg.linear.y = command.y_mps
        msg.angular.z = command.yaw_radps
        self.pub_cmd_vel.publish(msg)

    def _tick_sequence(self, now):
        if now < self.seq_next_at:
            return
        if self.seq_last_was_move:
            self._stop_robot()
        if self.seq_last_was_move and not self._imu_ready():
            if now < self.seq_next_at + self.gp("settle_timeout_s"):
                return
            self.get_logger().warn("Sequence: settle timeout after move; continuing.")

        if not self.seq:
            self.get_logger().info(f"Sequence done -> {self.seq_end_state.name}")
            if self.seq_end_state in (State.MEASURE, State.FINAL_MEASURE):
                self._enter_measure(final=(self.seq_end_state == State.FINAL_MEASURE))
            else:
                self.state = self.seq_end_state
                if self.state == State.WAIT_STEP:
                    self.get_logger().info(
                        "STEP READY: initial pose reached. Run "
                        "ros2 service call /spearhead_servo/next_step "
                        "std_srvs/srv/Trigger {} to begin visual servo."
                    )
            return

        item = self.seq.pop(0)
        if item["type"] == "fsm":
            if item["code"] == int(self.gp("grip_close_code")):
                self.grip_attempt += 1
                self.grip_attempt_started_at = now
            self.pub_fsm.publish(Int32(data=item["code"]))
            self.get_logger().info(f"/fsm_command {item['code']}")
            self.seq_last_was_move = False
        elif item["type"] == "verify_grip":
            self.grip_verify_started_at = now
            self.proximity_seq_at_verify_start = self.proximity_seq
            self.state = State.VERIFY_GRIP
            self.seq_last_was_move = False
            return
        else:
            self._send_move(item["x"], item["y"], item["wait"])
            self.seq_last_was_move = True
        self.seq_next_at = now + item["wait"]

    def _tick_grip_verification(self, now):
        """Decide the grasp outcome from the proximity sensor.

        Runs after the close, lift, wiggle and retreat have finished. Uses the
        first proximity sample received after verification started; if none
        arrives within grip_outcome_timeout_s the attempt is a failure
        (proximity_timeout). With require_proximity_outcome=false nothing is
        verified, so the attempt is logged as 'unverified', never 'success'.
        """
        if not self.gp("require_proximity_outcome"):
            self._log_attempt("unverified", "proximity_not_required", None, now)
            self.get_logger().warn(
                f"GRIP ATTEMPT {self.grip_attempt}: unverified "
                "(require_proximity_outcome=false)"
            )
            self.seq_next_at = now
            self.state = State.SEQUENCE
            return

        fresh = self.proximity_seq > self.proximity_seq_at_verify_start
        waited_s = now - self.grip_verify_started_at
        if not fresh and waited_s < float(self.gp("grip_outcome_timeout_s")):
            return  # keep waiting for a sample taken after the retreat

        sensor_value = self.proximity_value if fresh else None
        outcome, cause = proximity_outcome(sensor_value)
        attempt_duration_s = self._log_attempt(outcome, cause, sensor_value, now)
        self.pub_grip_outcome.publish(Bool(data=(outcome == "success")))
        self.get_logger().info(
            f"GRIP ATTEMPT {self.grip_attempt}: {outcome}"
            f"{' (' + cause + ')' if cause else ''} after {attempt_duration_s:.3f}s, "
            f"proximity={sensor_value}"
        )
        self.seq_next_at = now
        self.state = State.SEQUENCE

    def _tick_measure(self, now, final):
        if self._is_bbox_ibvs():
            self._tick_bbox_measure(now)
            return

        if not self._imu_ready():
            timeout = self.gp("settle_timeout_s") + self.gp("measure_timeout_s")
            if now - self.measure_start > timeout:
                self._measure_failed("IMU never settled")
            return

        measurement = self.detect()
        if measurement is not None:
            self.measure_buf.append(measurement)

        if len(self.measure_buf) >= self.gp("frames_per_measurement"):
            median_measurement = np.median(np.array(self.measure_buf), axis=0)
            self.measure_failures = 0
            if final:
                self._final(median_measurement)
            else:
                self._decide(median_measurement)
            return

        if now - self.measure_start > self.gp("measure_timeout_s"):
            self._measure_failed("detection timeout")

    def _tick_bbox_measure(self, now):
        if not self._imu_ready():
            timeout = self.gp("settle_timeout_s") + self.gp("measure_timeout_s")
            if now - self.measure_start > timeout:
                self._measure_failed("IMU never settled")
            return

        # One sample per new color frame, with a new depth frame since the stop:
        # repeated ticks on the same frame must not fill the median buffer.
        if (
            self.color_seq != self.last_measure_seq
            and self.depth_seq > self.measure_depth_seq0
        ):
            self.last_measure_seq = self.color_seq
            target, z_m = self._detect_bbox_target_with_depth()
            if target is not None and z_m is not None:
                _, _, width, height = target["bbox"]
                self.measure_buf.append(
                    np.array(
                        [target["u"], target["v"], width, height, z_m],
                        dtype=np.float64,
                    )
                )

        if len(self.measure_buf) >= self.gp("frames_per_measurement"):
            bbox = np.median(np.array(self.measure_buf), axis=0)
            self.measure_failures = 0
            self._decide_bbox(bbox)
            return

        if now - self.measure_start > self.gp("measure_timeout_s"):
            self._measure_failed("bbox detection timeout")

    def _imu_ready(self):
        return self.gp("ignore_imu") or self.gyro_quiet

    def _tick_settle(self, now):
        if now - self.move_sent_at >= self.gp("move_wait_s") and self._imu_ready():
            self._stop_robot()
        elif now - self.move_sent_at > (
            self.gp("settle_timeout_s") + self.gp("move_wait_s")
        ):
            self.get_logger().warn("Settle timeout; continuing.")
        else:
            return

        if self.move_queue:
            x, y = self.move_queue.pop(0)
            self._send_move(x, y)
        else:
            self._enter_measure()

    def _decide(self, measurement):
        err_x = measurement[0] - self.setpoint[0]
        err_y = measurement[1] - self.setpoint[1]
        err_z = measurement[2] - self.setpoint[2]
        self.get_logger().info(
            f"meas X={measurement[0]:+.4f} Y={measurement[1]:+.4f} "
            f"Z={measurement[2]:.4f} | lat={err_x:+.4f} "
            f"vert={err_y:+.4f} depth={err_z:+.4f}"
        )

        lat_ok = abs(err_x) <= self.gp("tol_lateral_m")
        dep_ok = abs(err_z) <= self.gp("tol_depth_m")
        if lat_ok and dep_ok:
            self.get_logger().info("Converged at standoff -> FINAL_MEASURE.")
            self._enter_measure(final=True)
            return

        self.cycles += 1
        if self.cycles > self.gp("max_cycles"):
            self._fault("max_servo_cycles_exceeded")
            self.get_logger().error("FAULT: max servo cycles exceeded.")
            return

        gain = self.gp("gain")
        clamp = self.gp("max_step_m")
        strafe_clamp = self.gp("max_strafe_m")
        fwd = 0.0 if dep_ok else float(
            np.clip(self.gp("fwd_from_z_sign") * gain * err_z, -clamp, clamp)
        )
        strafe = 0.0 if lat_ok else float(
            np.clip(self.gp("strafe_from_x_sign") * gain * err_x, -strafe_clamp, strafe_clamp)
        )

        self.move_queue = self._plan_moves(fwd, strafe)
        if not self.move_queue:
            self._enter_measure()
            return

        x, y = self.move_queue.pop(0)
        self._send_move(x, y)
        if self.state != State.FAULT:
            self.state = State.SETTLE
        self.get_logger().info(
            f"cycle {self.cycles}: {1 + len(self.move_queue)} move(s) planned"
        )

    def _decide_bbox(self, bbox):
        u, v, width, height = bbox[0], bbox[1], bbox[2], bbox[3]
        goal_u, goal_v, goal_w, goal_h = self.bbox_goal
        err_u = u - goal_u
        err_v = v - goal_v
        u_ok = abs(err_u) <= self.gp("bbox_tol_u_px")
        v_ok = abs(err_v) <= self.gp("bbox_tol_v_px")

        if len(bbox) != 5:
            self._fault("bbox_no_depth")
            self.get_logger().error("FAULT: bbox servo requires a valid depth measurement.")
            return

        # --- bbox chooses/centers the target; depth controls standoff ---
        z_m = bbox[4]
        target_depth_m = self.gp("bbox_target_depth_m")
        err_z = z_m - target_depth_m
        fwd_ok = abs(err_z) <= self.gp("tol_depth_m")
        fwd_log = f"dZ={err_z:+.3f} m (z={z_m:.3f} goal={target_depth_m:.3f})"

        self.get_logger().info(
            f"bbox C=({u:.1f}, {v:.1f}) size=({width:.1f}, {height:.1f}) "
            f"| dU={err_u:+.1f}px dV={err_v:+.1f}px {fwd_log}"
        )

        if z_m < self.gp("bbox_overshoot_depth_m"):
            if self._cycle_limit_hit():
                return
            backup, _ = depth_standoff_command(
                z_m,
                self.gp("bbox_target_depth_m"),
                self.gp("bbox_overshoot_depth_m"),
                self.gp("gain"),
                self.gp("max_forward_mps") * self.gp("move_wait_s"),
                self.gp("fwd_from_z_sign"),
            )
            self.get_logger().warn(
                f"BBOX OVERSHOOT: z={z_m:.3f} m < "
                f"{self.gp('bbox_overshoot_depth_m'):.3f} m; backing up only."
            )
            self.move_queue = self._plan_moves(backup.x_mps, 0.0)
            if not self.move_queue:
                self._fault("bbox_backup_not_planned")
                self.get_logger().error("FAULT: overshoot but no backup move was planned.")
                return
            x, y = self.move_queue.pop(0)
            self._send_move(x, y)
            if self.state != State.FAULT:
                self.state = State.SETTLE
            return

        if u_ok and v_ok and fwd_ok and self._yaw_aligned():
            self.get_logger().info(
                "BBOX GOAL reached within tolerance -> grip, arm up, retreat."
            )
            self._queue_grab_and_retreat()
            return

        if u_ok and v_ok and fwd_ok:
            if self._cycle_limit_hit():
                return
            self.get_logger().info("BBOX position/depth ready; correcting held yaw.")
            self._send_move(0.0, 0.0)
            if self.state != State.FAULT:
                self.state = State.SETTLE
            return

        # This node has no pitch/vertical correction. Once horizontal and
        # forward distance are correct, a remaining vertical image error is unsafe.
        if u_ok and fwd_ok and not v_ok:
            self._fault("bbox_vertical_mismatch")
            self.get_logger().error(
                "FAULT: bbox vertical goal mismatch cannot be corrected. "
                "Check camera mounting or robot rotation."
            )
            return

        if self._cycle_limit_hit():
            return

        clamp = self.gp("max_step_m")
        strafe_clamp = self.gp("max_strafe_m")

        fwd = 0.0 if fwd_ok else float(
            np.clip(
                self.gp("fwd_from_z_sign") * self.gp("gain") * err_z,
                -clamp, clamp,
            )
        )

        strafe = 0.0 if u_ok else float(
            np.clip(
                self.gp("strafe_from_x_sign")
                * self.gp("bbox_lateral_gain_m_per_px")
                * err_u,
                -strafe_clamp,
                strafe_clamp,
            )
        )

        self.move_queue = self._plan_moves(fwd, strafe)
        if not self.move_queue:
            self._fault("bbox_no_move_planned")
            self.get_logger().error(
                "FAULT: bbox error is outside tolerance but no move was planned."
            )
            return

        x, y = self.move_queue.pop(0)
        self._send_move(x, y)
        if self.state != State.FAULT:
            self.state = State.SETTLE
        self.get_logger().info(
            f"bbox cycle {self.cycles}: {1 + len(self.move_queue)} move(s) planned"
        )

    def _final(self, measurement):
        sign = self.gp("fwd_from_z_sign")
        lat_resid = measurement[0] - self.setpoint[0]
        approach = self.gp("final_approach_m") + (
            measurement[2] - self.setpoint[2]
        )
        fwd = sign * approach

        if abs(fwd) < self.gp("min_move_m"):
            self._fault("blind_approach_under_floor")
            self.get_logger().error(
                f"FAULT: blind approach {fwd:+.3f} is under the move floor "
                f"({self.gp('min_move_m')}). Increase final_approach_m or teach "
                "a deeper standoff."
            )
            return

        self.get_logger().info(
            f"BLIND_APPROACH x={fwd:+.4f}; lateral residual "
            f"{lat_resid:+.4f} m dropped, then grab {self.gp('grab_codes')}, "
            "then retreat."
        )

        self.seq = [
            {
                "type": "move",
                "x": float(fwd),
                "y": 0.0,
                "wait": self.gp("blind_wait_s"),
            }
        ]
        self._queue_grab_and_retreat()

    def _queue_grab_and_retreat(self):
        """Append grab commands and retreat without adding a blind approach."""
        sign = self.gp("fwd_from_z_sign")

        # 1. Close the gripper first
        for code, wait in zip(self.gp("grab_codes"), self.gp("grab_waits")):
            self.seq.append({"type": "fsm", "code": int(code), "wait": float(wait)})

        # 2. Back-and-forth wiggle after grip is closed
        for _ in range(int(self.gp("wiggle_repeats"))):
            self.seq.append({
                "type": "move",
                "x": float(-sign * self.gp("wiggle_back_m")),
                "y": 0.0,
                "wait": float(self.gp("wiggle_wait_s")),
            })
            self.seq.append({
                "type": "move",
                "x": float(sign * self.gp("wiggle_fwd_m")),
                "y": 0.0,
                "wait": float(self.gp("wiggle_wait_s")),
            })
        self.seq.append(
            {
                "type": "move",
                "x": float(-self.gp("fwd_from_z_sign") * self.gp("retreat_m")),
                "y": 0.0,
                "wait": self.gp("retreat_wait_s"),
            }
        )
        # Read the sensor only after the close, lift, and retreat are complete.
        self.seq.append({"type": "verify_grip", "wait": 0.0})
        self.seq_end_state = State.DONE
        self.seq_next_at = 0.0
        self.seq_last_was_move = False
        self.state = State.SEQUENCE


def main(args=None):
    rclpy.init(args=args)
    node = SpearheadServo()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == "__main__":
    main()