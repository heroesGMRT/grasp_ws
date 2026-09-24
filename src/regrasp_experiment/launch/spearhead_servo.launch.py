"""Launch the spearhead visual servo node along with proximity sensor and controller.

Original arguments and defaults are unchanged. On top of that, runs can be
tagged (method / condition / clutter / object group / run id) so attempt logs
can be pooled into the D1, D5 and D6 tables, and any parameter that already
exists in the YAML can be overridden for sweeps (D3) and detector baselines (D4).

Nothing new is passed to the node unless you set it: empty argument = the YAML
value stays in charge.
"""

import hashlib
import json
import os
import re
from datetime import datetime

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

OBJECT_GROUPS = ("seen", "unseen", "adversarial")  # D5
CLUTTER_LEVELS = ("c1", "c2", "c3")  # D6
SERVO_MODES = ("bbox_ibvs", "depth_3d")
NODE_NAME = "spearhead_servo"


def _slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-") or "na"


def _yaml_params(path):
    """Return the node's ros__parameters block from a params file ({} on failure)."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return dict(data.get(NODE_NAME, {}).get("ros__parameters", {}))
    except Exception:
        return {}


def _parse_overrides(text, known):
    """Parse 'gain=0.6;max_cycles=8' into typed values; reject unknown keys.

    Keys must already exist in the YAML: rclpy silently ignores overrides for
    parameters the node does not declare, so a typo would otherwise do nothing.
    """
    out = {}
    for item in filter(None, (chunk.strip() for chunk in text.split(";"))):
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"'overrides' entry '{item}' is not key=value.")
        if known and key not in known:
            raise ValueError(
                f"'overrides' key '{key}' is not in the params file; "
                "check the spelling against the node's declared parameters."
            )
        out[key] = yaml.safe_load(raw.strip())
    return out


def _launch_setup(context, *args, **kwargs):
    def s(name):
        return LaunchConfiguration(name).perform(context).strip()

    def b(name):
        return s(name).lower() in ("true", "1", "yes", "on")

    def choice(name, allowed, empty_ok=False):
        value = s(name).lower()
        if value == "" and empty_ok:
            return ""
        if value not in allowed:
            raise ValueError(
                f"Launch argument '{name}' must be one of {allowed}, got '{value}'."
            )
        return value

    actions = []
    teach = b("teach")
    params_file = s("params_file")
    yaml_params = _yaml_params(params_file)

    # ---- tags (D1, D5, D6). Empty = untagged. ---------------------------
    tags = {
        "method": s("method"),
        "condition": s("condition"),
        "run_id": s("run_id"),
        "object_group": choice("object_group", OBJECT_GROUPS, empty_ok=True),
        "object_id": s("object_id"),
        "clutter_level": choice("clutter_level", CLUTTER_LEVELS, empty_ok=True).upper(),
    }

    # ---- overrides of existing YAML parameters (D3 sweep, D4 detector) ----
    overrides = _parse_overrides(s("overrides"), set(yaml_params))
    if s("yolo_weights"):
        overrides["yolo_weights"] = s("yolo_weights")
    if s("yolo_conf"):
        overrides["yolo_conf"] = float(s("yolo_conf"))

    servo_mode = choice("servo_mode", SERVO_MODES)
    effective = dict(yaml_params)
    effective.update(overrides)

    # ---- attempt log path -------------------------------------------------
    attempt_log_file = s("attempt_log_file")
    tagged = any(tags.values()) or bool(overrides)
    if not attempt_log_file and tagged:
        parts = [tags["method"], tags["condition"], tags["clutter_level"], tags["object_group"]]
        weights = effective.get("yolo_weights", "")
        if s("yolo_weights") and weights:
            parts.append("det-" + os.path.splitext(os.path.basename(weights))[0])
        if overrides.keys() - {"yolo_weights"}:
            digest = hashlib.sha1(
                json.dumps(overrides, sort_keys=True, default=str).encode()
            ).hexdigest()[:6]
            parts.append("ov-" + digest)
        if tags["run_id"]:
            parts.append("run" + tags["run_id"])
        stem = "_".join(_slug(p) for p in parts if p) or "run"
        attempt_log_file = os.path.join(s("log_dir"), stem + ".csv")
    elif not attempt_log_file:
        attempt_log_file = "/home/heroes/spearhead_attempt_log.csv"  # original default

    params = {
        "teach": teach,
        "auto_start": b("auto_start"),
        "setpoint_file": s("setpoint_file"),
        "teach_show_ui": b("teach_show_ui"),
        "step_mode": b("step_mode"),
        "servo_mode": servo_mode,
        "cmd_vel_topic": s("cmd_vel_topic"),
        "odom_topic": s("odom_topic"),
        "proximity_topic": s("proximity_topic"),
        "attempt_log_file": attempt_log_file,
    }
    params.update(overrides)
    effective.update(params)

    # ---- metadata beside the CSV, for reproducibility ---------------------
    if not teach and tagged:
        os.makedirs(os.path.dirname(attempt_log_file) or ".", exist_ok=True)
        if os.path.exists(attempt_log_file):
            actions.append(
                LogInfo(
                    msg=f"WARNING: {attempt_log_file} already exists; reusing a "
                    "run_id can mix two runs in one log."
                )
            )
        meta = {
            "tags": tags,
            "overrides": overrides,
            "params_file": params_file,
            "effective_params": effective,
            "launched_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.splitext(attempt_log_file)[0] + ".meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    if tagged:
        actions.append(
            LogInfo(
                msg=f"[experiment] tags={ {k: v for k, v in tags.items() if v} } "
                f"overrides={overrides or 'none'} log={attempt_log_file}"
            )
        )

    # ---- Nodes to Launch ----
    actions.extend([
        # Spearhead Visual Servo Node
        Node(
            package="regrasp_experiment",
            executable="spearhead_servo",
            name=NODE_NAME,
            output="screen",
            emulate_tty=True,
            parameters=[params_file, params],  # YAML first, launch overrides second
        ),
        # Proximity Sensor Node
        Node(
            package='proxymity',
            executable='proxymity_node',
            name='proximity_sensor',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'sensor_type': LaunchConfiguration('sensor_type'),
                'gpio_chip': LaunchConfiguration('gpio_chip'),
                'publish_rate_hz': LaunchConfiguration('proximity_rate'),
                'simulate': LaunchConfiguration('simulate'),
                'frame_id': LaunchConfiguration('frame_id'),
                'obstacle_threshold_cm': LaunchConfiguration('obstacle_threshold_cm'),
                'obstacle_confirm_count': LaunchConfiguration('obstacle_confirm_count'),
                'trigger_pin': LaunchConfiguration('trigger_pin'),
                'echo_pin': LaunchConfiguration('echo_pin'),
                'out_pin': LaunchConfiguration('out_pin'),
            }],
        ),
        # Proximity Controller Node
        Node(
            package='proxymity',
            executable='proxymity_controller_node',
            name='proximity_controller',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'sensor_topic': LaunchConfiguration('proximity_topic'),
                'relative_move_topic': LaunchConfiguration('relative_move_topic'),
                'fsm_command_topic': LaunchConfiguration('fsm_command_topic'),
                'publish_rate_hz': 10.0,
                'fsm_command_val': LaunchConfiguration('fsm_command_val'),
                'initial_relative_y': LaunchConfiguration('initial_relative_y'),
                'forward_relative_x': LaunchConfiguration('forward_relative_x'),
                'forward_duration': LaunchConfiguration('forward_duration'),
                'y_move_interval': LaunchConfiguration('y_move_interval'),
                'rotate_relative_z': LaunchConfiguration('rotate_relative_z'),
                'rotate_duration': LaunchConfiguration('rotate_duration'),
                'flash_area_name': LaunchConfiguration('flash_area_name'),
                'flash_command_val': LaunchConfiguration('flash_command_val'),
                'confirm_duration': LaunchConfiguration('confirm_duration'),
                'confirm_nudge_y': LaunchConfiguration('confirm_nudge_y'),
            }],
        ),
        # Dashboard Node
        Node(
            package="regrasp_experiment",
            executable="spearhead_dashboard",
            name="spearhead_dashboard",
            output="screen",
            condition=IfCondition(LaunchConfiguration("dashboard")),
        )
    ])
    return actions


def generate_launch_description():
    pkg_share = get_package_share_directory("regrasp_experiment")
    default_params = os.path.join(pkg_share, "config", "spearhead_servo.yaml")

    def arg(name, default, description):
        return DeclareLaunchArgument(name, default_value=default, description=description)

    return LaunchDescription(
        [
            # ---- original arguments (unchanged defaults) ----
            arg("params_file", default_params,
                "YAML parameter file for spearhead visual servo."),
            arg("teach", "false",
                "Run in teach mode and stream detections without moving."),
            arg("auto_start", "false",
                "Start the servo sequence automatically on boot."),
            arg("setpoint_file", "/home/heroes/spearhead_setpoint.yaml",
                "Path used to load and save the taught setpoint."),
            arg("teach_show_ui", "true",
                "Show the OpenCV camera preview during teach mode."),
            arg("step_mode", "false",
                "Pause after the defined initial pose until next_step."),
            arg("servo_mode", "bbox_ibvs",
                "Visual servo mode: bbox_ibvs or depth_3d."),
            arg("cmd_vel_topic", "/cmd_vel",
                "Twist command topic owned by this visual servo node."),
            arg("odom_topic", "/teensy_odom_raw",
                "Odometry topic used to hold the base yaw."),
            arg("proximity_topic", "/proximity/obstacle",
                "Bool grip-success sensor topic."),
            arg("attempt_log_file", "",
                "CSV for grasp attempts. Empty = original default, or an "
                "auto-name under log_dir when tags/overrides are set."),
            arg("dashboard", "false",
                "Launch the local Tk operator dashboard (requires a desktop display)."),
            # ---- experiment tags (all optional) ----
            arg("log_dir", "/home/heroes/spearhead_logs",
                "Directory for auto-named per-run logs and *.meta.json files."),
            arg("method", "", "Method name (D1)."),
            arg("condition", "", "Condition label (D1)."),
            arg("run_id", "", "Run index within the method/condition cell (D1)."),
            arg("object_group", "", "seen, unseen or adversarial (D5)."),
            arg("object_id", "", "Object identifier (D5)."),
            arg("clutter_level", "", "c1, c2 or c3 (D6)."),
            # ---- overrides of parameters that already exist in the YAML ----
            arg("yolo_weights", "", "Override yolo_weights, e.g. another detector (D4)."),
            arg("yolo_conf", "", "Override yolo_conf."),
            arg("overrides", "",
                "Extra YAML-parameter overrides for sweeps (D3), 'key=value;key=value'. "
                "Keys must exist in the params file."),
            # ---- Proximity Sensor Arguments ----
            arg('sensor_type', 'l18d80', 'Sensor type: "hcsr04" or "l18d80"'),
            arg('gpio_chip', '/dev/gpiochip0', 'GPIO chip device path'),
            arg('proximity_rate', '30.0', 'Publish rate in Hz for proximity sensor'),
            arg('simulate', 'false', 'Run in simulation mode (no GPIO)'),
            arg('frame_id', 'proximity_link', 'Frame ID for Range message'),
            arg('obstacle_threshold_cm', '10.0', 'Obstacle detection threshold in cm for HC-SR04'),
            arg('obstacle_confirm_count', '7', 'Consecutive True raw reads required to confirm an obstacle'),
            arg('trigger_pin', '11', 'GPIO line for HC-SR04 TRIG pin'),
            arg('echo_pin', '12', 'GPIO line for HC-SR04 ECHO pin'),
            arg('out_pin', '144', 'GPIO line for L18D80 OUT pin'),
            # ---- Proximity Controller Arguments ----
            arg('relative_move_topic', '/relative_move_slow', 'Topic for Vector3 relative move commands'),
            arg('fsm_command_topic', '/fsm_command', 'Topic for FSM integer commands'),
            arg('fsm_command_val', '31', 'FSM command code to send when sensor returns True'),
            arg('initial_relative_y', '-0.5', 'Relative Y move when sensor is clear (strafing)'),
            arg('forward_relative_x', '-1.5', 'Relative X move when sensor detects an obstacle'),
            arg('forward_duration', '3.0', 'Duration in seconds to send forward relative move command'),
            arg('y_move_interval', '0.5', 'Interval in seconds between incremental Y relative move publishes'),
            arg('rotate_relative_z', '180.0', 'Relative Z move to rotate the robot'),
            arg('rotate_duration', '2.0', 'Duration in seconds to send the rotation command'),
            arg('flash_area_name', 'AREA_2', 'Area name sent to /fsm/area_command'),
            arg('flash_command_val', '-1', 'FSM command to publish once green flash is confirmed'),
            arg('confirm_duration', '1.0', 'Duration in seconds sensor must stay True to confirm detection'),
            arg('confirm_nudge_y', '-0.3', 'Relative Y nudge when confirmation fails'),
            OpaqueFunction(function=_launch_setup),
        ]
    )