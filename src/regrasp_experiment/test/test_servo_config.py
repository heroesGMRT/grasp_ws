import unittest
from pathlib import Path


class ServoConfigTest(unittest.TestCase):
    def test_cmd_vel_and_depth_safety_defaults(self):
        config_path = Path(__file__).parents[1] / "config" / "spearhead_servo.yaml"
        text = config_path.read_text(encoding="utf-8")

        self.assertIn('cmd_vel_topic: "/cmd_vel"', text)
        self.assertIn('odom_topic: "/teensy_odom_raw"', text)
        self.assertIn("bbox_target_depth_m: 0.21", text)
        self.assertIn("bbox_overshoot_depth_m: 0.20", text)
        self.assertIn("depth_min_m: 0.10", text)
        self.assertIn("max_yaw_radps: 0.40", text)
        self.assertIn('proximity_topic: "/proximity/obstacle"', text)
        self.assertIn('grip_outcome_timeout_s: 4.0', text)
        self.assertIn('grip_outcome_topic: "/spearhead_servo/grip_success"', text)
