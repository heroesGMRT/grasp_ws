from math import pi
import unittest

from regrasp_experiment.motion_control import (
    VelocityCommand,
    depth_standoff_command,
    heading_command,
    return_to_start_command,
    wrap_angle,
)


class MotionControlTest(unittest.TestCase):
    def test_heading_command_uses_shortest_wrapped_yaw_error(self):
        self.assertAlmostEqual(wrap_angle(-2.0 * pi), 0.0)
        self.assertAlmostEqual(
            heading_command(pi - 0.05, -pi + 0.05, gain=2.0, max_yaw_radps=0.5),
            -0.2,
        )

    def test_depth_below_overshoot_threshold_commands_backup_only(self):
        command, is_backup = depth_standoff_command(
            measured_depth_m=0.19,
            target_depth_m=0.21,
            overshoot_depth_m=0.20,
            gain_mps_per_m=0.75,
            max_forward_mps=0.10,
            fwd_from_z_sign=-1.0,
        )

        self.assertTrue(is_backup)
        self.assertAlmostEqual(command.x_mps, 0.015)
        self.assertEqual((command.y_mps, command.yaw_radps), (0.0, 0.0))

    def test_depth_at_target_requires_no_forward_motion(self):
        command, is_backup = depth_standoff_command(
            measured_depth_m=0.21,
            target_depth_m=0.21,
            overshoot_depth_m=0.20,
            gain_mps_per_m=0.75,
            max_forward_mps=0.10,
            fwd_from_z_sign=-1.0,
        )

        self.assertFalse(is_backup)
        self.assertEqual(command, VelocityCommand(x_mps=0.0, y_mps=0.0, yaw_radps=0.0))

    def test_return_to_start_rotates_world_error_into_body_frame(self):
        command = return_to_start_command(
            start_x_m=1.0,
            start_y_m=0.0,
            start_yaw_rad=0.0,
            current_x_m=0.0,
            current_y_m=0.0,
            current_yaw_rad=pi / 2,
            linear_gain=1.0,
            max_linear_mps=0.5,
            yaw_gain=1.0,
            max_yaw_radps=0.4,
        )

        self.assertAlmostEqual(command.x_mps, 0.0, places=6)
        self.assertAlmostEqual(command.y_mps, -0.5, places=6)
        self.assertAlmostEqual(command.yaw_radps, -0.4, places=6)

    def test_return_to_start_never_exceeds_combined_linear_speed_limit(self):
        command = return_to_start_command(
            start_x_m=1.0,
            start_y_m=1.0,
            start_yaw_rad=0.0,
            current_x_m=0.0,
            current_y_m=0.0,
            current_yaw_rad=0.0,
            linear_gain=0.8,
            max_linear_mps=0.08,
            yaw_gain=1.0,
            max_yaw_radps=0.4,
        )

        self.assertLessEqual((command.x_mps**2 + command.y_mps**2) ** 0.5, 0.08)
