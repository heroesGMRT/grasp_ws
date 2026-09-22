"""ROS-independent command calculations for the visual-servo controller."""

from dataclasses import dataclass
from math import atan2, cos, hypot, sin


@dataclass(frozen=True)
class VelocityCommand:
    """Body-frame forward, strafe, and yaw velocity commands."""

    x_mps: float
    y_mps: float
    yaw_radps: float


def wrap_angle(angle_rad: float) -> float:
    """Return an angle in [-pi, pi] using the shortest representation."""
    return atan2(sin(angle_rad), cos(angle_rad))


def heading_command(
    target_yaw_rad: float,
    current_yaw_rad: float,
    gain: float,
    max_yaw_radps: float,
) -> float:
    """Return a bounded yaw-rate command toward the held heading."""
    error = wrap_angle(target_yaw_rad - current_yaw_rad)
    return max(-max_yaw_radps, min(max_yaw_radps, gain * error))


def depth_standoff_command(
    measured_depth_m: float,
    target_depth_m: float,
    overshoot_depth_m: float,
    gain_mps_per_m: float,
    max_forward_mps: float,
    fwd_from_z_sign: float,
) -> tuple[VelocityCommand, bool]:
    """Return forward correction and whether it is a mandatory backup.

    ``fwd_from_z_sign`` converts camera-depth error into the robot's forward
    command convention. A target closer than the overshoot threshold bypasses
    the normal tolerance logic and is always reported as backup-only.
    """
    error_m = measured_depth_m - target_depth_m
    forward_mps = fwd_from_z_sign * gain_mps_per_m * error_m
    forward_mps = max(-max_forward_mps, min(max_forward_mps, forward_mps))
    if measured_depth_m < overshoot_depth_m:
        return VelocityCommand(forward_mps, 0.0, 0.0), True
    return VelocityCommand(forward_mps, 0.0, 0.0), False


def return_to_start_command(
    start_x_m: float,
    start_y_m: float,
    start_yaw_rad: float,
    current_x_m: float,
    current_y_m: float,
    current_yaw_rad: float,
    linear_gain: float,
    max_linear_mps: float,
    yaw_gain: float,
    max_yaw_radps: float,
) -> VelocityCommand:
    """Return a bounded body-frame command toward an odometry start pose."""
    dx_world = start_x_m - current_x_m
    dy_world = start_y_m - current_y_m
    cos_yaw = cos(current_yaw_rad)
    sin_yaw = sin(current_yaw_rad)
    x_body = cos_yaw * dx_world + sin_yaw * dy_world
    y_body = -sin_yaw * dx_world + cos_yaw * dy_world
    x_body *= linear_gain
    y_body *= linear_gain
    speed = hypot(x_body, y_body)
    if speed > max_linear_mps:
        scale = max_linear_mps / speed
        x_body *= scale
        y_body *= scale
    return VelocityCommand(
        x_mps=x_body,
        y_mps=y_body,
        yaw_radps=heading_command(
            start_yaw_rad, current_yaw_rad, yaw_gain, max_yaw_radps
        ),
    )
