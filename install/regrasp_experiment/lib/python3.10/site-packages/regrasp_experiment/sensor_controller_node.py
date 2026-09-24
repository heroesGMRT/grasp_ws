"""
sensor_controller_node — ROS 2 node that controls robot movement based on proximity sensor state.

Behavior:
  - Subscribes to `/proximity/obstacle` (std_msgs/Bool).
  - When the node starts, the robot lowers spearhead (FSM 32) and arms the sensor after 5.0s.
  - Once armed, if the sensor is False (no obstacle), the robot moves to the right using `/relative_move_slow` (y: initial_relative_y), stepping every y_move_interval seconds.
  - When the sensor returns True (obstacle detected):
    - Stops Y movement and enters a CONFIRMING state.
    - If the sensor stays True for confirm_duration seconds, proceeds to MOVING_FORWARD.
    - If the sensor goes False during confirmation, nudges back right (confirm_nudge_y) and returns to MOVING_Y.
  - In MOVING_FORWARD:
    - Publishes a relative move command (x: forward_relative_x) to `/relative_move_slow`.
    - Waits for forward_duration.
    - Publishes FSM command to `/fsm_command` (std_msgs/Int32) once.
    - Shuts down.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Int32
from geometry_msgs.msg import Vector3


class ProximityControllerNode(Node):
    """ROS 2 node that controls robot movement based on proximity sensor state."""

    def __init__(self):
        super().__init__('proximity_controller_node')

        # ---------- Parameters ----------
        self.declare_parameter('sensor_topic', '/proximity/obstacle')
        self.declare_parameter('relative_move_topic', '/relative_move_slow')
        self.declare_parameter('fsm_command_topic', '/fsm_command')

        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('fsm_command_val', 31)

        # Relative move distance parameters
        self.declare_parameter('initial_relative_y', -0.5)
        self.declare_parameter('forward_relative_x', -1.5)
        self.declare_parameter('forward_duration', 2.0)
        self.declare_parameter('y_move_interval', 0.5)

        # Confirmation parameters
        self.declare_parameter('confirm_duration', 1.0)
        self.declare_parameter('confirm_nudge_y', -0.3)

        # ---------- Retrieve Parameters ----------
        self.sensor_topic = self.get_parameter('sensor_topic').value
        self.relative_move_topic = self.get_parameter('relative_move_topic').value
        self.fsm_command_topic = self.get_parameter('fsm_command_topic').value

        self.publish_rate_hz = self.get_parameter('publish_rate_hz').value
        self.fsm_command_val = self.get_parameter('fsm_command_val').value

        self.initial_relative_y = self.get_parameter('initial_relative_y').value
        self.forward_relative_x = self.get_parameter('forward_relative_x').value
        self.forward_duration = self.get_parameter('forward_duration').value
        self.y_move_interval = self.get_parameter('y_move_interval').value
        self.confirm_duration = self.get_parameter('confirm_duration').value
        self.confirm_nudge_y = self.get_parameter('confirm_nudge_y').value

        # ---------- State Machine Setup ----------
        self.STATE_START = 'START'
        self.STATE_MOVING_Y = 'MOVING_Y'
        self.STATE_CONFIRMING = 'CONFIRMING'
        self.STATE_MOVING_FORWARD = 'MOVING_FORWARD'
        self.STATE_FINISHED = 'FINISHED'

        self.state = self.STATE_START
        self.last_sensor_value = None
        self.sensor_history = []
        self.forward_start_time = None
        self.moving_y_start_time = None
        self.last_y_publish_time = None
        self.current_y_speed = self.initial_relative_y
        self.confirm_start_time = None

        # Gate: sensor is ignored until FSM 32 has been sent and confirmed
        self.armed = False

        # ---------- Publishers & Subscribers ----------
        self.relative_move_pub = self.create_publisher(Vector3, self.relative_move_topic, 10)
        self.fsm_pub = self.create_publisher(Int32, self.fsm_command_topic, 10)

        self.sensor_sub = self.create_subscription(
            Bool,
            self.sensor_topic,
            self.sensor_callback,
            10
        )

        # ---------- Control Timer ----------
        period = 1.0 / self.publish_rate_hz
        self.timer = self.create_timer(period, self.timer_callback)

        # One-shot timer: publish FSM 32 after 1.0s delay (allowing DDS discovery to settle)
        self.startup_timer = self.create_timer(1.0, self._lower_spearhead)

        self.get_logger().info(
            f"Proximity Controller Node started:\n"
            f"  Subscribed to: '{self.sensor_topic}'\n"
            f"  Publishing Vector3 to: '{self.relative_move_topic}'\n"
            f"  Publishing FSM to: '{self.fsm_command_topic}'\n"
            f"  Initial State: {self.state} (sensor DISARMED)"
        )

    def _lower_spearhead(self) -> None:
        """One-shot callback: publishes startup command 32 to lower spearhead, then starts 5s delay to arm."""
        self.startup_timer.cancel()

        # Publish FSM startup command (32) now that connections are established
        startup_msg = Int32()
        startup_msg.data = 32
        self.fsm_pub.publish(startup_msg)
        self.get_logger().info("Published FSM startup command 32 (lower spearhead). Waiting 5.0s before arming sensor...")

        # Start a 5.0s timer to arm the sensor after lowering is done
        self.arm_timer = self.create_timer(5.0, self._arm_sensor)

    def _arm_sensor(self) -> None:
        """One-shot callback: arms the sensor after 5.0s lowering delay."""
        self.armed = True
        self.arm_timer.cancel()
        self.get_logger().info("Sensor ARMED — now responding to proximity events.")

    def sensor_callback(self, msg: Bool) -> None:
        """Process incoming sensor readings to update internal state transitions."""
        if not self.armed:
            return  # Ignore all sensor readings until armed

        sensor_val = msg.data
        self.last_sensor_value = sensor_val

        if sensor_val is True:
            if self.state in (self.STATE_START, self.STATE_MOVING_Y):
                # Obstacle first detected — stop and enter confirmation period
                self.state = self.STATE_CONFIRMING
                self.confirm_start_time = self.get_clock().now()
                self.moving_y_start_time = None
                self.get_logger().info(
                    f"Sensor -> True! Stopping Y, entering CONFIRMING "
                    f"(holding for {self.confirm_duration}s)."
                )

                # Send block 0 to stop all movement
                stop_msg = Vector3()
                stop_msg.x = 0.0
                stop_msg.y = 0.0
                stop_msg.z = 0.0
                self.relative_move_pub.publish(stop_msg)

            # If already CONFIRMING and sensor stays True, nothing to do —
            # the timer_callback will handle the duration check.

        else:
            if self.state == self.STATE_CONFIRMING:
                # Sensor went False during confirmation — nudge back right and re-search
                self.get_logger().info(
                    f"Sensor -> False during CONFIRMING! "
                    f"Nudging right (y: {self.confirm_nudge_y}) and returning to MOVING_Y."
                )
                self.confirm_start_time = None
                self.state = self.STATE_MOVING_Y
                self.moving_y_start_time = self.get_clock().now()

                # Nudge right
                nudge_msg = Vector3()
                nudge_msg.x = 0.0
                nudge_msg.y = self.confirm_nudge_y
                nudge_msg.z = 0.0
                self.relative_move_pub.publish(nudge_msg)
                self.last_y_publish_time = self.get_clock().now()

            elif self.state == self.STATE_START:
                self.state = self.STATE_MOVING_Y
                self.moving_y_start_time = self.get_clock().now()
                self.get_logger().info(f"Sensor message -> False (clear)! Changing movement to MOVING_Y (via /relative_move_slow y: {self.current_y_speed}).")

                # Send initial relative move
                rel_msg = Vector3()
                rel_msg.x = 0.0
                rel_msg.y = self.current_y_speed
                rel_msg.z = 0.0
                self.relative_move_pub.publish(rel_msg)
                self.last_y_publish_time = self.get_clock().now()

    def timer_callback(self) -> None:
        """Run periodic status check or keepalive."""
        now = self.get_clock().now()

        if self.state == self.STATE_MOVING_Y:
            # Send current Y relative move, but only every y_move_interval seconds
            if (self.last_y_publish_time is None or
                    (now - self.last_y_publish_time).nanoseconds / 1e9 >= self.y_move_interval):
                rel_msg = Vector3()
                rel_msg.x = 0.0
                rel_msg.y = self.current_y_speed
                rel_msg.z = 0.0
                self.relative_move_pub.publish(rel_msg)
                self.last_y_publish_time = now

            if self.moving_y_start_time is not None:
                elapsed = (now - self.moving_y_start_time).nanoseconds / 1e9
                if elapsed >= 15.0:
                    self.get_logger().info("Y movement timeout (15.0s) reached. Publishing FSM command 0 and shutting down.")
                    # Send stop command to ensure we are stopped
                    stop_msg = Vector3()
                    stop_msg.x = 0.0
                    stop_msg.y = 0.0
                    stop_msg.z = 0.0
                    self.relative_move_pub.publish(stop_msg)

                    # Publish FSM command 0
                    fsm_msg = Int32()
                    fsm_msg.data = 0
                    self.fsm_pub.publish(fsm_msg)
                    self.get_logger().info(f"Published FSM command 0 to {self.fsm_command_topic}")
                    self.state = self.STATE_FINISHED
                    raise SystemExit(0)

        elif self.state == self.STATE_CONFIRMING:
            # Check if confirmation period has elapsed with sensor still True
            if self.confirm_start_time is not None:
                elapsed = (now - self.confirm_start_time).nanoseconds / 1e9
                if elapsed >= self.confirm_duration:
                    # Sensor stayed True for the full confirmation period → proceed forward
                    self.get_logger().info(
                        f"CONFIRMING passed ({self.confirm_duration}s held True). "
                        f"Moving forward X: {self.forward_relative_x}."
                    )
                    self.state = self.STATE_MOVING_FORWARD
                    self.forward_start_time = self.get_clock().now()
                    self.confirm_start_time = None

                    # Send forward relative move
                    forward_msg = Vector3()
                    forward_msg.x = self.forward_relative_x
                    forward_msg.y = 0.0
                    forward_msg.z = 0.0
                    self.relative_move_pub.publish(forward_msg)

        elif self.state == self.STATE_MOVING_FORWARD:
            # We already published the forward move once upon entering this state.
            # Now we just wait for the duration to complete.
            elapsed = (now - self.forward_start_time).nanoseconds / 1e9

            if elapsed >= self.forward_duration:
                self.get_logger().info("Forward command duration completed. Publishing FSM command and shutting down.")

                # Publish FSM command
                fsm_msg = Int32()
                fsm_msg.data = self.fsm_command_val
                self.fsm_pub.publish(fsm_msg)
                self.get_logger().info(f"Published FSM command {self.fsm_command_val} to {self.fsm_command_topic}")
                self.state = self.STATE_FINISHED
                raise SystemExit(0)

    def publish_zero_velocity(self) -> None:
        """Publish zero relative move to stop all movement."""
        zero_rel = Vector3()
        self.relative_move_pub.publish(zero_rel)


def main(args=None):
    rclpy.init(args=args)
    node = ProximityControllerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        node.get_logger().info('Shutting down proximity controller node')
    finally:
        # Stop the robot on shutdown
        node.publish_zero_velocity()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()