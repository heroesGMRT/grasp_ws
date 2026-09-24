"""
proxymity_node — ROS 2 node for proximity sensors.

Supported sensors (set via `sensor_type` param):
  - "hcsr04"  : HC-SR04 ultrasonic distance sensor (TRIG + ECHO)
  - "l18d80"  : L18D80 IR obstacle detection sensor (OUT)

Publishes:
  /proximity/range     (sensor_msgs/Range) — distance / obstacle presence
  /proximity/raw       (std_msgs/Float32)  — raw value (cm for HCSR04, 0/1 for L18D80)
  /proximity/obstacle  (std_msgs/Bool)     — obstacle flag (L18D80 only)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Range
from std_msgs.msg import Bool, Float32

from .hcsr04_driver import HCSR04Driver, HCSR04DriverSim
from .l18d80_driver import L18D80Driver, L18D80DriverSim


class ProximitySensorNode(Node):
    """ROS 2 node that reads a proximity sensor and publishes range data."""

    def __init__(self):
        super().__init__('proximity_sensor_node')

        # ---------- Common Parameters ----------
        self.declare_parameter('sensor_type', 'l18d80')
        self.declare_parameter('gpio_chip', '/dev/gpiochip0')
        self.declare_parameter('publish_rate_hz', 30.0)
        self.declare_parameter('simulate', False)
        self.declare_parameter('frame_id', 'proximity_link')
        self.declare_parameter('radiation_type', Range.ULTRASOUND)
        self.declare_parameter('field_of_view', 0.261799)
        self.declare_parameter('min_range', 0.02)
        self.declare_parameter('max_range', 4.0)
        self.declare_parameter('obstacle_threshold_cm', 20.0)
        self.declare_parameter('obstacle_on_delay', 0.0)
        self.declare_parameter('obstacle_off_delay', 0.0)
        self.declare_parameter('oscillation_window', 2.0)
        self.declare_parameter('oscillation_max_toggles', 4)
        self.declare_parameter('obstacle_confirm_count', 7)

        # ---------- HC-SR04 Parameters ----------
        self.declare_parameter('trigger_pin', 11)
        self.declare_parameter('echo_pin', 12)

        # ---------- L18D80 Parameters ----------
        self.declare_parameter('out_pin', 144)
        self.declare_parameter('active_low', True)

        sensor_type = self.get_parameter('sensor_type').value
        gpio_chip = self.get_parameter('gpio_chip').value
        rate = self.get_parameter('publish_rate_hz').value
        self._simulate = self.get_parameter('simulate').value

        # Range message static fields
        self._frame_id = self.get_parameter('frame_id').value
        self._radiation_type = self.get_parameter('radiation_type').value
        self._fov = self.get_parameter('field_of_view').value
        self._min_range = self.get_parameter('min_range').value
        self._max_range = self.get_parameter('max_range').value
        self._obstacle_threshold_cm = self.get_parameter('obstacle_threshold_cm').value
        self._obstacle_on_delay = self.get_parameter('obstacle_on_delay').value
        self._obstacle_off_delay = self.get_parameter('obstacle_off_delay').value
        self._oscillation_window = self.get_parameter('oscillation_window').value
        self._oscillation_max_toggles = self.get_parameter('oscillation_max_toggles').value
        self._obstacle_confirm_count = self.get_parameter('obstacle_confirm_count').value

        # ---------- Obstacle state filtering (consecutive-sample debounce) ----------
        self._confirmed_state = False
        self._true_streak = 0

        # ---------- Init driver based on sensor type ----------
        self._sensor_type = sensor_type
        self._init_driver(sensor_type, gpio_chip)

        # ---------- Publishers ----------
        self._pub_range = self.create_publisher(Range, '/proximity/range', 10)
        self._pub_raw = self.create_publisher(Float32, '/proximity/raw', 10)
        self._pub_obstacle = self.create_publisher(Bool, '/proximity/obstacle', 10)

        # ---------- Timer ----------
        period = 1.0 / rate
        self._timer = self.create_timer(period, self._timer_callback)

        self.get_logger().info(
            f'ProximitySensorNode started — sensor={sensor_type}, '
            f'publishing to /proximity/range at {rate} Hz'
        )

    def _init_driver(self, sensor_type: str, gpio_chip: str) -> None:
        """Initialize the appropriate sensor driver based on type."""
        self._driver = None

        if sensor_type == 'hcsr04':
            trig_pin = self.get_parameter('trigger_pin').value
            echo_pin = self.get_parameter('echo_pin').value

            self.get_logger().info(
                f'HC-SR04 configured: TRIG={trig_pin}, ECHO={echo_pin}, chip={gpio_chip}'
            )

            if self._simulate or self._detect_simulation():
                self._driver = HCSR04DriverSim()
                self.get_logger().warn('Running HC-SR04 in SIMULATION mode')
            else:
                try:
                    self._driver = HCSR04Driver(
                        trigger_line=trig_pin,
                        echo_line=echo_pin,
                        gpio_chip=gpio_chip,
                    )
                    self.get_logger().info('HC-SR04 driver initialized on real GPIO')
                except Exception as e:
                    self.get_logger().error(f'Failed to init HC-SR04 GPIO: {e}')
                    self.get_logger().warn('Falling back to SIMULATION mode')
                    self._driver = HCSR04DriverSim()

        elif sensor_type == 'l18d80':
            out_pin = self.get_parameter('out_pin').value
            active_low = self.get_parameter('active_low').value

            self.get_logger().info(
                f'L18D80 configured: OUT={out_pin}, active_low={active_low}, chip={gpio_chip}'
            )

            if self._simulate or self._detect_simulation():
                self._driver = L18D80DriverSim()
                self.get_logger().warn('Running L18D80 in SIMULATION mode')
            else:
                try:
                    self._driver = L18D80Driver(
                        out_pin=out_pin,
                        gpio_chip=gpio_chip,
                        active_low=active_low,
                    )
                    self.get_logger().info('L18D80 driver initialized on real GPIO')
                except Exception as e:
                    self.get_logger().error(f'Failed to init L18D80 GPIO: {e}')
                    self.get_logger().warn('Falling back to SIMULATION mode')
                    self._driver = L18D80DriverSim()

        else:
            self.get_logger().error(f'Unknown sensor_type: {sensor_type}')
            raise ValueError(f'Unsupported sensor_type: {sensor_type}')

    def _detect_simulation(self) -> bool:
        """Detect if we're on actual Jetson hardware."""
        try:
            model = open('/proc/device-tree/model', 'r').read().strip()
            if 'jetson' in model.lower() or 'tegra' in model.lower():
                return False
            return True
        except Exception:
            return True

    def _timer_callback(self) -> None:
        """Read sensor and publish."""
        now = self.get_clock().now()

        if self._sensor_type == 'hcsr04':
            self._publish_hcsr04(now)
        elif self._sensor_type == 'l18d80':
            self._publish_l18d80(now)

    def _publish_hcsr04(self, now) -> None:
        """Publish HC-SR04 distance data."""
        distance_cm = self._driver.measure()
        distance_m = distance_cm / 100.0 if distance_cm > 0 else float('inf')

        # Obstacle Bool
        raw_obstacle = 0.0 < distance_cm <= self._obstacle_threshold_cm
        obstacle = self._update_obstacle_state(raw_obstacle, now)
        obs_msg = Bool()
        obs_msg.data = obstacle
        self._pub_obstacle.publish(obs_msg)

        # Range message
        msg = Range()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._frame_id
        msg.radiation_type = self._radiation_type
        msg.field_of_view = self._fov
        msg.min_range = self._min_range
        msg.max_range = self._max_range
        msg.range = distance_m if 0 <= distance_m <= self._max_range else float('inf')
        self._pub_range.publish(msg)

        # Raw Float32
        raw_msg = Float32()
        raw_msg.data = distance_cm if distance_cm > 0 else -1.0
        self._pub_raw.publish(raw_msg)

        self.get_logger().debug(
            f'HC-SR04: {distance_cm:.1f} cm ({distance_m:.2f} m) — obstacle={"YES" if obstacle else "NO"}',
            throttle_duration_sec=1.0,
        )

    def _publish_l18d80(self, now) -> None:
        """Publish L18D80 obstacle data."""
        raw_obstacle = self._driver.is_obstacle_detected()
        obstacle = self._update_obstacle_state(raw_obstacle, now)

        # Obstacle Bool
        if self._pub_obstacle:
            obs_msg = Bool()
            obs_msg.data = obstacle
            self._pub_obstacle.publish(obs_msg)

        # Range message (0.0 = obstacle, inf = clear)
        msg = Range()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = self._frame_id
        msg.radiation_type = Range.INFRARED
        msg.field_of_view = self._fov
        msg.min_range = self._min_range
        msg.max_range = self._max_range
        msg.range = 0.0 if obstacle else float('inf')
        self._pub_range.publish(msg)

        # Raw Float32 (1.0 = obstacle, 0.0 = clear)
        raw_msg = Float32()
        raw_msg.data = 1.0 if obstacle else 0.0
        self._pub_raw.publish(raw_msg)

        self.get_logger().debug(
            f'L18D80: obstacle={"YES" if obstacle else "NO"}',
            throttle_duration_sec=1.0,
        )

    def _update_obstacle_state(self, raw_obstacle: bool, now) -> bool:
        """
        Require obstacle_confirm_count consecutive True raw reads before
        reporting True. Any False read immediately resets to False.
        """
        if raw_obstacle:
            self._true_streak += 1
            if self._true_streak >= self._obstacle_confirm_count:
                self._confirmed_state = True
        else:
            self._true_streak = 0
            self._confirmed_state = False

        return self._confirmed_state

    def destroy_node(self) -> bool:
        if self._driver:
            self._driver.cleanup()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ProximitySensorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down proximity sensor node')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()