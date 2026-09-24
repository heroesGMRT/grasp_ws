"""
HC-SR04 Ultrasonic Sensor Driver for Jetson Orin Nano GPIO.

Uses libgpiod v2 for GPIO control via /dev/gpiochip* devices.
Converts pulse width (echo) to distance in centimeters.

Pin reference for Jetson Orin Nano 40-pin header:
  - Pin 9  = GND (Ground for sensor)
  - Pin 11 = GPIO11 (default TRIG)
  - Pin 12 = GPIO12 (default ECHO)
"""

import time

try:
    import gpiod
    HAS_GPIOD = True
except ImportError:
    HAS_GPIOD = False
    gpiod = None


class HCSR04Driver:
    """Driver for HC-SR04 ultrasonic distance sensor using libgpiod."""

    SOUND_SPEED_CM_US = 0.0343   # cm/microsecond

    def __init__(self, trigger_line: int, echo_line: int,
                 gpio_chip: str = '/dev/gpiochip0', timeout_ms: int = 30):
        """
        Args:
            trigger_line: GPIO line offset for TRIG pin.
            echo_line:    GPIO line offset for ECHO pin.
            gpio_chip:    GPIO chip device path (default: /dev/gpiochip0).
            timeout_ms:   Timeout for waiting echo pulse (default: 30ms = ~50cm max).
        """
        if not HAS_GPIOD:
            raise ImportError(
                "Python 'gpiod' bindings not found. "
                "Install via: pip3 install gpiod"
            )

        self._trigger_line = trigger_line
        self._echo_line = echo_line
        self._timeout_ns = timeout_ms * 1_000_000
        self._chip = gpiod.Chip(gpio_chip)

        # Request trigger as output (initially low)
        self._trigger = self._chip.request_lines(
            consumer='proxymity_hcsr04',
            config={
                trigger_line: gpiod.LineSettings(
                    direction=gpiod.line.Direction.OUTPUT,
                    output_value=gpiod.line.Value.INACTIVE,
                ),
            },
        )

        # Request echo as input with rising+falling edge detection
        self._echo = self._chip.request_lines(
            consumer='proxymity_hcsr04',
            config={
                echo_line: gpiod.LineSettings(
                    direction=gpiod.line.Direction.INPUT,
                    edge_detection=gpiod.line.Edge.BOTH,
                ),
            },
        )

        self._trigger_handle = self._trigger  # line request for trigger
        self._echo_handle = self._echo        # line request for echo

    def _send_trigger_pulse(self) -> None:
        """Send a 10 microsecond HIGH pulse on the TRIG pin."""
        offset = self._trigger_line
        self._trigger.set_value(offset, gpiod.line.Value.ACTIVE)
        time.sleep(0.000010)  # 10 µs
        self._trigger.set_value(offset, gpiod.line.Value.INACTIVE)

    def _measure_echo_pulse(self) -> float:
        """
        Measure the duration (in microseconds) of the ECHO HIGH pulse.

        Returns:
            Pulse width in microseconds. Returns 0 on timeout.
        """
        offset = self._echo_line
        timeout_sec = self._timeout_ns / 1_000_000_000.0  # Convert nanoseconds to seconds

        # 1. Wait for rising edge
        if not self._echo.wait_edge_events(timeout=timeout_sec):
            return 0.0  # Timeout waiting for rising edge

        rising_ns = None
        for event in self._echo.read_edge_events():
            if event.line_offset == offset and event.event_type == gpiod.EdgeEvent.Type.RISING_EDGE:
                rising_ns = event.timestamp_ns
                break

        if rising_ns is None:
            return 0.0  # Did not get rising edge

        # 2. Wait for falling edge
        if not self._echo.wait_edge_events(timeout=timeout_sec):
            return 0.0  # Timeout waiting for falling edge

        falling_ns = None
        for event in self._echo.read_edge_events():
            if event.line_offset == offset and event.event_type == gpiod.EdgeEvent.Type.FALLING_EDGE:
                falling_ns = event.timestamp_ns
                break

        if falling_ns is None:
            return 0.0  # Did not get falling edge

        pulse_width_us = (falling_ns - rising_ns) / 1000.0
        return pulse_width_us

    def measure(self) -> float:
        """
        Perform a distance measurement.

        Returns:
            Distance in centimeters. Returns -1.0 on measurement failure.
        """
        self._send_trigger_pulse()
        pulse_us = self._measure_echo_pulse()

        if pulse_us <= 0:
            return -1.0

        # Distance = (pulse time * speed of sound) / 2 (round-trip)
        distance_cm = (pulse_us * self.SOUND_SPEED_CM_US) / 2.0
        return round(distance_cm, 2)

    def cleanup(self) -> None:
        """Release GPIO lines."""
        self._trigger.release()
        self._echo.release()
        self._chip.close()


class HCSR04DriverSim:
    """
    Simulated HC-SR04 driver for development/testing when not on actual hardware.

    Returns simulated distance values that oscillate between 10-80 cm.
    """

    def __init__(self, *args, **kwargs):
        self._t = 0.0
        self._sim_speed = 30.0  # oscillation speed
        print("[HCSR04DriverSim] SIMULATION MODE — not using real GPIO")

    def measure(self) -> float:
        import math
        self._t += 0.1
        # Oscillate between 10cm and 80cm
        dist = 45.0 + 35.0 * math.sin(self._t / self._sim_speed * 2 * math.pi)
        return round(dist, 2)

    def cleanup(self):
        print("[HCSR04DriverSim] cleanup — nothing to do")
