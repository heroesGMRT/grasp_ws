import unittest
from pathlib import Path


class DashboardSourceTest(unittest.TestCase):
    def test_dashboard_does_not_overwrite_rclpy_node_clients_attribute(self):
        dashboard_path = (
            Path(__file__).parents[1]
            / "regrasp_experiment"
            / "spearhead_dashboard.py"
        )
        source = dashboard_path.read_text(encoding="utf-8")

        self.assertNotIn("self.clients =", source)
        self.assertIn("self.service_clients =", source)
