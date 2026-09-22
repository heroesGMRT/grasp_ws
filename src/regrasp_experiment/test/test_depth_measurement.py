import numpy as np
import unittest

from regrasp_experiment.depth_measurement import measure_bbox_depth


class DepthMeasurementTest(unittest.TestCase):
    def test_measure_bbox_depth_uses_inner_half_median_and_projects_center(self):
        depth = np.zeros((10, 10), dtype=np.uint16)
        depth[4:6, 4:6] = np.array([[1000, 1100], [1150, 5000]], dtype=np.uint16)

        result = measure_bbox_depth(
            depth,
            bbox=(2, 2, 6, 6),
            intrinsics=(100.0, 100.0, 5.0, 5.0),
            scale_to_m=0.001,
            depth_min_m=0.10,
            depth_max_m=1.20,
            min_valid_samples=3,
        )

        self.assertIsNotNone(result)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 1.1)

    def test_measure_bbox_depth_rejects_out_of_range_samples(self):
        depth = np.full((8, 8), 50, dtype=np.uint16)

        self.assertIsNone(measure_bbox_depth(
            depth,
            bbox=(1, 1, 6, 6),
            intrinsics=(100.0, 100.0, 4.0, 4.0),
            scale_to_m=0.001,
            depth_min_m=0.10,
            depth_max_m=1.20,
            min_valid_samples=2,
        ))
