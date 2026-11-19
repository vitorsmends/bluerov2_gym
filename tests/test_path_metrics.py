import unittest
import numpy as np
from bluerov2_gym.workflows.evaluation.path_metrics import geometric_path_distance,time_within_tolerance


class PathMetricTests(unittest.TestCase):
    def test_circle_uses_vertical_and_horizontal_distance(self):
        points=np.array([[1,0,-5],[1,0,-4.7],[0,0,-5]])
        np.testing.assert_allclose(geometric_path_distance(points,'circle',{'radius':1,'depth':-5}),[0,.3,1],atol=1e-12)

    def test_polygon_distance_uses_segments_not_nearest_vertex(self):
        points=np.array([[0,1,-5],[0,1.3,-5],[2,2,-5]])
        np.testing.assert_allclose(geometric_path_distance(points,'square',{'side':2,'depth':-5}),[0,.3,np.sqrt(2)],atol=1e-12)

    def test_fraction_does_not_reward_stopping_on_the_path_as_tracking(self):
        self.assertEqual(time_within_tolerance([0,.1,.2,.3],.2),75)
        # On-path location can still be far from a time-indexed reference.
        self.assertEqual(time_within_tolerance([0,0],.2),100)
        self.assertEqual(time_within_tolerance([1,1],.2),0)
