import metmhn.Utilityfunctions as utils
from metmhn.model import MetMHN
import numpy as np
import unittest

class LikelihoodTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        log_theta = utils.random_theta(self.n, 0.2)
        obs1 = 2 * np.random.random(self.n + 1) + 1
        obs2 = 2 * np.random.random(self.n + 1) + 1
        self.metMHN = MetMHN(log_theta, obs1, obs2)

    def test_unreachable(self):
        """Test that there is an error thrown for an unreachable state""" 
        unreachable_state = np.array([0, 1, 1, 1, 0, 0, 0, 1, 0])
        test_cases = [
            ("isPaired", "PT"),
            ("isPaired", "Met"),
            ("isPaired", "unknown"),
            ("isPaired", "sync"),
        ]
        for met_status, first_obs in test_cases:
            with self.subTest(met_status=met_status, first_obs=first_obs):
                with self.assertRaises(ValueError):
                    self.metMHN.likeliest_order(
                        unreachable_state,
                        met_status=met_status,
                        first_obs=first_obs,)
                    
    def test_invalid_states(self):
        """Test that there is an error thrown for invalid states""" 
        test_cases = [
            # Metastasis but PT events present
            ("isMetastasis", np.array([0, 1, 1, 1, 0, 0, 0, 1, 1])),
            # Metastasis but Seeding is absent
            ("isMetastasis", np.array([0, 1, 0, 1, 0, 0, 0, 0, 0])),
            # Present but Met events present
            ("present", np.array([1, 1, 0, 1, 1, 0, 0, 0, 1])),
            # Present but Seeding is absent
            ("present", np.array([1, 0, 0, 0, 1, 0, 0, 0, 0])),
            # Absent but Met events present
            ("absent", np.array([1, 1, 0, 1, 1, 0, 0, 0, 1])),
            # Absent but Seeding is present
            ("absent", np.array([1, 0, 0, 0, 1, 0, 0, 0, 1])),
        ]
        for met_status, invalid_state in test_cases:
            with self.subTest(met_status=met_status, state=invalid_state):
                with self.assertRaises(ValueError):
                    self.metMHN.likeliest_order(
                        invalid_state,
                        met_status=met_status,)
                    
if __name__ == "__main__":
    unittest.main()