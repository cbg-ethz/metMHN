import metmhn.Utilityfunctions as utils
from metmhn.model import MetMHN
from metmhn.state import State
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
                if first_obs == "sync":
                    with self.assertWarns(Warning):
                        with self.assertRaises(ValueError):
                            self.metMHN.likeliest_order(
                                unreachable_state,
                                met_status=met_status,
                                first_obs=first_obs,)
                else:
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

    @unittest.skip("Not implemented yet")
    def test_likelihood(self):
        """Test that the likelihoods are calculated correctly"""
        unpaired_diag = self.metMHN._get_diag_unpaired(state=State([1] * 5))
        test_cases = [
            # Test with PT events
            ((0, 3, 2), "absent", None,
             np.exp(self.log_theta[0, 0]) \
             / (1 - unpaired_diag[0]) \
             * np.exp(self.log_theta[3, [0, 3]].sum()) \
             / (np.exp(self.obs1[0]) - unpaired_diag[2 ** 0]) \
             * np.exp(self.log_theta[2, [0, 3, 2]].sum()) \
             / (np.exp(self.obs1[[0, 3]].sum()) - unpaired_diag[2 ** 0 + 2 ** 3]) \
             * np.exp(self.obs1[[0, 2, 3]].sum()) \
             / (np.exp(self.obs1[[0, 2, 3]].sum()) - unpaired_diag[2 ** 0 + 2 ** 2 + 2 ** 3]),
             )]

        for order, met_status, first_obs, expected_likelihood in test_cases:
            with self.subTest(order=order, met_status=met_status, first_obs=first_obs):
                likelihood = self.metMHN.likelihood(
                    order,
                    met_status=met_status,
                    first_obs=first_obs,)
                self.assertAlmostEqual(
                    likelihood, expected_likelihood, places=5)


if __name__ == "__main__":
    unittest.main()
