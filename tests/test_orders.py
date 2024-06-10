import metmhn.Utilityfunctions as utils
from metmhn.model import MetMHN
from metmhn.state import State, MetState
import numpy as np
import unittest


class LikelihoodTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        self.log_theta = utils.random_theta(self.n, 0.2)
        self.obs1 = 2 * np.random.random(self.n + 1) + 1
        self.obs2 = 2 * np.random.random(self.n + 1) + 1
        self.metMHN = MetMHN(self.log_theta, self.obs1, self.obs2)

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

    @unittest.expectedFailure
    def test_unpaired_likelihood(self):
        """Test that the likelihoods for unpaired orders are calculated correctly"""

        seeding = self.n * 2

        unpaired_diag_seeding = self.metMHN._get_diag_unpaired(
            state=State([1] * 5))
        unpaired_diag = self.metMHN._get_diag_unpaired(
            state=State([1] * 5), seeding=False)
        test_cases = [
            ((0, 6, 4), "absent", None,
             np.exp(self.log_theta[0, 0])
             / (1 - unpaired_diag_seeding[0])
             * np.exp(self.log_theta[3, [0, 3]].sum())
             / (np.exp(self.obs1[0]) - unpaired_diag_seeding[2 ** 0])
             * np.exp(self.log_theta[2, [0, 3, 2]].sum())
             / (np.exp(self.obs1[[0, 3]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 3])
             * np.exp(self.obs1[[0, 2, 3]].sum())
             / (np.exp(self.obs1[[0, 2, 3]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 2 + 2 ** 3]),
             ),
            ((0, 3, seeding, 2), "present", None,
             np.exp(self.log_theta[0, 0])
             / (1 - unpaired_diag_seeding[0])
             * np.exp(self.log_theta[3, [0, 3]].sum())
             / (np.exp(self.obs1[0]) - unpaired_diag_seeding[2 ** 0])
             * np.exp(self.log_theta[self.n, [self.n, 0, 3]].sum())
             / (np.exp(self.obs1[[0, 3]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 3]),
             * np.exp(self.log_theta[2, [0, 3, 2]].sum())
             / (np.exp(self.obs1[[0, 3]].sum())
                - unpaired_diag[2 ** 0 + 2 ** 3])
             * np.exp(self.obs1[[0, 2, 3]].sum())
             / (np.exp(self.obs1[[0, 2, 3]].sum())
                - unpaired_diag[2 ** 0 + 2 ** 2 + 2 ** 3]),
             ),
            ((0, 3, seeding, 2), "isMetastasis", None,
             np.exp(self.log_theta[0, 0])
             / (1 - unpaired_diag_seeding[0])
             * np.exp(self.log_theta[3, [0, 3]].sum())
             / (np.exp(self.obs1[0]) - unpaired_diag_seeding[2 ** 0])
             * np.exp(self.log_theta[self.n, [self.n, 0, 3]].sum())
             / (np.exp(self.obs1[[0, 3]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 3]),
             * np.exp(self.log_theta[2, [0, 3, 2, self.n]].sum())
             / (np.exp(self.obs2[[0, 3, self.n]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 3 + 2 ** self.n])
             * np.exp(self.obs2[[0, 2, 3, self.n]].sum())
             / (np.exp(self.obs2[[0, 2, 3, self.n]].sum())
                - unpaired_diag_seeding[
                    2 ** 0 + 2 ** 2 + 2 ** 3 + 2 ** self.n]),
             )
        ]
        for order, met_status, first_obs, expected_likelihood in test_cases:
            with self.subTest(
                    order=order, met_status=met_status, first_obs=first_obs):
                likelihood = self.metMHN.likelihood(
                    order,
                    met_status=met_status,
                    first_obs=first_obs,)
                np_assert_approx_equal(
                    likelihood, expected_likelihood, places=5)

    def test_paired_timed_likelihood(self):
        """Test that the likelihoods for paired, timed orders are calculated correctly"""
        seeding = self.n * 2

        unpaired_diag_seeding = self.metMHN._get_diag_unpaired(
            state=State.from_seq([1] * 5))
        unpaired_diag = self.metMHN._get_diag_unpaired(
            state=State.from_seq([1] * 5), seeding=False)
        paired_diag = self.metMHN._get_diag_paired(
            state=MetState.from_seq([1] * 11))
        test_cases = [
            ((0, 1, seeding, 4), (3, 5), self.metMHN._likelihood_pt_mt_timed,
             np.exp(self.log_theta[0, 0])
             / (1 - paired_diag[0])
             * np.exp(self.log_theta[self.n, [0, self.n]].sum())
             / (np.exp(self.obs1[0]) - paired_diag[2 ** 0 + 2 ** 1])
             * np.exp(self.log_theta[2, [0, 2]].sum())
             / (np.exp(self.obs1[[0]].sum())
                + np.exp(self.obs2[[0, self.n]].sum())
                - paired_diag[2 ** 0 + 1 ** 3 + 2 ** seeding])
             * np.exp(self.obs1[[0, 2]].sum())
             / (np.exp(self.obs1[[0, 2]].sum())
                + np.exp(self.obs2[[0, self.n]].sum())
                - paired_diag[2 ** 0 + 2 ** 1 + 2 ** 4 + 2 ** seeding])
             * np.exp(self.log_theta[1, [0, 1, self.n]].sum())
             / (np.exp(self.obs2[[0, self.n]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** self.n])
             * np.exp(self.log_theta[2, [0, 1, 2, self.n]].sum())
             / (np.exp(self.obs2[[0, 1, self.n]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 1 + 2 ** self.n])
             * np.exp(self.obs2[[0, 1, 2, self.n]].sum())
             / (np.exp(self.obs2[[0, 1, 2, self.n]].sum())
                - unpaired_diag_seeding[
                    2 ** 0 + 2 ** 1 + 2 ** 2 + 2 ** self.n]),
             ),
            ((0, 1, seeding, 5), (6, 2), self.metMHN._likelihood_mt_pt_timed,
             np.exp(self.log_theta[0, 0])
             / (1 - paired_diag[0])
             * np.exp(self.log_theta[self.n, [0, self.n]].sum())
             / (np.exp(self.obs1[0]) - paired_diag[2 ** 0 + 2 ** 1])
             * np.exp(self.log_theta[2, [0, 2, self.n]].sum())
             / (np.exp(self.obs1[[0]].sum())
                + np.exp(self.obs2[[0, self.n]].sum())
                - paired_diag[2 ** 0 + 1 ** 3 + 2 ** seeding])
             * np.exp(self.obs2[[0, 2, self.n]].sum())
             / (np.exp(self.obs1[[0]].sum())
                + np.exp(self.obs2[[0, 2, self.n]].sum())
                - paired_diag[2 ** 0 + 2 ** 1 + 2 ** 5 + 2 ** seeding])
             * np.exp(self.log_theta[3, [0, 3]].sum())
             / (np.exp(self.obs1[[0]].sum()) - unpaired_diag[2 ** 0])
             * np.exp(self.log_theta[1, [0, 1, 3]].sum())
             / (np.exp(self.obs2[[0, 3]].sum())
                - unpaired_diag[2 ** 0 + 2 ** 3])
             * np.exp(self.obs2[[0, 1, 3]].sum())
             / (np.exp(self.obs2[[0, 1, 3]].sum())
                - unpaired_diag_seeding[2 ** 0 + 2 ** 1 + 2 ** 3]),
             ),
        ]
        for order1, order2, func, expected_likelihood in test_cases:
            with self.subTest(order1=order1, order2=order2, func=func):
                likelihood = func(
                    order1,
                    order2,)
                np_assert_approx_equal(
                    likelihood, expected_likelihood, places=5)
                
    def test_paired_likelihood(self):
        """Test that the likelihoods for paired orders are calculated correctly"""
        
        seeding = self.n * 2
        test_cases = [
            MetState([0, 1, 3, 5, 6])
        ]




if __name__ == "__main__":
    unittest.main()
