import likelihood as fss
import numpy as np
import kronecker_vector as kv
import Utilityfunctions as ut
import explicit_statetespace as essp
import unittest

class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 5
        self.theta = ut.random_theta(self.n, 0.4)
        self.Q = essp.build_q(self.theta)
        self.q_diag = np.diag(np.diag(self.Q))
        self.p0 = np.zeros(2**(2*self.n+1))
        self.p0[0] = 1
        self.lam = np.random.exponential(10, 1)
        self.R = self.lam*np.eye(2**(2*self.n +1)) - self.Q

    def test_fss_q_p(self):
        """
        tests Q(theta) p
        """
        self.assertTrue(
            np.allclose(
                self.Q @ self.p0,
                kv.qvec(self.theta, self.p0, True, False)
            )
        )

    def test_fss_q_transp_p(self):
        """
        tests Q(theta)^T p
        """
        self.assertTrue(
            np.allclose(
                self.Q.T @ self.p0,
                kv.qvec(self.theta, self.p0, True, True)
            )
        )

    def test_fss_diag_Q_p(self):
        """
        tests diag(Q(theta)) * p
        """
        self.assertTrue(
            np.allclose(
                self.q_diag @ self.p0,
                (-1) * kv.diag_q(self.theta) * self.p0
            )
        )

    def test_fss_q_no_diag_p(self):
        """
        tests (Q - diag(Q(theta))) p
        """
        self.assertTrue(
            np.allclose(
                (self.Q - self.q_diag) @ self.p0,
                kv.qvec(self.theta, self.p0, False, False)
            )
        )

    def test_fss_resolvent_p(self):
        """
        tests (lambda * I - Q(theta))^(-1) p
        """
        self.assertTrue(
            np.allclose(
                np.linalg.solve(self.R.T, self.p0),
                fss.jacobi(self.theta, self.p0, self.lam, transp=True)
            )
        )

    def test_fss_q_grad_p(self):
        """
        tests q (d Q/d theta) p
        """
        p = fss.jacobi(self.theta, self.p0, self.lam)
        q = fss.jacobi(self.theta, p, self.lam, transp=True)
        theta_test = np.zeros_like(self.theta)
        self.assertTrue(
            np.allclose(
                essp.build_q_grad_p(theta_test, q, p),
                kv.q_partialQ_pth(theta_test, q, p, self.n)
            )
        )


if __name__ == "__main__":
    unittest.main()

