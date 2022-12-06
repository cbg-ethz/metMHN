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
        self.lam1 = np.random.exponential(1, 1)
        self.lam2 = np.random.exponential(1, 1)
        self.R = self.lam1*np.eye(2**(2*self.n +1)) - self.Q
        self.pTh1, self.pTh2 = fss.generate_pths(self.theta, self.p0, self.lam1, self.lam2)
        self.pTh = self.lam1 * self.lam2 / (self.lam1 - self.lam2)*(self.pTh2 - self.pTh1)


    def test_fss_q_p(self):
        """
        tests whether implicit and explicit version of Q(theta) p return the same results
        """
        self.assertTrue(
            np.allclose(
                self.Q @ self.p0,
                kv.qvec(self.theta, self.p0, True, False)
            )
        )


    def test_fss_q_transp_p(self):
        """
        tests whether implicit and explicit version of Q(theta)^T p return the same results
        """
        self.assertTrue(
            np.allclose(
                self.Q.T @ self.p0,
                kv.qvec(self.theta, self.p0, True, True)
            )
        )


    def test_fss_diag_Q_p(self):
        """
        tests whether implicit and explicit version of diag(Q(theta)) * p return the same results
        """
        self.assertTrue(
            np.allclose(
                self.q_diag @ self.p0,
                (-1) * kv.diag_q(self.theta) * self.p0
            )
        )


    def test_fss_q_no_diag_p(self):
        """
        tests whether implicit and explicit verion of (Q - diag(Q(theta))) p return the same results
        """
        self.assertTrue(
            np.allclose(
                (self.Q - self.q_diag) @ self.p0,
                kv.qvec(self.theta, self.p0, False, False)
            )
        )


    def test_fss_resolvent_p(self):
        """
        tests whether implicit and explicit version of (lambda * I - Q(theta))^(-1) p return the same results
        """
        self.assertTrue(
            np.allclose(
                np.linalg.solve(self.R.T, self.p0),
                fss.jacobi(self.theta, self.p0, self.lam1, transp=True)
            )
        )


    def test_gen_pths(self):
        """
        tests if pTh is a valid distribution
        """
        self.assertTrue(np.around(sum(self.pTh), decimals=5) == 1)


    def test_fss_q_grad_p(self):
        """
        tests whether implicit and explicit versions of q (d Q/d theta) p return the same results
        """
        p = fss.jacobi(self.theta, self.p0, self.lam1)
        q = fss.jacobi(self.theta, p, self.lam1, transp=True)
        theta_test = np.zeros_like(self.theta)
        self.assertTrue(
            np.allclose(
                essp.build_q_grad_p(theta_test, q, p),
                kv.q_partialQ_pth(theta_test, q, p, self.n)
            )
        )


    def test_fss_grad(self):
        """
        tests if the numeric and the analytic gradient of S_D d S_D/ d theta_ij for all ij match
        Adapted from https://github.com/spang-lab/LearnMHN/blob/main/test/test_state_space_restriction.py
        """
        pD = ut.finite_sample(self.pTh, 50)
        h = 1e-10
        original_score = fss.likelihood(self.theta, pD, self.lam1, self.lam2, self.pTh1, self.pTh2)
        # compute the partial derivatives dS_D/d theta_ij numerically
        numerical_gradient = np.empty((self.n+1, self.n+1), dtype=float)
        for i in range(self.n+1):
            for j in range(self.n+1):
                theta_copy = self.theta.copy()
                theta_copy[i, j] += h
                new_score = fss.likelihood(theta_copy, pD, self.lam1, self.lam2, self.pTh1, self.pTh2)
                numerical_gradient[i, j] = (new_score - original_score) / h

        # compute the partial derivatives dS_D/d theta_ij numerically
        new_score = fss.likelihood(self.theta, pD, self.lam1+h, self.lam2, self.pTh1, self.pTh2)
        deriv_lam1 = (new_score - original_score) / h

        new_score = fss.likelihood(self.theta, pD, self.lam1, self.lam2+h, self.pTh1, self.pTh2)
        deriv_lam2 = (new_score - original_score) / h
        grad = np.append(numerical_gradient.flatten(), [deriv_lam1, deriv_lam2])

        analytic_gradient = fss.gradient(self.theta, pD, self.lam1, self.lam2, self.n, self.p0)
        self.assertTrue(
            np.allclose(
                np.around(grad, decimals=3),
                    np.around(analytic_gradient, decimals=3)
            )
        )

if __name__ == "__main__":
    unittest.main()

