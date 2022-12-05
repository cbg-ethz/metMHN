import likelihood as fss
import ssr_likelihood as ssr
import numpy as np
import kronecker_vector as kv
import ssr_kronecker_vector as ssr_kv
import Utilityfunctions as utils
import explicit_statetespace as essp
import unittest


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 5
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.Q = essp.build_q(self.log_theta)
        self.q_diag = np.diag(np.diag(self.Q))
        self.p0 = np.zeros(2**(2*self.n+1))
        self.p0[0] = 1
        self.lam = np.random.exponential(10, 1)
        self.R = self.lam*np.eye(2**(2*self.n + 1)) - self.Q
        self.state = np.random.randint(2, size=2*self.n+1)
        self.n_ss = self.state.sum()

    def test_fss_q_p(self):
        """
        tests Q(theta) p
        """
        self.assertTrue(
            np.allclose(
                self.Q @ self.p0,
                kv.qvec(self.log_theta, self.p0, True, False)
            )
        )

    def test_fss_Q_transp_p(self):
        """
        tests Q(theta)^T p
        """
        self.assertTrue(
            np.allclose(
                self.Q.T @ self.p0,
                kv.qvec(self.log_theta, self.p0, True, True)
            )
        )

    def test_fss_diag_Q_p(self):
        """
        tests diag(Q(theta)) * p
        """
        self.assertTrue(
            np.allclose(
                self.q_diag @ self.p0,
                (-1) * kv.diag_q(self.log_theta) * self.p0
            )
        )

    def test_fss_q_no_diag_p(self):
        """
        tests (Q - diag(Q(theta))) p
        """
        self.assertTrue(
            np.allclose(
                (self.Q - self.q_diag) @ self.p0,
                kv.qvec(self.log_theta, self.p0, False, False)
            )
        )

    def test_fss_resolvent_p(self):
        """
        tests (lambda * I - Q(theta))^(-1) p
        """
        self.assertTrue(
            np.allclose(
                np.linalg.solve(self.R.T, self.p0),
                fss.jacobi(self.log_theta, self.p0, self.lam, transp=True)
            )
        )

    def test_fss_q_grad_p(self):
        """
        tests q (d Q/d theta) p
        """
        p = fss.jacobi(self.log_theta, self.p0, self.lam)
        q = fss.jacobi(self.log_theta, p, self.lam, transp=True)
        theta_test = np.zeros_like(self.log_theta)
        self.assertTrue(
            np.allclose(
                essp.build_q_grad_p(theta_test, q, p),
                kv.q_partialQ_pth(theta_test, q, p, self.n)
            )
        )

    def test_ssr_Q_p(self):
        """
        Tests restricted version of Q(theta) e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(np.allclose(
                    essp.ssr_build_q(dpoint=self.state,
                                     log_theta=self.log_theta) @ p,
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                n=self.n, state=self.state)
                ))

    def test_ssr_Q_no_diag_p(self):
        """
        Tests restricted version of (Q(theta) - diag(Q(theta))) e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                q = ssr_kv.kronvec(log_theta=self.log_theta,
                                   p=p, n=self.n, state=self.state)
                q[j] = 0
                self.assertTrue(np.allclose(
                    q,
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                n=self.n, state=self.state, diag=False)
                ))

    def test_ssr_diag_Q(self):
        """
        Tests restricted version of diag(Q(theta))
        """
        self.assertTrue(np.allclose(
            np.diag(essp.ssr_build_q(
                dpoint=self.state, log_theta=self.log_theta)),
            ssr_kv.kron_diag(log_theta=self.log_theta,
                             n=self.n, state=self.state)
        ))

    def test_ssr_resolvent_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                assert (np.allclose(
                    np.linalg.inv(self.lam * np.identity(1 << self.n_ss)
                        - essp.ssr_build_q(dpoint=self.state, log_theta=self.log_theta)) @ p,
                    ssr.R_i_inv_vec(log_theta=self.log_theta,
                                    x=p, lam=self.lam, state=self.state),
                ))

    def test_ssr_Q_T_p(self):
        """
        Tests restricted version of Q(theta).T e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                assert (np.allclose(
                    essp.ssr_build_q(dpoint=self.state,
                                     log_theta=self.log_theta).T @ p,
                    ssr.kronvec(log_theta=self.log_theta, p=p,
                                n=self.n, state=self.state, transpose=True)
                ))

    def test_ssr_Q_no_diag_T_p(self):
        """
        Tests restricted version of (Q(theta).T - diag(Q(theta))) e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                q = ssr_kv.kronvec(log_theta=self.log_theta,
                                   p=p, n=self.n, state=self.state, transpose=True)
                q[j] = 0
                self.assertTrue(np.allclose(
                    q,
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                n=self.n, state=self.state, diag=False, transpose=True)
                ))

    def test_ssr_q_grad_p(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        restricted = utils.ssr_to_fss(self.state)
        p = ssr.R_i_inv_vec(log_theta=self.log_theta, x=self.p0[restricted],
                            lam=self.lam, state=self.state)
        q = ssr.R_i_inv_vec(log_theta=self.log_theta, x=self.p0[restricted],
                            lam=self.lam, state=self.state, transpose=True)
        p_fss = np.zeros(1 << (2*self.n + 1))
        q_fss = np.zeros(1 << (2*self.n + 1))
        p_fss[restricted] = p
        q_fss[restricted] = q
        self.assertTrue(
            np.allclose(
                kv.q_partialQ_pth(np.zeros_like(
                    self.log_theta), q_fss, p_fss, self.n),
                ssr.x_partial_Q_y(log_theta=self.log_theta,
                                  x=p, y=q, state=self.state)
            )
        )


if __name__ == "__main__":
    unittest.main()
