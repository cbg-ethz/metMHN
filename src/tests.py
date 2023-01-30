import likelihood as fss
import ssr_likelihood as ssr
import numpy as np
import kronecker_vector as kv
import ssr_kronecker_vector as ssr_kv
import Utilityfunctions as utils
import explicit_statetespace as essp
import mhn as mhn
import unittest


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 5
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.Q = essp.Q(self.log_theta)
        self.q_diag = np.diag(np.diag(self.Q))
        self.p_0 = np.zeros(2**(2*self.n+1))
        self.p_0[0] = 1
        self.lam1 = np.random.exponential(10, 1)
        self.lam2 = np.random.exponential(10, 1)
        self.R = self.lam1*np.eye(2**(2*self.n + 1)) - self.Q
        self.state = np.random.randint(2, size=2*self.n+1)
        self.n_ss = self.state.sum()
        self.p_th_1, self.p_th_2 = fss.generate_pths(self.log_theta, self.p_0, self.lam1, self.lam2)
        self.p_th = self.lam1 * self.lam2 / (self.lam1 - self.lam2)*(self.p_th_2 - self.p_th_1)


    def test_fss_q_p(self):
        """
        tests whether implicit and explicit version of Q(theta) p return the same results
        """
        self.assertTrue(
            np.allclose(
                self.Q @ self.p_0,
                kv.qvec(self.log_theta, self.p_0, True, False)
            )
        )

    def test_fss_q_transp_p(self):
        """
        tests whether implicit and explicit version of Q(theta)^T p return the same results
        """
        self.assertTrue(
            np.allclose(
                self.Q.T @ self.p_0,
                kv.qvec(log_theta=self.log_theta,
                        p=self.p_0, diag=True, transpose=True)
            )
        )

    def test_fss_diag_Q_p(self):
        """
        tests whether implicit and explicit version of diag(Q(theta)) * p return the same results
        """
        self.assertTrue(
            np.allclose(
                self.q_diag @ self.p_0,
                (-1) * kv.diag_q(self.log_theta) * self.p_0
            )
        )

    def test_fss_q_no_diag_p(self):
        """
        tests whether implicit and explicit verion of (Q - diag(Q(theta))) p return the same results
        """
        self.assertTrue(
            np.allclose(
                (self.Q - self.q_diag) @ self.p_0,
                kv.qvec(self.log_theta, self.p_0, False, False)
            )
        )

    def test_fss_resolvent_p(self):
        """
        tests whether implicit and explicit version of (lambda * I - Q(theta))^(-1) p return the same results
        """
        self.assertTrue(
            np.allclose(
                np.linalg.solve(self.R.T, self.p_0),
                fss.jacobi(self.log_theta, self.p_0, self.lam1, transpose=True)
            )
        )

    def test_gen_pths(self):
        """
        tests if p_th is a valid distribution
        """
        self.assertTrue(np.around(sum(self.p_th), decimals=5) == 1)


    def test_fss_q_grad_p(self):
        """
        tests whether implicit and explicit versions of q (d Q/d theta) p return the same results
        """
        p = fss.jacobi(self.log_theta, self.p_0, self.lam1)
        q = fss.jacobi(self.log_theta, p, self.lam1, transpose=True)
        theta_test = np.zeros_like(self.log_theta)
        self.assertTrue(
            np.allclose(
                essp.build_q_grad_p(theta_test, q, p),
                kv.q_partialQ_pth(theta_test, q, p, self.n)
            )
        )

    def test_fss_met_marginalization(self):
        """
        tests whether explicit marginalization over primary tumor states in the full joint distribution and
        direct generation of the marginal distribution yield the same results
        """
        full_met_marg = utils.marginalize(self.p_th, self.n, True)
        mhn_met_marg = self.lam1 * self.lam2 / (self.lam1 - self.lam2) * \
            (mhn.generate_pTh(self.log_theta, self.lam2) -
             mhn.generate_pTh(self.log_theta, self.lam1))
        self.assertTrue(np.allclose(full_met_marg, mhn_met_marg))

    def test_fss_prim_marginalization(self):
        """
        tests whether explicit marginalization over metastases states in the full joint distribution and
        direct generation of the marginal distribution yield the same results
        """
        marg = utils.marginalize(self.p_th, self.n, False)
        theta_copy = self.log_theta.copy()
        met_base = self.log_theta[-1, -1]
        theta_copy[:, -1] = 0.
        theta_copy[-1, -1] = met_base
        mhn_marg = self.lam1 * self.lam2 / (self.lam1 - self.lam2) * \
            (mhn.generate_pTh(theta_copy, self.lam2) -
             mhn.generate_pTh(theta_copy, self.lam1))
        self.assertTrue(np.allclose(marg, mhn_marg))

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
                    essp.ssr_build_q(state=self.state,
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
                state=self.state, log_theta=self.log_theta)),
            ssr_kv.kron_diag(log_theta=self.log_theta,
                             n=self.n, state=self.state)
        ))

    def test_ssr_jacobian_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector using jacobi inverse
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                assert (np.allclose(
                    np.linalg.inv(self.lam1 * np.identity(1 << self.n_ss)
                        - essp.ssr_build_q(state=self.state, log_theta=self.log_theta)) @ p,
                    ssr.R_i_jacobian_vec(log_theta=self.log_theta,
                                         x=p, lam=self.lam1, state=self.state),
                ))

    def test_ssr_resolvent_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector using forward substitution
        """
        for i in range(1 << self.n_ss):
            with self.subTest(i=i):
                p = np.zeros(1 << self.n_ss)
                p[i] = 1
                assert (np.allclose(
                    np.linalg.inv(self.lam1 * np.identity(1 << self.n_ss)
                        - essp.ssr_build_q(state=self.state, log_theta=self.log_theta)) @ p,
                    ssr.R_i_jacobian_vec(log_theta=self.log_theta,
                                         x=p, lam=self.lam1, state=self.state),
                ))

    def test_ssr_resolvent_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector using forward substitution
        """
        for i in range(1 << self.n_ss):
            with self.subTest(i=i):
                p = np.zeros(1 << self.n_ss)
                p[i] = 1
                assert (np.allclose(
                    np.linalg.inv(self.lam1 * np.identity(1 << self.n_ss)
                        - essp.ssr_build_q(state=self.state, log_theta=self.log_theta)) @ p,
                    ssr.R_i_jacobian_vec(log_theta=self.log_theta,
                                    x=p, lam=self.lam1, state=self.state),
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
                    essp.ssr_build_q(state=self.state,
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

    def test_ssr_Q_grad_p(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        restricted = utils.ssr_to_fss(self.state)
        p = ssr.R_i_jacobian_vec(log_theta=self.log_theta, x=self.p_0[restricted],
                                 lam=self.lam1, state=self.state)
        q = ssr.R_i_jacobian_vec(log_theta=self.log_theta, x=self.p_0[restricted],
                                 lam=self.lam1, state=self.state, transpose=True)
        p_fss = np.zeros(1 << (2*self.n + 1))
        q_fss = np.zeros(1 << (2*self.n + 1))
        p_fss[restricted] = p
        q_fss[restricted] = q
        self.assertTrue(
            np.allclose(
                kv.q_partialQ_pth(log_theta=self.log_theta,
                                  q=p_fss, p_th=q_fss, n=self.n),
                ssr_kv.x_partial_Q_y(log_theta=self.log_theta,
                                  x=p, y=q, state=self.state)
            )
        )

    def test_fss_grad(self):
        """
        tests if the numeric and the analytic gradient of S_D d S_D/ d theta_ij for all ij match
        Adapted from https://github.com/spang-lab/LearnMHN/blob/main/test/test_state_space_restriction.py
        """
        p_D = utils.finite_sample(self.p_th, 50)
        h = 1e-10
        original_score = fss.log_likelihood(
            self.log_theta, p_D, self.lam1, self.lam2, self.p_th_1, self.p_th_2)
        # compute the gradient numerically
        # compute the partial derivatives dS_D/d theta_ij numerically
        numerical_gradient = np.empty((self.n+1, self.n+1), dtype=float)
        for i in range(self.n+1):
            for j in range(self.n+1):
                theta_copy = self.log_theta.copy()
                theta_copy[i, j] += h
                new_score = fss.log_likelihood(
                    theta_copy, p_D, self.lam1, self.lam2, self.p_th_1, self.p_th_2)
                numerical_gradient[i, j] = (new_score - original_score) / h

        # compute the partial derivatives dS_D/d theta_ij numerically
        new_score = fss.log_likelihood(
            self.log_theta, p_D, self.lam1+h, self.lam2, self.p_th_1, self.p_th_2)
        deriv_lam1 = (new_score - original_score) / h

        new_score = fss.log_likelihood(
            self.log_theta, p_D, self.lam1, self.lam2+h, self.p_th_1, self.p_th_2)
        deriv_lam2 = (new_score - original_score) / h
        grad = np.append(numerical_gradient.flatten(),
                         [deriv_lam1, deriv_lam2])

        analytic_gradient = fss.gradient(
            self.log_theta, p_D, self.lam1, self.lam2, self.n, self.p_0)
        self.assertTrue(
            np.allclose(
                np.around(grad, decimals=3),
                np.around(analytic_gradient, decimals=3)
            )
        )


if __name__ == "__main__":
    unittest.main()
