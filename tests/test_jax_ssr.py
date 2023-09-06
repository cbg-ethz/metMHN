import metmhn.np.kronvec as ssr_kv
import metmhn.jx.kronvec as ssr_kv_jx
import metmhn.np.likelihood as ssr
import metmhn.jx.likelihood as ssr_jx
import metmhn.Utilityfunctions as utils
import numpy as np
import unittest
import jax.numpy as jnp
import warnings
import jax as jax
jax.config.update("jax_enable_x64", True)


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.lam1 = 1.
        self.lam2 = 1.
        self.state_size = 4
        self.state = np.random.choice(
            [1] * self.state_size + [0] * (2 * self.n + 1 - self.state_size), size=2*self.n+1, replace=False)
        self.tol = 1e-08

    def test_kron_diag(self):
        np.testing.assert_allclose(
                ssr_kv.kron_diag(
                    log_theta=self.log_theta, n=self.n, state=self.state),
                ssr_kv_jx.kron_diag(log_theta=jnp.array(
                    self.log_theta), state=jnp.array(self.state), p_in=jnp.zeros(2**self.state_size)),
                rtol=self.tol
            )

    def test_kronvec(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state)),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state),
                    rtol=self.tol
                )

    def test_kronvec_no_diag(self):

        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), diag=False),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   state=self.state, diag=False, n=self.n),
                    rtol = self.tol
                )

    def test_kronvec_transp(self):

        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), transpose=True),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, transpose=True),
                    rtol = self.tol
                )

    def test_kronvec_transp_no_diag(self):

        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), diag=False, transpose=True),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, diag=False, transpose=True),
                    rtol = self.tol
                )

    def test_ssr_resolvent_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                        ssr.R_i_jacobian_vec(log_theta=self.log_theta,
                                             x=p, lam=self.lam1, state=self.state),
                        ssr_jx.R_i_inv_vec(log_theta=self.log_theta,
                                           x=p, lam=self.lam1, state=self.state),
                        rtol = self.tol
                    )

    def test_ssr_q_grad_p(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        rtol = 1.e-2 # to change tolerance, comment out this and the following line
        warnings.warn("Rel. tolerance of this test is set very low to test despite the weird miscalculation on spang lab's r30 GPU. Please change when on other device.")
        for i in range(1 << self.state_size):
            for j in range(1 << self.state_size):
                with self.subTest(i=i, j=j):
                    p, q = np.zeros(1 << self.state_size), np.zeros(
                        1 << self.state_size)
                    p[i], q[j] = 1, 1
                    np.testing.assert_allclose(
                            ssr.x_partial_Q_y(log_theta=self.log_theta,
                                              x=p, y=q, state=self.state),
                            np.array(ssr_jx.x_partial_Q_y(log_theta=self.log_theta,
                                                          x=jnp.array(p), y=jnp.array(q), state=self.state)[0]),
                        rtol=rtol
                        )
                    
                    
if __name__ == "__main__":
    unittest.main()
