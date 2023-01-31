import ssr_kronecker_vector as ssr_kv
import ssr_kronvec_jax as ssr_kv_jx
import ssr_likelihood_jax as ssr_jx
import ssr_likelihood as ssr
import Utilityfunctions as utils
import numpy as np
import unittest
import jax.numpy as jnp


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.lam1 = np.random.exponential(10, 1)
        self.lam2 = np.random.exponential(10, 1)
        self.state_size = 4
        self.state = np.random.choice(
            [1] * self.state_size + [0] * (2 * self.n + 1 - self.state_size), size=2*self.n+1, replace=False)

    def test_kron_diag(self):
        self.assertTrue(
            np.allclose(
                ssr_kv.kron_diag(
                    log_theta=self.log_theta, n=self.n, state=self.state),
                ssr_kv_jx.kron_diag(log_theta=jnp.array(
                    self.log_theta), state=jnp.array(self.state), p_in=jnp.zeros(2**self.state_size))
            )
        )

    def test_kronvec(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state)),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state)
                ))

    def test_kronvec_no_diag(self):

        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), diag=False),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   state=self.state, diag=False, n=self.n)
                ))

    def test_kronvec_transp(self):

        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), transpose=True),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, transpose=True)
                ))

    def test_kronvec_transp_no_diag(self):

        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), diag=False, transpose=True),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, diag=False, transpose=True)
                ))

    def test_ssr_resolvent_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(
                    np.allclose(
                        ssr.R_i_jacobian_vec(log_theta=self.log_theta,
                                             x=p, lam=self.lam1, state=self.state),
                        ssr_jx.R_i_inv_vec(log_theta=self.log_theta,
                                           x=p, lam=self.lam1, state=self.state)
                    ))

    @unittest.skip("This is weird on r30 GPU")
    def test_ssr_q_grad_p(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        for i in range(1 << self.state_size):
            for j in range(1 << self.state_size):
                with self.subTest(i=i, j=j):
                    p, q = np.zeros(1 << self.state_size), np.zeros(
                        1 << self.state_size)
                    p[i], q[j] = 1, 1
                    self.assertTrue(
                        np.allclose(
                            ssr.x_partial_Q_y(log_theta=self.log_theta,
                                              x=p, y=q, state=self.state),
                            np.array(ssr_jx.x_partial_Q_y(log_theta=jnp.array(self.log_theta),
                                                          x=jnp.array(p), y=jnp.array(q), state=jnp.array(self.state)))
                        )
                    )

    @unittest.skip("This is weird on r30 GPU")
    def test_ssr_gradient(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        p0 = np.zeros(1 << self.state_size)
        p0[0] = 1
        p_D = ssr.R_i_jacobian_vec(log_theta=self.log_theta, x=p0,
                                   lam=self.lam1, state=self.state)
        self.assertTrue(
            np.allclose(
                ssr.gradient(
                    log_theta=self.log_theta,
                    p_D=p_D, lam1=self.lam1, lam2=self.lam2, state=self.state),
                ssr_jx.gradient(
                    log_theta=jnp.array(self.log_theta),
                    p_D=jnp.array(p_D), lam1=self.lam1, lam2=self.lam2, state=jnp.array(self.state), state_size=self.state_size),
            )
        )


if __name__ == "__main__":
    unittest.main()
