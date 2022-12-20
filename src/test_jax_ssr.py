import ssr_kronecker_vector as ssr_kv
import ssr_kronvec_jax as ssr_kv_jx
import ssr_likelihood_jax as ssr_jx
import ssr_likelihood as ssr
import Utilityfunctions as utils
import numpy as np
import unittest
import jax.numpy as jnp
import time
import jax


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.p0 = np.zeros(2**(2*self.n+1))
        self.p0[0] = 1
        self.lam1 = np.random.exponential(10, 1)
        self.lam2 = np.random.exponential(10, 1)
        self.n_ss = 0
        while self.n_ss < 2:
            # self.state = np.random.randint(2, size=2*self.n+1)
            self.state = np.array([1, 1, 0, 1, 0, 0, 0, 0, 0])
            self.n_ss = self.state.sum()

    @unittest.skip("")
    def test_with_profiler(self):
        with jax.profiler.trace("/tmp/tensorboard"):
            p0 = np.zeros(1 << self.n_ss)
            p0[0] = 1
            p = ssr.R_i_inv_vec(log_theta=self.log_theta, x=p0,
                                lam=self.lam1, state=self.state)
            q = ssr.R_i_inv_vec(log_theta=self.log_theta, x=p0,
                                lam=self.lam1, state=self.state, transpose=True)
            self.assertTrue(
                np.allclose(
                    ssr.x_partial_Q_y(log_theta=self.log_theta,
                                      x=p, y=q, state=self.state),
                    np.array(ssr_jx.x_partial_Q_y(
                        log_theta=jnp.array(self.log_theta),
                        x=jnp.array(p), y=jnp.array(q), state=jnp.array(self.state), n=self.n))
                )
            )

    #
    # def test_speed(self):
    #         for j in range(1 << self.n_ss):
    #             p = np.zeros(1 << self.n_ss)
    #             p[j] = 1
    #             t0 = time.time()
    #             ssr_kv_jx.kronvec_sync(log_theta=jnp.array(self.log_theta), p=jnp.array(
    #                 p), n=self.n, i=0, state=jnp.array(self.state))
    #             t1 = time.time()
    #             ssr_kv.kronvec(log_theta=self.log_theta, p=p,
    #                                 n=self.n, state=self.state)
    #             t2 = time.time()
    #             print(f"jax {t1-t0:3.5f}, no jax {t2-t1:3.5f}")

    @unittest.skip("")
    def test_kron_diag(self):
        self.assertTrue(
            np.allclose(
                ssr_kv.kron_diag(
                    log_theta=self.log_theta, n=self.n, state=self.state),
                ssr_kv_jx.kron_diag(log_theta=jnp.array(
                    self.log_theta), n=self.n, state=jnp.array(self.state), state_size=sum(self.state))
            )
        )

    @unittest.skip("")
    def test_kronvec(self):
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.n_ss),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state)
                ))

    @unittest.skip("")
    def test_kronvec_no_diag(self):

        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.n_ss, diag=False),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, diag=False)
                ))

    @unittest.skip("")
    def test_kronvec_transp(self):

        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.n_ss, transpose=True),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, transpose=True)
                ))

    @unittest.skip("")
    def test_kronvec_transp_no_diag(self):

        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.n_ss, diag=False, transpose=True),
                    ssr_kv.kronvec(log_theta=self.log_theta, p=p,
                                   n=self.n, state=self.state, diag=False, transpose=True)
                ))

    @unittest.skip("")
    def test_ssr_resolvent_p(self):
        """
        Test the restricted version of R^-1 e_i = (lam I - Q)^-1 e_i for e_i the
        ith standard base vector
        """
        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(
                    np.allclose(
                        ssr.R_i_inv_vec(log_theta=self.log_theta,
                                        x=p, lam=self.lam1, state=self.state),
                        ssr_jx.R_i_inv_vec(log_theta=self.log_theta,
                                           x=p, lam=self.lam1, state=self.state, state_size=self.n_ss)
                    ))

    @unittest.skip("")
    def test_ssr_q_grad_p(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        p0 = np.zeros(1 << self.n_ss)
        p0[0] = 1
        p = ssr.R_i_inv_vec(log_theta=self.log_theta, x=p0,
                            lam=self.lam1, state=self.state)
        q = ssr.R_i_inv_vec(log_theta=self.log_theta, x=p0,
                            lam=self.lam1, state=self.state, transpose=True)
        self.assertTrue(
            np.allclose(
                ssr.x_partial_Q_y(log_theta=self.log_theta,
                                  x=p, y=q, state=self.state),
                np.array(ssr_jx.x_partial_Q_y(log_theta=jnp.array(self.log_theta),
                                              x=jnp.array(p), y=jnp.array(q), state=jnp.array(self.state), n=self.n))
            )
        )

    # @unittest.skip("")
    def test_ssr_gradient(self):
        """
        Tests restricted version of q (d Q/d theta) p
        """
        m = 30
        t0, t1, t2 = list(), list(), list()
        for _ in range(m):
            p0 = np.zeros(1 << self.n_ss)
            p0[0] = 1
            p_D = ssr.R_i_inv_vec(log_theta=self.log_theta, x=p0,
                                lam=self.lam1, state=self.state)
            t0.append(time.time())
            a = ssr.gradient(log_theta=self.log_theta,
                                p_D=p_D, lam1=self.lam1, lam2=self.lam2, state=self.state)
            t1.append(time.time())
            b = ssr_jx.gradient(
                        log_theta=self.log_theta,
                        p_D=p_D, lam1=self.lam1, lam2=self.lam2, state=jnp.array(self.state), state_size=self.n_ss, n=self.n)
            t2.append(time.time())
            self.assertTrue(
                np.allclose(
                    a,
                    b,
                    rtol=0,
                    atol=1e-03
                )
            )
        nj_time = np.array(t1) - np.array(t0)
        j_time = np.array(t2) - np.array(t1) 
        print(f"No jax: {nj_time.mean(): 3.7f} ({nj_time.std(): 3.7f})")
        print(f"Jax:    {j_time[1:].mean(): 3.7f} ({j_time[1:].std(): 3.7f}), compile time {j_time[0]:.7f}")

if __name__ == "__main__":
    unittest.main()
