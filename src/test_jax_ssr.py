import ssr_kronecker_vector as ssr_kv
import ssr_kronvec_jax as ssr_kv_jx
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
        # self.Q = essp.build_q(self.log_theta)
        # self.q_diag = np.diag(np.diag(self.Q))
        self.p0 = np.zeros(2**(2*self.n+1))
        self.p0[0] = 1
        self.lam1 = np.random.exponential(10, 1)
        self.lam2 = np.random.exponential(10, 1)
        # self.R = self.lam1*np.eye(2**(2*self.n + 1)) - self.Q
        self.n_ss = 0
        while self.n_ss < 2:
            self.state = np.random.randint(2, size=2*self.n+1)
            self.n_ss = self.state.sum()
        # print(self.state)
        # self.pTh1, self.pTh2 = fss.generate_pths(self.log_theta, self.p0, self.lam1, self.lam2)
        # self.pTh = self.lam1 * self.lam2 / (self.lam1 - self.lam2)*(self.pTh2 - self.pTh1)

    # def test_with_profiler(self):
    #     with jax.profiler.trace("/tmp/tensorboard"):
    #         for j in range(1 << self.n_ss):
    #             p = np.zeros(1 << self.n_ss)
    #             p[j] = 1
    #             ssr_kv_jx.kronvec_sync(log_theta=jnp.array(self.log_theta), p=jnp.array(
    #                 p), n=self.n, i=0, state=jnp.array(self.state))

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

    def test_kron_diag(self):
        self.assertTrue(
            np.allclose(
                ssr_kv.kron_diag(
                    log_theta=self.log_theta, n=self.n, state=self.state),
                ssr_kv_jx.kron_diag(log_theta=jnp.array(
                    self.log_theta), n=self.n, state=jnp.array(self.state), state_size=sum(self.state))
            )
        )
    
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


if __name__ == "__main__":
    unittest.main()
