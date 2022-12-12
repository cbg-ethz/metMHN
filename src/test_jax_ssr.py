import ssr_kronecker_vector as ssr_kv
import ssr_kronvec_jax as ssr_kv_jx
import Utilityfunctions as utils
import numpy as np
import unittest
import jax.numpy as jnp


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 1
        self.log_theta = utils.random_theta(self.n, 0.4)
        # self.Q = essp.build_q(self.log_theta)
        # self.q_diag = np.diag(np.diag(self.Q))
        self.p0 = np.zeros(2**(2*self.n+1))
        self.p0[0] = 1
        self.lam1 = np.random.exponential(10, 1)
        self.lam2 = np.random.exponential(10, 1)
        # self.R = self.lam1*np.eye(2**(2*self.n + 1)) - self.Q
        self.state = np.random.randint(2, size=2*self.n+1)
        # self.state=np.zeros(2*self.n+1, dtype=int)
        # self.state[-1] = 0
        self.state = np.array([1, 1, 0])
        # print(self.state)
        self.n_ss = self.state.sum()
        # self.pTh1, self.pTh2 = fss.generate_pths(self.log_theta, self.p0, self.lam1, self.lam2)
        # self.pTh = self.lam1 * self.lam2 / (self.lam1 - self.lam2)*(self.pTh2 - self.pTh1)

    def test_kronvec_seed(self):

        for j in range(1 << self.n_ss):
            with self.subTest(j=j):
                p = np.zeros(1 << self.n_ss)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec_seed(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state)),
                    ssr_kv.kronvec_seed(log_theta=self.log_theta, p=p,
                                        n=self.n, state=self.state)
                ))

    def test_kronvec_seed_no_diag(self):

            for j in range(1 << self.n_ss):
                with self.subTest(j=j):
                    p = np.zeros(1 << self.n_ss)
                    p[j] = 1
                    self.assertTrue(np.allclose(
                        ssr_kv_jx.kronvec_seed(log_theta=jnp.array(self.log_theta), p=jnp.array(
                            p), n=self.n, state=jnp.array(self.state), diag=False),
                        ssr_kv.kronvec_seed(log_theta=self.log_theta, p=p,
                                            n=self.n, state=self.state, diag=False)
                    ))

    def test_kronvec_seed_transp(self):

            for j in range(1 << self.n_ss):
                with self.subTest(j=j):
                    p = np.zeros(1 << self.n_ss)
                    p[j] = 1
                    self.assertTrue(np.allclose(
                        ssr_kv_jx.kronvec_seed(log_theta=jnp.array(self.log_theta), p=jnp.array(
                            p), n=self.n, state=jnp.array(self.state), transpose=True),
                        ssr_kv.kronvec_seed(log_theta=self.log_theta, p=p,
                                            n=self.n, state=self.state, transpose=True)
                    ))


    def test_kronvec_seed_transp_no_diag(self):

            for j in range(1 << self.n_ss):
                with self.subTest(j=j):
                    p = np.zeros(1 << self.n_ss)
                    p[j] = 1
                    self.assertTrue(np.allclose(
                        ssr_kv_jx.kronvec_seed(log_theta=jnp.array(self.log_theta), p=jnp.array(
                            p), n=self.n, state=jnp.array(self.state), transpose=True, diag=False),
                        ssr_kv.kronvec_seed(log_theta=self.log_theta, p=p,
                                            n=self.n, state=self.state, transpose=True, diag=False)
                    ))

if __name__ == "__main__":
    unittest.main()
