import ssr_kronecker_vector as ssr_kv
import ssr_kronvec_jax as ssr_kv_jx
import ssr_likelihood_jax as ssr_jx
import ssr_likelihood as ssr
import vanilla
import Utilityfunctions as utils
import numpy as np
import unittest
import jax.numpy as jnp
import jax
import explicit_statetespace as essp


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.log_theta[-1] = -np.infty
        self.lam1 = np.random.exponential(10, 1)
        self.v_state_size = 2
        self.state_size = 2 * self.v_state_size
        self.v_state = np.random.choice(
            [1] * self.v_state_size + [0] * (self.n - self.v_state_size), size=self.n, replace=False)
        self.state = np.zeros(2 * self.n + 1, dtype=int)
        self.state[:-1:2] = self.v_state
        self.state[1::2] = self.v_state
        self.reachable = utils.reachable_states(
            self.n)[utils.ssr_to_fss(self.state)]

    def test_vanilla_kron_diag(self):
        self.assertTrue(
            np.allclose(
                ssr_kv_jx.kron_diag(log_theta=jnp.array(
                    self.log_theta), n=self.n, state=jnp.array(self.state), state_size=self.state_size)[self.reachable],
                vanilla.kron_diag(log_theta=jnp.array(self.log_theta[:-1, :-1]), n=self.n, state=jnp.array(
                    self.v_state), state_size=self.v_state_size)
            )
        )

    def test_vanilla_kronvec(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.state_size)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    n=self.n, state=jnp.array(self.v_state), state_size=self.v_state_size)
                ))

    def test_vanilla_kronvec_no_diag(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.state_size, diag=False)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    n=self.n, state=jnp.array(self.v_state), state_size=self.v_state_size, diag=False)
                ))

    def test_vanilla_kronvec_transp(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.state_size, transpose=True)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    n=self.n, state=jnp.array(self.v_state), state_size=self.v_state_size, transpose=True)
                ))

    def test_vanilla_kronvec_no_diag_transp(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                self.assertTrue(np.allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), n=self.n, state=jnp.array(self.state), state_size=self.state_size, diag=False, transpose=True)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    n=self.n, state=jnp.array(self.v_state), state_size=self.v_state_size, diag=False, transpose=True)
                ))

    def test_vanilla_resolvent_p(self):

        Q = essp.build_q(self.log_theta)[np.ix_(utils.reachable_states(self.n) & utils.ssr_to_fss(
            self.state), utils.reachable_states(self.n) & utils.ssr_to_fss(self.state))]
        R = self.lam1*np.eye(2**(self.v_state_size)) - Q
        for j in range(1 << self.v_state_size):
            with self.subTest(j=j):
                p = np.zeros(1 << self.v_state_size)
                p[j] = 1
                self.assertTrue(
                    np.allclose(
                        np.linalg.solve(R, p),
                        vanilla.R_i_inv_vec(log_theta=self.log_theta[:-1, :-1],
                                            x=p, lam=self.lam1, state=self.v_state, state_size=self.v_state_size)
                    ))

    def test_vanilla_q_grad_p(self):
        reachable = utils.reachable_states(
            self.n)[utils.ssr_to_fss(state=self.state)]
        for i in range(1 << self.state_size):
            if not reachable[i]:
                continue
            for j in range(1 << self.state_size):
                with self.subTest(i=i, j=j):
                    if reachable[j]:
                        continue
                    p, q = np.zeros(1 << self.state_size), np.zeros(
                        1 << self.state_size)
                    p[i], q[j] = 1, 1
                    self.assertTrue(
                        np.allclose(
                            ssr.x_partial_Q_y(log_theta=self.log_theta,
                                              x=p, y=q, state=self.state)[:-1, :-1],
                            np.array(vanilla.x_partial_Q_y(
                                log_theta=jnp.array(self.log_theta[:-1, :-1]),
                                x=p[reachable],
                                y=q[reachable],
                                state=jnp.array(self.v_state),
                                n=self.n
                            ))
                        )
                    )


if __name__ == "__main__":
    unittest.main()
