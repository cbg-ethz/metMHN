import metmhn.jx.kronvec as ssr_kv_jx
import metmhn.np.likelihood as ssr
import metmhn.jx.vanilla as vanilla
import metmhn.Utilityfunctions as utils
import numpy as np
import unittest
import jax.numpy as jnp


class KroneckerTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n = 4
        self.log_theta = utils.random_theta(self.n, 0.4)
        self.log_theta[-1] = -np.infty
        self.lam1 = 1.
        self.v_state_size = 2
        self.state_size = 2 * self.v_state_size
        self.v_state = np.random.choice(
            [1] * self.v_state_size + [0] * (self.n - self.v_state_size), size=self.n, replace=False)
        self.state = np.zeros(2 * self.n + 1, dtype=int)
        self.state[:-1:2] = self.v_state
        self.state[1::2] = self.v_state
        self.reachable = utils.reachable_states(
            self.n)[utils.ssr_to_fss(self.state)]
        self.tol = 1e-08


    def test_vanilla_kron_diag(self):
        np.testing.assert_allclose(
            ssr_kv_jx.kron_diag(log_theta=jnp.array(
                self.log_theta), state=jnp.array(self.state), p_in=jnp.ones(2**self.state_size), d_e=1.)[self.reachable],
            vanilla.kron_diag(log_theta=jnp.array(self.log_theta[:-1, :-1]), state=jnp.array(
                self.v_state), diag=jnp.ones(2**self.v_state_size)),
            rtol=self.tol
        )


    def test_vanilla_kronvec(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), d_e=1.)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    state=jnp.array(self.v_state)),
                    rtol=self.tol                
                )


    def test_vanilla_kronvec_no_diag(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), d_e = 1., diag=False)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    state=jnp.array(self.v_state), diag=False),
                    rtol=self.tol
                )


    def test_vanilla_kronvec_transp(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), d_e=1., transpose=True)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    state=jnp.array(self.v_state), transpose=True),
                    rtol=self.tol
                )


    def test_vanilla_kronvec_no_diag_transp(self):
        for j in range(1 << self.state_size):
            with self.subTest(j=j):
                if not self.reachable[j]:
                    continue
                p = np.zeros(1 << self.state_size)
                p[j] = 1
                np.testing.assert_allclose(
                    ssr_kv_jx.kronvec(log_theta=jnp.array(self.log_theta), p=jnp.array(
                        p), state=jnp.array(self.state), d_e=1.,diag=False, transpose=True)[self.reachable],
                    vanilla.kronvec(log_theta=self.log_theta[:-1, :-1], p=p[self.reachable],
                                    state=jnp.array(self.v_state), diag=False, transpose=True),
                    rtol=self.tol
                )


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
                    np.testing.assert_allclose(
                            ssr.x_partial_Q_y(log_theta=self.log_theta,
                                              x=p, y=q, state=self.state)[:-1, :-1],
                            np.array(vanilla.x_partial_Q_y(
                                log_theta=jnp.array(self.log_theta[:-1, :-1]),
                                x=p[reachable],
                                y=q[reachable],
                                state=jnp.array(self.v_state),
                            )[0]),
                            rtol=self.tol
                        )
                    


if __name__ == "__main__":
    unittest.main()
