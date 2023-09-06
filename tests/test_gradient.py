import metmhn.regularized_optimization as regopt
import metmhn.Utilityfunctions as utils
import jax.numpy as jnp
import numpy as np
import unittest
import jax as jax
jax.config.update("jax_enable_x64", True)

def finite_difference(fun, fun_args, n, h):
    g_ = np.zeros((n,n))
    d_fd = np.zeros(n)
    d_sd = np.zeros(n)
    score = fun(*fun_args)
    for i in range(n):
        for j in range(n):
            theta = fun_args[0]
            theta_h = theta.at[i,j].add(h)
            args_h = [theta_h, *fun_args[1:]]
            score_h = fun(*args_h)
            g_[i,j] = (score_h - score)/h

        fd_h = fun_args[1].at[i].add(h)
        args_h = [fun_args[0], fd_h, *fun_args[2:]]
        score_h = fun(*args_h)
        d_fd[i] = (score_h - score)/h

        if len(fun_args) == 4:
            sd_h = fun_args[2].at[i].add(h)
            args_h = [*fun_args[:2], sd_h, fun_args[-1]]
            score_h = fun(*args_h)
            d_sd[i] = (score_h - score)/h
    return np.concatenate((g_.flatten(), d_fd, d_sd)) 


class DerivativeTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n_mut = 3
        N = 2*self.n_mut+1
        rng = np.random.default_rng(seed=42)
        self.theta = jnp.array(utils.random_theta(self.n_mut, 0.2))
        self.fd_effects = jnp.log(jnp.array([1,2,3,4]))
        self.sd_effects = jnp.log(jnp.array([0.5, 1.5, 2.5, 3.5]))

        self.state_prim_met = jnp.array(np.append(rng.binomial(1, 0.6, 2*self.n_mut), 1)).reshape((1, N))
        self.state_prim_only = jnp.array(np.append(rng.binomial(1, 0.6, 2*self.n_mut,), 0)).reshape((1, N))
        self.state_met = jnp.array(np.append(rng.binomial(1, 0.6, 2*self.n_mut),1)).reshape((1,N))
        self.state_coupled = jnp.array(np.append(rng.binomial(1, 0.6, 2*self.n_mut), 1)).reshape((1,N))

        self.h = 1e-08  # Stepsize for finite difference method
        self.tol = 1e-04    # Tolerance for comparisson between numeric and analytic solution

    def test_prim_only_deriv(self):
        params = [self.theta, self.fd_effects, self.state_prim_only]
        g_num = finite_difference(regopt.lp_prim_only, params, self.n_mut+1, self.h)
        np.testing.assert_allclose(g_num, 
                                   np.array(regopt.grad_prim_only(self.theta, self.fd_effects, 
                                                                  self.state_prim_only)[1]),
                                   rtol=self.tol)
    
    def test_prim_met_deriv(self):
        params = [self.theta, self.fd_effects, self.state_prim_met]
        g_num = finite_difference(regopt.lp_prim_only, params, self.n_mut+1, self.h)
        np.testing.assert_allclose(g_num, 
                                   np.array(regopt.grad_prim_only(self.theta, self.fd_effects, 
                                                                  self.state_prim_met)[1]), 
                                   rtol=self.tol)
    
    def test_met_only_deriv(self):
        params = [self.theta, self.fd_effects, self.sd_effects, self.state_met]
        g_num = finite_difference(regopt.lp_met_only, params, self.n_mut+1, self.h)
        np.testing.assert_allclose(g_num, 
                                   np.array(regopt.grad_met_only(self.theta, self.fd_effects, 
                                                                 self.sd_effects, self.state_met)[1]), 
                                   rtol=self.tol)
    
    def test_coupled_deriv(self):
        params = [self.theta, self.fd_effects, self.sd_effects, self.state_coupled]
        g_num = finite_difference(regopt.lp_coupled, params, self.n_mut+1, self.h)
        np.testing.assert_allclose(g_num, 
                                   np.array(regopt.grad_coupled(self.theta, self.fd_effects, 
                                                                self.sd_effects, self.state_coupled)[1]), 
                                   rtol=self.tol)
    
    def test_full_deriv(self):
        n_tot =  self.n_mut + 1
        params = np.concatenate((self.theta.flatten(), self.fd_effects, self.sd_effects))
        params[n_tot*(n_tot+1)-1] = 0.
        g_num = np.zeros(n_tot*(n_tot+2))
        score = regopt.log_lik(params, self.state_prim_only, self.state_prim_met, 
                               self.state_met, self.state_coupled, 0., 0.8)
        for i in range(n_tot*(n_tot+2)):
            params_h = params.copy()
            params_h[i] += self.h
            score_h = regopt.log_lik(params_h, self.state_prim_only, self.state_prim_met, 
                                     self.state_met, self.state_coupled, 0., 0.8)
            g_num[i] = (score_h - score)/self.h
        g_num[n_tot*(n_tot+1)-1] = 0.
        np.testing.assert_allclose(g_num, 
                                   np.array(regopt.grad(params, self.state_prim_only, self.state_prim_met, 
                                                        self.state_met, self.state_coupled, 0., 0.8)),
                                   rtol=self.tol)

if __name__ == "__main__":
    unittest.main()

