import metmhn.regularized_optimization as regopt
import metmhn.Utilityfunctions as utils
import metmhn.simulations as simul
import jax.numpy as jnp
import numpy as np
import unittest
import jax as jax
jax.config.update("jax_enable_x64", True)

class LikelihoodTestCase(unittest.TestCase):
    @classmethod
    def setUp(self):
        self.n_sim = int(5e04)
        rng = np.random.default_rng(seed=42)
        self.n_mut = 3
        self.n_states = 2**(2*self.n_mut + 1)
        
        self.theta = jnp.array(utils.random_theta(self.n_mut, 0.2))
        self.fd_effects = jnp.array(rng.normal(0, 1, size=self.n_mut))
        self.fd_effects = jnp.append(self.fd_effects, -1)
        self.sd_effects = jnp.array(rng.normal(0, 1, size=self.n_mut+1))
        self.dat, _ = simul.simulate_dat(self.theta, self.fd_effects, 
                                         self.sd_effects, self.n_sim, rng)
        self.counts = dict(zip([i for i in range(self.n_states)], 
                          [ 0 for i in range(self.n_states)]))
        
        prim_hashed = list(np.packbits(self.dat, axis=1, bitorder="little")[:,0])
        for i in prim_hashed:
            self.counts[i] += 1


    def test_lp_prim(self):
        dp = np.array([1, 1, 1, 1, 1,1, 0]).reshape((1,-1))
        ind = np.packbits(dp, axis=1, bitorder="little")[0]
        sim_freq = np.log(self.counts[ind[0]]/self.n_sim)
        ana_freq = regopt.lp_prim_only(self.theta, self.fd_effects, jnp.array(dp))
        print(sim_freq, ana_freq)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)
    

    def test_lp_prim_met(self):
        mt_states = np.kron(utils.state_space(self.n_mut), np.array([0,1]))
        pt_states =  np.kron(np.ones((2**self.n_mut, self.n_mut), dtype=np.int8), 
                             np.array([1,0])) 
        combined = np.column_stack((mt_states + pt_states, 
                                    np.ones((2**self.n_mut,1), dtype=np.int8))
                                    )
        inds = np.packbits(combined, axis=1, bitorder="little")[:,0]
        sim_freq = 0.
        for i in inds:
            sim_freq += self.counts[i]
        sim_freq = np.log(sim_freq/self.n_sim)
        dp = np.ones((1, 2*self.n_mut+1), dtype=np.int8)
        ana_freq = regopt.lp_prim_only(self.theta, self.fd_effects, jnp.array(dp))
        print(sim_freq, ana_freq)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)
    

    def test_lp_coupled(self):
        dp = np.array([1, 1, 1, 1, 1, 1, 1]).reshape((1,-1))
        ind = np.packbits(dp, axis=1, bitorder="little")[0]
        sim_freq = np.log(self.counts[ind[0]]/self.n_sim)
        ana_freq = regopt.lp_coupled(self.theta, self.fd_effects, self.sd_effects, jnp.array(dp))
        print(sim_freq, ana_freq)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)