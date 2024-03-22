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
        self.n_sim = int(1e05)
        rng = np.random.default_rng(seed=42)
        self.n_mut = 3
        self.n_states = 2**(2*self.n_mut + 1)
        
        self.theta = jnp.array(utils.random_theta(self.n_mut, 0.2))
        self.d_pt = jnp.array(rng.normal(0, 1, size=self.n_mut+1))
        self.d_mt = jnp.array(rng.normal(0, 1, size=self.n_mut+1))
        rng_key = jax.random.PRNGKey(seed=42)
        self.dat = simul.simulate_dat_jax(self.theta, self.d_pt, self.d_mt, self.n_sim, original_key=rng_key)
        self.counts = dict(zip([i for i in range(self.n_states)], 
                          [ 0 for i in range(self.n_states)]))
        
        genos_hashed = list(np.packbits(self.dat[:,:-1], axis=1, bitorder="little")[:,0])
        for i in genos_hashed:
            self.counts[i] += 1


    def test_lp_prim(self):
        dp = np.array([1, 1, 1, 1, 1,1, 0, -99, 0]).reshape((1,-1))
        ind = np.packbits(dp[:,:-2], axis=1, bitorder="little")[0]
        sim_freq = np.log(self.counts[ind[0]]/self.n_sim)
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, jnp.array(dp), 0)
        print(np.exp(sim_freq), np.exp(ana_freq))
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
        dp = jnp.array([1]*6+[1,-99,1]).reshape((1,-1))
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, dp, 0)

        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)
    

    def test_lp_met(self):
        pt_states = np.kron(utils.state_space(self.n_mut), np.array([1,0]))
        mt_states =  np.kron(np.ones((2**self.n_mut, self.n_mut), dtype=np.int8), 
                             np.array([0,1])) 
        combined = np.column_stack((mt_states + pt_states, 
                                    np.ones((2**self.n_mut,1), dtype=np.int8))
                                    )
        inds = np.packbits(combined, axis=1, bitorder="little")[:,0]
        sim_freq = 0.
        for i in inds:
            sim_freq += self.counts[i]
        sim_freq = np.log(sim_freq/self.n_sim)
        dp = jnp.array([1]*6+[1,-99,2]).reshape((1,-1))
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, dp, 0)
        print(sim_freq, ana_freq)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)


    def test_lp_coupled_0(self):
        dp = jnp.array([1]*6+[1, 0, 3]).reshape((1,-1))
        ind = np.packbits(dp[:,:-2], axis=1, bitorder="little")[0]
        sim_freq = np.log(self.counts[ind[0]]/self.n_sim)
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, dp, 0)
        print(sim_freq, ana_freq)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)


    def test_lp_coupled_1(self):
        dp = jnp.array([1]*6+[1, 1, 3]).reshape((1,-1))
        
        counts = dict(zip([i for i in range(self.n_states)], 
                          [ 0 for i in range(self.n_states)]))
        
        genos_hashed = list(np.packbits(self.dat[self.dat[:,-1]==1,:-1], 
                                        axis=1, bitorder="little")[:,0])
        for i in genos_hashed:
            counts[i] += 1
        ind = np.packbits(dp[:,:-2], axis=1, bitorder="little")[0]
        sim_freq = np.log(counts[ind[0]]/self.n_sim)
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, dp, 0)
        print(sim_freq, ana_freq)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)


    def test_lp_coupled_2(self):
        dp = jnp.array([1]*6+[1, 2, 3]).reshape((1,-1))
        
        counts = dict(zip([i for i in range(self.n_states)], 
                          [ 0 for i in range(self.n_states)]))
        
        genos_hashed = list(np.packbits(self.dat[self.dat[:,-1]==2,:-1], 
                                        axis=1, bitorder="little")[:,0])
        for i in genos_hashed:
            counts[i] += 1
        ind = np.packbits(dp[:,:-2], axis=1, bitorder="little")[0]
        sim_freq = np.log(counts[ind[0]]/self.n_sim)
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, dp, 0)
        print(f"Number of occurences:{counts[ind[0]]}")
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)


    def test_lp_empty(self):        
        dp = jnp.array([0]*6+[1,0,3]).reshape((1,-1))        
        ind = np.packbits(dp[:,:-2], axis=1, bitorder="little")[0]
        print(f"Number of occurences:{self.counts[ind[0]]}")
        sim_freq = np.log(self.counts[ind[0]]/self.n_sim)
        ana_freq = regopt.score(self.theta, self.d_pt, self.d_mt, dp, 0)
        np.testing.assert_approx_equal(sim_freq, ana_freq, significant=2)