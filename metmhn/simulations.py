import numpy as np
import jax.numpy as jnp
from jax import jit, vmap, lax
import jax.random as jrp 
import jax


def single_traject(
        log_theta: np.ndarray,
        pt_d_ef: np.ndarray,
        mt_d_ef: np.ndarray,
        rng: np.random.Generator,
        ) -> np.ndarray:
    """Sample a trajectory to first diagnosis or from first diagnosis to second diagnosis

    Args:
        log_theta (np.ndarray): Theta matrix with logarithmic entries
        d_effects (np.ndarray): Effects of mutations in diagnosis rate
        rng (np.random.Generator): Random number generator
        state (np.ndarray, optional): Starting state of a tumor (PT-MT) sequence
        second_s (bool, optional): If true MT-state influences diag-rate, 
            else PT-state influences diagnosis rate. Defaults to False.

    Returns:
        np.ndarray: PT-MT-state, Pre Seeding Muts
    """
    b_rates = np.diag(log_theta.copy())
    log_theta_prim = log_theta.copy()
    log_theta_prim[:-1,-1] = 0.
    n = log_theta.shape[0]
    n_d = n+1
    # 0:(n-1) PT mut. rates, (n-1) Seeding rate, (n) PT diag. rate
    # (n+1):2n+1 MT mut. rates, 2n+1 MT diag. rate
    r_rates = np.zeros(2*n+2, dtype=np.float64)
    inds = np.arange(0, 2*n+2, dtype=np.int8)
    pre_seeding = np.zeros(n, dtype=np.int8)
    state = np.zeros(2*n+2, dtype=np.int8)
    obs_order = 0.
    while True:
        # Seeding didn't happen yet
        if state[n-1] == 0:
            r_rates[n] = np.exp(np.dot(pt_d_ef, state[:n]))
            r_rates[:n] = np.exp(log_theta @ state[:n] + b_rates)
            r_rates[:n] *= (1-state[:n])
            out_rate = np.sum(r_rates)
            next_event = rng.choice(inds, size=1, p=r_rates/out_rate)
            # Tumor is diagnosed, trajectory stops
            if next_event == n:
                break
            else:
                state[[next_event, next_event+n_d]] = 1
                pre_seeding[next_event] = 1
        # Seeding already happened
        else:
            # Calculate MT-diag. rate as function of MT-state
            r_rates[-1] = np.exp(np.dot(mt_d_ef, state[n_d:-1]))*(1-state[-1])
            # Calculate PT-diag. rate as function of PT-state
            r_rates[n] = np.exp(np.dot(pt_d_ef, state[:n]))*(1-state[n])
            
            # If only the PT is diagnosed only its progression is halted
            r_rates[:n] = np.exp(log_theta_prim @ state[:n] + b_rates)*(1-state[n])
            r_rates[:n] *= (1-state[:n])

            # If only the MT is diagnosed only its progression is halted
            r_rates[n_d:-1] = np.exp(log_theta @ state[n_d:-1] + b_rates)*(1-state[-1])
            r_rates[n_d:-1] *= (1-state[n_d:-1])

            out_rate = np.sum(r_rates)
            next_event = rng.choice(inds, size=1, p=r_rates/out_rate)
            state[next_event] = 1
            if state[n] == 1 and state[-1]==0:
                obs_order = 1
            elif state[n] == 0 and state[-1] == 1:
                obs_order = 2
            else:
                pass
        # Both tumors are diagnosed, trajectory stops
        if state[-1] + state[n] == 2:
            break
    state = state.reshape((-1,2), order="F").flatten()
    return state[:-3], pre_seeding, obs_order


def simulate_dat(
    theta_in: np.ndarray,
    pt_d_ef: np.ndarray,
    mt_d_ef: np.ndarray,
    n_dat: int,
    rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a dataset consisting of observations and PT/MT observation times

    Args:
        theta_in (np.ndarray): Theta matrix with logarithmic entries
        pt_d_ef (np.ndarray): Effects of PT-Mutations on its diagnosis
        mt_d_ef (np.ndarray): Effects of MT-Mutations on its diagnosis
        n_dat (int): number of datapoints to simulate
        rng (np.random.Generator): Numpy Random Number Generator

    Returns:
        tuple[np.ndarray, np.ndarray]: dataset and pre-seeded muts
    """
    n = theta_in.shape[0]
    dat = np.zeros((n_dat, 2*n), dtype=np.int8)
    pre_seeding_counts = np.zeros(n, dtype=np.int32)
    for i in range(n_dat):
        datum, ps, obs_order = single_traject(theta_in, pt_d_ef, mt_d_ef, rng)
        dat[i,:-1] = datum
        dat[i, -1] = obs_order
        pre_seeding_counts += ps

    return dat, pre_seeding_counts[:-1]


def p_shared_mut_pre_seed(theta: np.ndarray, 
                          pt_d_ef: np.ndarray, 
                          mt_d_ef: np.ndarray, 
                          n_dat: int, 
                          rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    n = theta.shape[0]
    pre_seeded_muts = np.zeros(n-1, dtype=np.int32)
    total_muts = np.zeros(n-1, dtype=np.int32)
    i = 0
    while i < n_dat:
        datum, psp, _ = single_traject(theta, pt_d_ef, mt_d_ef, rng)
        if datum[-1] == 1:
            both = (datum[:-2:2]+datum[1:-1:2]==2)
            pre_seeded_muts += psp[:-1] * both
            total_muts += both
            i += 1
        else:
            pass
    return pre_seeded_muts, total_muts 


def p_any_mut_pre_seed(theta: np.ndarray, 
                       pt_d_ef: np.ndarray, 
                       mt_d_ef: np.ndarray, 
                       n_dat: int, 
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = theta.shape[0]
    pre_seeded_muts = np.zeros(n-1, dtype=np.int32)
    total_muts_prim = np.zeros(n-1, dtype=np.int32)
    total_muts_met = np.zeros(n-1, dtype=np.int32)
    i = 0
    while i < n_dat:
        datum, psp, _ = single_traject(theta, pt_d_ef, mt_d_ef, rng)
        if datum[-1] == 1:
            pre_seeded_muts += psp[:-1]
            total_muts_prim += datum[:-1:2]
            total_muts_met +=  datum[1::2]
            i += 1     
    return pre_seeded_muts, total_muts_prim, total_muts_met


def single_traject_jax(
        log_theta: jnp.ndarray,
        pt_d_ef: jnp.ndarray,
        mt_d_ef: jnp.ndarray,
        rng_key: jrp.PRNGKey,
        ) -> jnp.ndarray:
    """Sample a trajectory to first diagnosis or from first diagnosis to second diagnosis

    Args:
        log_theta (np.ndarray): Theta matrix with logarithmic entries
        d_effects (np.ndarray): Effects of mutations in diagnosis rate
        rng (np.random.Generator): Random number generator
        state (np.ndarray, optional): Starting state of a tumor (PT-MT) sequence
        second_s (bool, optional): If true MT-state influences diag-rate, 
            else PT-state influences diagnosis rate. Defaults to False.

    Returns:
        np.ndarray: PT-MT-state, Pre Seeding Muts
    """
    
    def tumor_dynamics(carry):
        r_key, state, obs_order = carry

        # Calculate MT reaction rates, all are zero if seeding didn't happen yet
        mt_rates = jnp.concatenate((jnp.exp(log_theta @ state[n_d:-1] + b_rates)*(1-state[-1])*(1-state[n_d:-1]), 
                                    jnp.exp(jnp.dot(mt_d_ef, state[n_d:-1]))*(1-state[-1])), axis=None)
        
        mt_rates = lax.select(state[n-1],
                              mt_rates, 
                              jnp.zeros_like(mt_rates)
                              )
        
        # Ordering of reaction rates:
        # 0:(n-1) PT mut. rates, (n-1) Seeding rate, (n) PT diag. rate
        # (n+1):2n+1 MT mut. rates, 2n+1 MT diag. rate
        r_rates = jnp.concatenate((
            jnp.exp(log_theta_prim @ state[:n] + b_rates)*(1-state[n])*(1-state[:n]),
            jnp.exp(jnp.dot(pt_d_ef, state[:n]))*(1-state[n]),
            mt_rates), 
            axis=None)

        out_rate = jnp.sum(r_rates)
        obs_order = state[n] + 2*state[-1]
        new_key, sub_key = jrp.split(r_key)
        next_event = jrp.choice(sub_key, inds, p=r_rates/out_rate)
        state = lax.cond(state[n-1] == 1,
                         lambda state, i: state.at[i].set(1),
                         lambda state, i: state.at[jnp.array([i, i+n_d])].set(1),
                         state, next_event)
        return new_key, state, obs_order

    def stop_fun(carry):
        _, state, _ = carry
        # Stop the simulation if either both PT (state[n]) and MT (state[-1]) are diagnosed
        # OR: if PT is diagnosed prior to Seeding (state[n-1])
        return  (state[n] * state[-1] + state[n] * (1-state[n-1])) < 1
    
    b_rates = jnp.diag(log_theta.copy())
    log_theta_prim = log_theta.copy()
    log_theta_prim = log_theta_prim.at[:-1,-1].set(0.)
    n = log_theta.shape[0]
    n_d = n+1
    inds = jnp.arange(0, 2*n+2, dtype=jnp.int8)
    state = jnp.zeros(2*n+2, dtype=jnp.int8)
    obs_order = 0
    _, state, obs_order = lax.while_loop(stop_fun, tumor_dynamics, (rng_key, state, obs_order))
    state = state.reshape((-1,2), order="F").flatten()
    
    return jnp.concatenate((state[:-3], obs_order), axis=None)


def single_traject_jax_cond(
        log_theta: jnp.ndarray,
        pt_d_ef: jnp.ndarray,
        mt_d_ef: jnp.ndarray,
        rng_key: jrp.PRNGKey,
        obs_point: jnp.ndarray, 
        ) -> jnp.ndarray:
    """Sample a single trajectory from the model (log_theta, pt_d_eff, mt_d_eff) conditioned to end in obs_point.
    Currently only conditioning on the PT-genotype of a PT-MT pair is implemented, so this function returns
    a datapoint with the PT-genotype equal to the PT-genotype of obs_point and a variable MT-genotype.

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        pt_d_ef (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        mt_d_ef (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        rng_key (jrp.PRNGKey): JAX random key as returned by jax.random.PRNGkey
        obs_point (jnp.ndarray): Genotype which should be sampled. 
            The simulation is conditioned to end up in this genotype. 
            Currently only conditioning on a PT without seeding is implemented.

    Returns:
        jnp.ndarray: Single realization with genotype obs_point
    """

    def tumor_dynamics(carry):
        r_key, state, obs_order = carry
        # Ordering of reaction rates:
        # 0:(n-1) PT mut. rates, (n-1) Seeding rate, (n) PT diag. rate
        # (n+1):2n+1 MT mut. rates, 2n+1 MT diag. rate
        cond_flag_pt = state[:(n-1)].sum()==obs_point[:(n-1)].sum()

        # Calculate MT reaction rates, all are zero if seeding didn't happen yet
        mt_rates = jnp.concatenate((jnp.exp(log_theta @ state[n_d:-1] + b_rates)*(1-state[-1])*(1-state[n_d:-1]), 
                                    jnp.exp(jnp.dot(mt_d_ef, state[n_d:-1]))*(1-state[-1])), axis=None)
        
        mt_rates = lax.select(state[n-1],
                              mt_rates, 
                              jnp.zeros_like(mt_rates)
                              )
        
        # Ordering of reaction rates:
        # 0:(n-1) PT mut. rates, (n-1) Seeding rate, (n) PT diag. rate
        # (n+1):2n+1 MT mut. rates, 2n+1 MT diag. rate
        r_rates = jnp.concatenate((
            jnp.exp(log_theta_prim @ state[:n] + b_rates)*(1-state[n])*(1-state[:n]),
            jnp.exp(jnp.dot(pt_d_ef, state[:n]))*(1-state[n]) * cond_flag_pt,
            mt_rates), 
            axis=None)
        

        r_rates *= obs_point # conditioning
        out_rate = jnp.sum(r_rates)
        obs_order = state[n] + 2*state[-1]
        new_key, sub_key = jrp.split(r_key)
        next_event = jrp.choice(sub_key, inds, p=r_rates/out_rate)
        state = lax.cond(state[n-1] == 1,
                         lambda state, i: state.at[i].set(1),
                         lambda state, i: state.at[jnp.array([i, i+n_d])].set(1),
                         state, next_event)
        return new_key, state, obs_order

    def stop_fun(carry):
        _, state, _ = carry
        # Stop the simulation if either both PT (state[n]) and MT (state[-1]) are diagnosed
        # OR: if PT is diagnosed prior to Seeding (state[n-1])
        return  (state[n] * state[-1] + state[n] * (1-state[n-1])) < 1
    
    b_rates = jnp.diag(log_theta.copy())
    log_theta_prim = log_theta.copy()
    log_theta_prim = log_theta_prim.at[:-1,-1].set(0.)
    n = log_theta.shape[0]
    n_d = n+1
    inds = jnp.arange(0, 2*n+2, dtype=jnp.int8)
    state = jnp.zeros(2*n+2, dtype=jnp.int8)
    obs_order = 0
    _, state, obs_order = lax.while_loop(stop_fun, tumor_dynamics, (rng_key, state, obs_order))
    state = state.reshape((-1,2), order="F").flatten()
    return jnp.concatenate((state[:-3], obs_order), axis=None)


def simulate_dat_jax(log_theta: jnp.ndarray,
                     pt_d_ef: jnp.ndarray, 
                     mt_d_ef: jnp.ndarray,
                     n_sim: int,
                     obs_point: jnp.ndarray = None,
                     seed: int = 42):
    original_key = jrp.PRNGKey(seed)
    rng_keys = jrp.split(original_key, n_sim)
    if obs_point is None:
        # Perform an uncoditioned Simulation
        dat = vmap(single_traject_jax, (None, None, None, 0), 0)(log_theta, pt_d_ef, mt_d_ef, rng_keys)
    else:
        # Perform a simulation condition to end up in (a set of) state(s) obs_point
        tmp = obs_point[:-1].reshape((-1,2), order="C")
        # Reshape obs_point into the interal simulation layout
        point_mod = jnp.concatenate((tmp[:, 0], # PT-states
                                    obs_point[-1], # Seeding
                                    jnp.array([1]), # PT-diag
                                    tmp[:,1], # MT-states
                                    obs_point[-1], # Seeding
                                    jnp.array([1])), # MT-diag
                                    axis=None
                                    )
        dat = vmap(single_traject_jax_cond, (None, None, None, 0, None), 0)(log_theta, pt_d_ef, mt_d_ef, rng_keys, point_mod)
    return dat


def p_unobs_seeding(log_theta: jnp.ndarray, 
                    pt_d_ef: jnp.ndarray, 
                    mt_d_ef: jnp.ndarray, 
                    obs_point: jnp.ndarray, 
                    n_sim: int, seed: int = 42) -> jnp.ndarray:
    obs_point_mod = obs_point.at[1::2].set(1)
    obs_point_mod = obs_point_mod.at[-1].set(1)
    dat = simulate_dat_jax(log_theta, pt_d_ef, mt_d_ef, n_sim, obs_point_mod, seed)
    dat_pt_first_diag =  dat[dat[:,-1]!=2, -2]
    n_pt_first = dat_pt_first_diag.shape[0]
    n_mt_pres_no_diag = jnp.sum(dat_pt_first_diag)
    return n_mt_pres_no_diag/n_pt_first