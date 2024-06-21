import numpy as np
import jax.numpy as jnp
from jax import vmap, jit, lax
from typing import Tuple
import jax.random as jrp
from functools import partial

@jit
def single_traject(
        log_theta: jnp.ndarray,
        pt_d_ef: jnp.ndarray,
        mt_d_ef: jnp.ndarray,
        rng_key: jrp.PRNGKey,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Sample a trajectory to first diagnosis or from first diagnosis to second diagnosis

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        pt_d_ef (jnp.ndarray): Log. effects of mutations in the primary tumor on its rate of observation
        mt_d_ef (jnp.ndarray): Log. effects of mutations in the metastasis on its rate of observation
        n_sim (int): Number of datapoints to simulate
        original_key (jrp.PRNGKey): Original random key to use for simulations

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Final tumor genotypes, tumor trajectory
    """

    def tumor_dynamics(carry):
        r_key, state, traject, i = carry

        # 0:(n-1) PT mut. rates, (n-1) Seeding rate, (n) PT diag. rate
        pt_rates = jnp.concatenate((jnp.exp(log_theta_prim @ state[:n] + b_rates)*(1-state[:n]),
                                   jnp.exp(jnp.dot(pt_d_ef, state[:n]))*(1-state[n])), axis=None)
        # Rates are zero if PT is diagnosed
        pt_rates = lax.select(state[n],
                              jnp.zeros_like(pt_rates), 
                              pt_rates
                              )
        # 0:(n-1) MT mut. rates, (n-1) Seeding rate, (n) MT diag. rate
        mt_rates = jnp.concatenate((jnp.exp(log_theta @ state[n_d:-1] + b_rates)*(1-state[n_d:-1]), 
                                    jnp.exp(jnp.dot(mt_d_ef, state[n_d:-1]))*(1-state[-1])), axis=None)
        
        mt_rates = lax.select(state[n-1]*(1-state[-1]),
                              mt_rates, 
                              jnp.zeros_like(mt_rates)
                              )
        r_rates = jnp.concatenate((pt_rates, mt_rates))

        out_rate = jnp.sum(r_rates)
        new_key, sub_key = jrp.split(r_key)
        next_event = jrp.choice(sub_key, inds, p=r_rates/out_rate)
        state = lax.cond(state[n-1] == 1,
                         lambda state, i: state.at[i].set(1),
                         lambda state, i: state.at[jnp.array([i, i+n_d])].set(1),
                         state, next_event)
        traject = traject.at[i].set(next_event)
        i += 1
        return new_key, state, traject, i

    def stop_fun(carry):
        _, state, _, _ = carry
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
    traject = jnp.ones(2*n+2, dtype=jnp.int8)*(-99)
    ind = 0
    _, state, traject, _ = lax.while_loop(stop_fun, tumor_dynamics, (rng_key, state, traject, ind))
    state = state.reshape((-1,2), order="F").flatten() 
    return state[:-3], traject


def simulate_orders(log_theta: jnp.ndarray,
                     pt_d_ef: jnp.ndarray, 
                     mt_d_ef: jnp.ndarray,
                     n_sim: int,
                     original_key: jrp.PRNGKey,
                     ) -> jnp.ndarray:
    """Simulate a full dataset containing fully observed tumor trajectories

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        pt_d_ef (jnp.ndarray): Log. effects of mutations in the primary tumor on its rate of observation
        mt_d_ef (jnp.ndarray): Log. effects of mutations in the metastasis on its rate of observation
        n_sim (int): Number of datapoints to simulate
        original_key (jrp.PRNGKey): Original random key to use for simulations

    Returns:
        jnp.ndarray: Dataset of tumor trajectories, each row corresponds to an observation.
            Events are encoded as follows (n denotes the total number of events):
            0:(n-1)  Genomic events in the PT
            n-1      Seeding event
            n        Observation of the PT
            (n+1):2n Genomic events in the MT
            2n       Seeding
            2n+1     Observation of the MT
    """
    rng_keys = jrp.split(original_key, n_sim)
    _, orders = vmap(single_traject, (None, None, None, 0), 0)(log_theta, pt_d_ef, mt_d_ef, rng_keys)
    return orders


def simulate_dat(log_theta: jnp.ndarray,
                  pt_d_ef: jnp.ndarray, 
                  mt_d_ef: jnp.ndarray,
                  n_sim: int,
                  original_key: jrp.PRNGKey,
                  ) -> jnp.ndarray:
    """Simulate a full dataset containing fully observed tumor trajectories

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        pt_d_ef (jnp.ndarray): Log. effects of mutations in the primary tumor on its rate of observation
        mt_d_ef (jnp.ndarray): Log. effects of mutations in the metastasis on its rate of observation
        n_sim (int): Number of datapoints to simulate
        original_key (jrp.PRNGKey): Original random key to use for simulations

    Returns:
        jnp.ndarray: Dataset of tumor trajectories, each row corresponds to an observation.
            Events are encoded as follows (n denotes the total number of events):
            0:(n-1)  Genomic events in the PT
            n-1      Seeding event
            n        Observation of the PT
            (n+1):2n Genomic events in the MT
            2n       Seeding
            2n+1     Observation of the MT
    """
    def get_obs_order(dat, pt_obs, mt_obs):
        ai = jnp.where(dat==pt_obs, size=1)[0]
        bi = jnp.where(dat==mt_obs, size=1)[0]
        return lax.select(ai<bi, 1, 2)

    rng_keys = jrp.split(original_key, n_sim)
    n = log_theta.shape[0]
    obs, orders = vmap(single_traject, (None, None, None, 0), 0)(log_theta, pt_d_ef, mt_d_ef, rng_keys)
    paired_inds = jnp.where(obs[:,-1]==1)[0]
    obs_order = jnp.zeros(n_sim, dtype=jnp.int8).reshape((-1,1))
    oo = vmap(get_obs_order, (0, None, None), 0)(orders[paired_inds,:], n, 2*n+1).flatten()
    obs_order = obs_order.at[paired_inds,0].set(oo)
    return jnp.hstack((obs, obs_order))


def _extract_bse(traject: jnp.ndarray, n: int, seeding_num: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Extract events that occured before the seeding (bse) from an trajectory. This function converts 
    a trajectory effectively in two bit strings.In the first bit string only the pre seeding events are 
    set to 1, while in the second bit string all events active in the trajectory are set to 1.


    Args:
        traject (jnp.ndarray): a trajectory as returne by single_traject_order
        n (int): Total number of events
        seeding_num (int): Index of the seeding event in the event list

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Preseeding events, total events
    """
    bsc = jnp.zeros(n, dtype=jnp.int8)
    tc = jnp.zeros(2*n+2, dtype=jnp.int8)
    stride =  n+1

    def loop_body(val):
        psf, bsc, tc, ind = val        
        bsc, tc = lax.cond(psf,
                           # Update preseeding counts, and total counts
                           # Event occured jointly in PT and MT -> update both counts
                           lambda t, bsc, tc: (bsc.at[t[ind]].set(1), 
                                               tc.at[jnp.array([t[ind], t[ind]+stride])].set(1)),
                           # Post seeding counts, event occured in PT or MT only
                           lambda t, bsc, tc: (bsc.at[t[ind]].set(0), 
                                               tc.at[t[ind]].set(1)),
                           traject, bsc, tc)
        
        # Check if seeding occurs at current index
        psf = lax.cond(traject[ind] == seeding_num,
                       lambda x: x*0,
                       lambda x: x*1,
                       psf)
        ind += 1
        return (psf, bsc, tc, ind)
    
    def cond_fun(val):
        _, _, _, ind = val
        # Stop when end of trajectory is reached
        # Indicated by padding values of -99
        return traject[ind] != -99
    
    _, bsc, tc, _ = lax.while_loop(cond_fun, loop_body, (1, bsc, tc, 0))
    return bsc, tc


@partial(jit, static_argnames=["n"])
def extract_bse(traject: jnp.ndarray, n: int, seeding_num: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """ Extract events that occured before the seeding (bse) from an trajectory. This function converts 
    a trajectory effectively in two bit strings.In the first bit string only the pre seeding events are 
    set to 1, while in the second bit string all events active in the trajectory are set to 1.
    IMPORTANT: For trajectories without seeding, bitstrings containing only zeros are returned.


    Args:
        traject (jnp.ndarray): a trajectory as returne by single_traject_order
        n (int): Total number of events
        seeding_num (int): Index of the seeding event in the event list

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: Preseeding events, total events
    """
    return lax.cond(jnp.any(traject == seeding_num),
                    lambda x: _extract_bse(x, n, seeding_num),
                    lambda x: (jnp.zeros(n, dtype=jnp.int8), 
                               jnp.zeros(2*n+2, dtype=jnp.int8)),
                    traject)


def preseeding_probs(dat: jnp.ndarray, n: int, seeding_num: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Calculate the probability for each mutation to have happend prior to the seeding, conditional on the mutations 
    and the seeding being active in the tumor. The function returns two vectors, one conditioned on the mutation 
    being observed in the PT and one conditioned on the mutation being active in the MT.

    Args:
        dat (jnp.ndarray): Dataset containing tumor trajectories, such as returned by simulate orders
        n (int): Total number of events
        seeding_num (int): Index of the seeding event in the event list

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]: P(t_mut < t_seed | PT(mut)=1, seed=1), P(t_mut < t_seed | MT(mut)=1, seed=1) 
    """
    bsc, tc = vmap(extract_bse, in_axes=(0, None, None))(dat, n, seeding_num)
    bsp_pt_cond = jnp.sum(bsc[:,:-1], axis=0)/jnp.sum(tc[:, :seeding_num], axis=0)
    bsp_mt_cond = jnp.sum(bsc[:,:-1], axis=0)/jnp.sum(tc[:, (seeding_num+2):-2], axis=0)
    return bsp_pt_cond, bsp_mt_cond