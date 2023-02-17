import jax.numpy as jnp
import numpy as np
from ssr_likelihood_jax import _log_prob_coupled, _log_prob_single, R_i_inv_vec, _g_coupled, _g_3_lhs
from ssr_kronvec_jax import obs_inds_2
import vanilla as mhn
import jax.lax as lax

def L1(theta: np.array, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    theta = np.fill_diagonal(theta_, 0)
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    np.fill_diagonal(theta_, 0)
    return theta_ / np.sqrt(theta_**2 + eps)


def log_prob_single(log_theta: jnp.array, dat: jnp.array, lam1: float, penal: float) -> jnp.array:
    """
    calculates the likelihood to see all single tumors in dat
    Args:
        dat (jnp.array): dataset containiong all single tumors
        log_theta (jnp.array): theta matrix with logarithmic entries
        lam1 (float): Rate of first sampling

    Returns:
        jnp.array: Likelihood
    """
    n = int(np.sqrt(log_theta.shape[0])-1)
    log_theta = log_theta.reshape((n+1, n+1))
    l1 = L1(log_theta)
    log_theta = jnp.array(log_theta)
    score = 0.0
    for i in range(dat.shape[0]):
        marg_met = bool(dat.at[i, -1].get())
        if marg_met:
            state_obs = dat.at[i, 0:2*n+1:2].get()
            log_theta = log_theta.at[0:n, -1].set(0.0)
        else:
            state_obs = jnp.append(dat.at[i, 1:2*n+1:2].get(), dat.at[i, -2].get())
        p0 = jnp.zeros(2**int(state_obs.sum()))
        p0 = p0.at[0].set(1.0)
        score += _log_prob_single(log_theta, lam1, state_obs, p0)
    score = np.array(score)/dat.shape[0]
    return -(score - penal * l1)


def log_prob_coupled(log_theta: jnp.array, dat: jnp.array, lam1: float, lam2: float, penal: float) -> jnp.array:
    """
    Calulates the likelihood of dat
    Args:
    dat (jnp.array): dataset
    log_theta (jnp.array): logarithmic theta
    lam1 (float): Rate of first sampling
    lam2 (float): rate of second sampling

    Returns:
    jnp.array: likelihood
    """
    n = int(np.sqrt(log_theta.shape[0])-1)
    log_theta = log_theta.reshape((n+1, n+1))
    l1 = L1(log_theta)
    log_theta = jnp.array(log_theta)
    score = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        prim_first = bool(dat.at[i, -1].get())
        if prim_first:
            latent_state = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
        else:    
            latent_state = state.at[0:2*n+1:2].get()
        latent_dist = jnp.zeros(2**int(latent_state.sum()))
        p0 = jnp.zeros(2**int(state.sum()))
        p0 = p0.at[0].set(1.0)
        score += _log_prob_coupled(log_theta, lam1, lam2, state, p0, 
        latent_dist, latent_state, prim_first)
    score = np.array(score)/dat.shape[0]
    return -(score - penal * l1)


def grad_coupled(log_theta: np.array, dat: jnp.array, lam1:float, lam2: float, penal: float) -> jnp.array:
    """
    Calculates the gradient for all coupled datapoints in a dataset
    Args:
        dat (jnp.array): dataset containing the tumor genotypes as rows
        log_theta (jnp.array): theta matrix with logarithmic entries
        lam1 (float): Rate of first sampling
        lam2 (float): rate of second sampling

    Returns:
        jnp.array: gradient
    """
    n = int(np.sqrt(log_theta.shape[0])-1)
    log_theta = log_theta.reshape((n+1, n+1))
    l1 = L1_(log_theta)
    log_theta = jnp.array(log_theta)
    g = jnp.zeros((n+1,n+1))
    for i in range(dat.shape[0]):
        prim_first = bool(dat.at[i, -1].get())
        state = dat.at[i, 0:2*n+1].get()
        if prim_first:
            second_obs = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
        else:    
            second_obs = state.at[i, 0:2*n+1:2].get()
        
        # Calculate Joint dist at t1
        p0 = jnp.zeros(2**int(state.sum()))
        p0 = p0.at[0].set(1.0)
        pTh1 = lam1 * R_i_inv_vec(log_theta, p0, lam1,  state, transpose = False)
        
        # Get indices of latent_states at t1/t2
        latent_states = obs_inds_2(pTh1, state, prim_first)
        latent_size = int(2**second_obs.sum())
        
        # Get observed dist at t1
        latent_inds = jnp.where(latent_states, size=latent_size)[0]
        pTh1_obs = pTh1.at[latent_inds].get()
        nk = pTh1_obs.sum()
        pTh1_obs /= nk
        
        log_theta_2 = lax.cond(prim_first,
                                lambda th: th,
                                lambda th: th.at[:n, -1].set(0.0),
                                operand = log_theta)
        g_2, pTh2 = mhn.gradient(log_theta_2, lam2, second_obs, pTh1_obs)
        # Derivative of constant is 0.
        g_2 = lax.cond(prim_first,
                        lambda g: g,
                        lambda g: g.at[:n, -1].set(0.0),
                        operand = g_2)
        
        q_big = _g_3_lhs(log_theta_2, pTh1, pTh2, latent_inds, second_obs, lam2)
        g += g_2 + _g_coupled(log_theta, state, q_big, latent_states, pTh1, nk, lam1)
    g = np.array(g) / dat.shape[0]
    return -(g - penal * l1).flatten()

def grad_single(log_theta:jnp.array, dat: jnp.array, lam1:float, penal: float) -> jnp.array:
    """
    returns gradient of the likelihood for all single datapoints 
    Args:
        dat (jnp.array): dataset, containing only single tumors
        log_theta (jnp.array): theta matrix with logarithmic entries
        lam1 (float): Rate of first sampling

    Returns:
        jnp.array: Gradient
    """
    n = int(np.sqrt(log_theta.shape[0])-1)
    log_theta = log_theta.reshape((n+1, n+1))
    l1 = L1_(log_theta)
    log_theta = jnp.array(log_theta).reshape((n+1, n+1))
    g = jnp.zeros((n+1,n+1))
    for i in range(dat.shape[0]):
        marg_met = bool(dat.at[i, -1].get())
        if marg_met:
            state_obs = dat.at[i, 0:2*n+1:2].get()
            log_theta = log_theta.at[0:n, -1].set(0.0)
        else:
            state_obs = jnp.append(dat.at[i, 1:2*n+1:2].get(), dat.at[i, -2].get())
        p0 = jnp.zeros(2**int(state_obs.sum()))
        p0 = p0.at[0].set(1.0)
        g += mhn.gradient(log_theta, lam1, state_obs, p0)[0]
    g = np.array(g / dat.shape[0])
    return -(g - penal * l1).flatten()


# Only included for testing purposes
# ToDo: Remove later
#def grad_coupled(log_theta: np.array, dat: jnp.array, lam1:float, lam2: float, penal: float) -> jnp.array:
#    """
#    Calculates the gradient for all coupled datapoints in a dataset
#    Args:
#        dat (jnp.array): dataset containing the tumor genotypes as rows
#        log_theta (jnp.array): theta matrix with logarithmic entries
#        lam1 (float): Rate of first sampling
#        lam2 (float): rate of second sampling
#
#    Returns:
#        jnp.array: gradient
#    """
#    n = int(np.sqrt(log_theta.shape[0])-1)
#    log_theta = log_theta.reshape((n+1, n+1))
#    l1 = L1_(log_theta)
#    log_theta = jnp.array(log_theta)
#    g = jnp.zeros((n+1,n+1))
#    for i in range(dat.shape[0]):
#        prim_first = bool(dat.at[i, -1].get())
#        state = dat.at[i, 0:2*n+1].get()
#        if prim_first:
#            latent_state_1 = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
#            latent_state_2 = state.at[0:2*n+1:2].get()
#
#        else:    
#            latent_state_1 = state.at[i, 0:2*n+1:2].get()
#            latent_state_2 = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
#        latent_dist_1 = jnp.zeros(2**int(latent_state_1.sum()))
#        latent_dist_2 = jnp.zeros(2**int(latent_state_2.sum()))
#        p0 = jnp.zeros(2**int(state.sum()))
#        p0 = p0.at[0].set(1.0)
#        g += _grad_coupled(log_theta, lam1, lam2, state, p0, 
#               latent_dist_1, latent_dist_2, latent_state_1, prim_first)
#    g = np.array(g) / dat.shape[0]
#    return -(g - penal * l1).flatten()
#
#
#def grad_coupled_2(log_theta: np.array, dat: jnp.array, lam1:float, lam2: float, penal: float) -> jnp.array:
#    """
#    Calculates the gradient for all coupled datapoints in a dataset
#    Args:
#        dat (jnp.array): dataset containing the tumor genotypes as rows
#        log_theta (jnp.array): theta matrix with logarithmic entries
#        lam1 (float): Rate of first sampling
#        lam2 (float): rate of second sampling
#
#    Returns:
#        jnp.array: gradient
#    """
#    n = int(np.sqrt(log_theta.shape[0])-1)
#    log_theta = log_theta.reshape((n+1, n+1))
#    l1 = L1_(log_theta)
#    log_theta = jnp.array(log_theta)
#    g = jnp.zeros((n+1,n+1))
#    for i in range(dat.shape[0]):
#        prim_first = bool(dat.at[i, -1].get())
#        state = dat.at[i, 0:2*n+1].get()
#        if prim_first:
#            second_obs = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
#        else:    
#            second_obs = state.at[i, 0:2*n+1:2].get()
#        
#        # Calculate Joint dist at t1
#        p0 = jnp.zeros(2**int(state.sum()))
#        p0 = p0.at[0].set(1.0)
#        pTh1 = lam1 * R_i_inv_vec(log_theta, p0, lam1,  state, transpose = False)
#        
#        # Get indices of latent_states at t1/t2
#        latent_states = obs_inds_2(pTh1, state, prim_first)
#        latent_size = int(2**second_obs.sum())
#        
#        # Get observed dist at t1
#        latent_inds = jnp.where(latent_states, size=latent_size)[0]
#        pTh1_obs = pTh1.at[latent_inds].get()
#        nk = pTh1_obs.sum()
#        pTh1_obs /= nk
#        
#        # Calculate the gradients
#        g_1 = _g_coupled_1(log_theta, lam1, state, pTh1, latent_states, nk)
#        
#        log_theta_2 = lax.cond(prim_first,
#                                lambda th: th,
#                                lambda th: th.at[:n, -1].set(0.0),
#                                operand = log_theta)
#        g_2, pTh2 = mhn.gradient(log_theta_2, lam2, second_obs, pTh1_obs)
#        # Derivative of constant is 0.
#        g_2 = lax.cond(prim_first,
#                        lambda g: g,
#                        lambda g: g.at[:n, -1].set(0.0),
#                        operand = g_2)
#        
#        q = jnp.zeros_like(pTh2)
#        q = q.at[-1].set(1/pTh2.at[-1].get())
#        q = mhn.R_inv_vec(log_theta_2, q, lam2, second_obs, transpose = True)
#        q_big = jnp.zeros_like(pTh1)
#        q_big =  q_big.at[latent_inds].set(q)
#        
#        g_3 = _g_coupled_3(log_theta, state, q_big, latent_states, pTh1, nk, lam1)
#        g_4 = _g_coupled_4(log_theta, latent_states, pTh1, state, lam1, nk)
#
#        g += g_1 + g_2 + g_3 + g_4
#    g = np.array(g) / dat.shape[0]
#    return -(g - penal * l1).flatten()