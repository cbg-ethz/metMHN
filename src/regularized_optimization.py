import jax.numpy as jnp
import numpy as np
import ssr_likelihood_jax as ssr
import vanilla as mhn
from ssr_kronvec_jax import obs_inds_2

def L1(theta: np.array, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    theta_[np.diag_indices(theta.shape[0])] = 0.
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    theta_[np.diag_indices(theta.shape[0])] = 0.
    return np.append(theta_ / np.sqrt(theta_**2 + eps), [0., 0.]).flatten()


def lp_prim_only(log_theta: jnp.array, dat: jnp.array, lam1: jnp.array, n: int) -> jnp.array:
    """Calculates the marginal likelihood of observing an unpaired primary tumor at t_1

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        jnp.array: log(P(state))
    """
    score = 0.0
    for i in range(dat.shape[0]):
        state_obs = dat.at[i, 0:2*n+1:2].get()
        m =state_obs.sum().astype(int)
        p0 = jnp.zeros(2**m)
        p0 = p0.at[0].set(1.0)
        score += ssr._lp_prim_obs(log_theta, lam1, state_obs, p0)
    return score


def lp_coupled(log_theta: jnp.array, dat: jnp.array, lam1: jnp.array, lam2: jnp.array, n: int) -> np.array:
    """Calculates the marginal likelihood of observing a coupled prim-met pair

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        jnp.array: log(P(state))
    """
    score = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        latent_state = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
        latent_dist = jnp.zeros(2**(latent_state.sum().astype(int) - 1))
        m = state.sum().astype(int)
        p0 = jnp.zeros(2**m)
        p0 = p0.at[0].set(1.0)
        score += ssr._lp_coupled(log_theta, lam1, lam2, state, p0, 
                                   latent_dist, latent_state)
    return score


def lp_met_only(log_theta: jnp.array, dat: jnp.array, lam1: jnp.array, lam2: jnp.array, n: int) -> jnp.array:
    """Calculates the marginal likelihood of observing an unpaired metastasis at t_2

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        jnp.array: log(P(state))
    """
    # Unpack parameters
    score = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        state_obs = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
        m = state_obs.sum().astype(int)
        p0 = jnp.zeros(2**m)
        p0 = p0.at[0].set(1.0)
        score += ssr._lp_met_obs(log_theta, lam1, lam2, state_obs, p0)
    return score


def log_lik(params: np.array, dat_prim_no_met: jnp.array, dat_prim_met: jnp.array, dat_coupled: jnp.array, 
            dat_met_only: jnp.array, penal: float, weights=jnp.array([1., 1., 1., 1.])) -> np.array:
    """Calculates the negative log. likelihood 

    Args:
        params (np.array): n**2 + 2 dimensional array holding the parameters of the model 
        dat_prim_met (jnp.arrray): dataset containing prim tumors only, that generated no mets.
        dat_prim_no_met (jnp.array): dataset containing prim tumors only, that generated a mets. 
        dat_coupled (jnp.array): dataset containing coupled prim-met. pairs
        dat_met_only (jnp.array): dataset containg only met_tumors
        penal (float): weight of L1-penalization
        weights (jnp.array, optional): weights to correct for sampling biases. Defaults to jnp.array([0.25, 0.25, 0.25, 0.25]).

    Returns:
        np.array: - log. likelihood
    """
    log_theta = params[0:-2]
    n = jnp.sqrt(log_theta.shape[0]).astype(int) - 1
    log_theta = log_theta.reshape((n+1, n+1))
    l1 = L1(log_theta)
    log_theta = jnp.array(log_theta)
    
    lam1 = jnp.exp(params[-2])
    lam2 = jnp.exp(params[-1])

    scores = jnp.zeros(4)
    scores = scores.at[0].set(lp_prim_only(log_theta, dat_prim_no_met, lam1, n)) # Met seeded, not observed
    scores = scores.at[1].set(lp_prim_only(log_theta, dat_prim_met, lam1, n)) # No seeding
    scores = scores.at[2].set(lp_coupled(log_theta, dat_coupled, lam1, lam2, n))
    scores = scores.at[3].set(lp_met_only(log_theta, dat_met_only, lam1, lam2, n))
    
    w = jnp.array([dat_prim_no_met.shape[0], dat_prim_met.shape[0], dat_coupled.shape[0], dat_met_only.shape[0]])
    w = jnp.dot(w, weights)
    ret = jnp.dot(scores/w, weights)
    return(-np.array(ret) + penal * l1)


def grad_coupled(log_theta: jnp.array, dat: jnp.array, lam1: jnp.array, lam2: jnp.array, n: int) -> jnp.array:
    """gradient of lp_coupled
    
    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        jnp.array: d log(P(state))/ d theta_ij
    """
    # Unpack parameters
    g = jnp.zeros((n+1,n+1), dtype=float)
    dlam1 = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        second_obs = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
        
        # Calculate joint dist at t1
        p0 = jnp.zeros(2**state.sum())
        p0 = p0.at[0].set(1.0)
        pTh1 = lam1 * ssr.R_i_inv_vec(log_theta, p0, lam1,  state, transpose = False)
        
        # Get indices of latent_states at t1
        latent_states = obs_inds_2(pTh1, state, True)
        latent_size = 2**(jnp.sum(second_obs).astype(int)-1)
        
        # Get observed dist at t1
        latent_inds = jnp.where(latent_states, size=latent_size)[0]
        pTh1_obs = pTh1.at[latent_inds].get()
        pTh1_obs = jnp.append(jnp.zeros_like(pTh1_obs), pTh1_obs)
        nk = jnp.sum(pTh1_obs)
        pTh1_obs = pTh1_obs.at[:].multiply(1/nk)
        g_2, pTh2 = mhn.gradient(log_theta, lam2, second_obs, pTh1_obs)
        q_big = ssr._g_3_lhs(log_theta, pTh1, pTh2, latent_inds, second_obs, lam2)
        
        dtheta, dlam = ssr._g_coupled(log_theta, state, q_big, latent_states, pTh1, nk, lam1)
        g += dtheta + g_2 
        dlam1 += dlam
    return jnp.append(g.flatten(), jnp.array([dlam1, 0.0]))

def grad_prim_only(log_theta:jnp.array, dat: jnp.array, lam1: jnp.array, n: int) -> jnp.array:
    """gradient of lp_prim
    
    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        jnp.array: d log(P(state))/ d theta_ij
    """
    g = jnp.zeros((n+1,n+1))
    dlam1 = 0.0
    for i in range(dat.shape[0]):
        state_obs = dat.at[i, 0:2*n+1:2].get()
        p0 = jnp.zeros(2**int(state_obs.sum()))
        p0 = p0.at[0].set(1.0)
        g_, dlam = ssr._grad_prim_obs(log_theta, state_obs, p0, lam1)
        g += g_
        dlam1 += dlam
    return jnp.append(g.flatten(), jnp.array([dlam1, 0.0]))

def grad_met_only(log_theta: jnp.array, dat: jnp.array, lam1: jnp.array, lam2: jnp.array, n: int) -> jnp.array:
    """ Gradient of lp_met_only

      Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        jnp.array: d log(P(state))/ d theta_ij
    """
    # Unpack parameters
    g = jnp.zeros((n+1,n+1), dtype=float)
    dlam1 = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        state_obs = jnp.append(state.at[1:2*n+1:2].get(), state.at[-1].get())
        p0 = jnp.zeros(2**state_obs.sum())
        p0 = p0.at[0].set(1.0)
        g_, dlam = ssr._grad_met_obs(log_theta, state_obs, p0, lam1, lam2)
        g += g_
        dlam1 += dlam
    return jnp.append(g.flatten(), jnp.array([dlam1, 0.0]))


def grad(params, dat_prim_no_met, dat_prim_met, dat_coupled, dat_met_only, penal, weights = jnp.array([0.25, 0.25, 0.25, 0.25])) -> jnp.array:
    """Calculates the gradient of log_lik wrt. to all log(\theta_ij) and wrt. \lambda_1

    Args:
        params (np.array): n**2 + 2 dimensional array holding the parameters of the model 
        dat_prim_met (jnp.arrray): dataset containing prim tumors only, that generated no mets.
        dat_prim_no_met (jnp.array): dataset containing prim tumors only, that generated a mets. 
        dat_coupled (jnp.array): dataset containing coupled prim-met. pairs
        dat_met_only (jnp.array): dataset containg only met_tumors
        penal (float): weight of L1-penalization
        weights (jnp.array, optional): weights to correct for sampling biases. Defaults to jnp.array([0.25, 0.25, 0.25, 0.25]).

    Returns:
        np.array: n**2 + 2 dimensional gradient vector
    """
    # Unpack parameters
    lam1 = jnp.exp(params[-2])
    lam2 = jnp.exp(params[-1])
    n = int(jnp.sqrt(params.shape[0] - 3))
    log_theta = params[0:-2].reshape((n+1, n+1))
    l1 = L1_(log_theta)

    # Transfer theta to the device
    log_theta = jnp.array(log_theta)
    g = jnp.zeros(((n+1)**2+2, 4))
    nDat = dat_prim_no_met.shape[0] + dat_prim_met.shape[0] + dat_coupled.shape[0] + dat_met_only.shape[0]
    g = g.at[:,0].set(grad_prim_only(log_theta, dat_prim_no_met, lam1, n))
    g = g.at[:,1].set(grad_prim_only(log_theta, dat_prim_met, lam1, n))
    g = g.at[:,2].set(grad_coupled(log_theta, dat_coupled, lam1, lam2, n))
    g = g.at[:,3].set(grad_met_only(log_theta, dat_met_only, lam1, lam2, n))
    w = jnp.array([dat_prim_no_met.shape[0], dat_prim_met.shape[0], dat_coupled.shape[0], dat_met_only.shape[0]])
    w = jnp.dot(w, weights)
    ret =  jnp.dot(g/w, weights)
    return np.array(-ret + penal * l1)
