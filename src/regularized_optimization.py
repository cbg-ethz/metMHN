import jax.numpy as jnp
import numpy as np
from jx import likelihood as ssr
from jx import vanilla as mhn
from jx.kronvec import obs_states

def L1(theta: np.array, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    theta_[np.diag_indices(theta.shape[0])] = 0.
    #theta_[:, -1] = 0.
    return np.sum(np.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    theta_[np.diag_indices(theta.shape[0])] = 0.
    #theta_[:, -1] = 0.
    return np.append(theta_ / np.sqrt(theta_**2 + eps), [0., 0.]).flatten()


def L2(th_in: np.array, lam1: np.array) -> np.array:
    th_diag = np.diagonal(th_in).copy()
    ret = np.exp(2 * th_diag)
    return np.sum(ret) + np.exp(2*lam1)


def L2_(th_in: np.array, lam1: np.array) -> np.array:
    th_diag = np.diagonal(th_in).copy()
    theta_ = 2 * np.exp(2 * th_diag)
    return np.append(np.diag(theta_).flatten(), [2 * np.exp(2 * lam1), 0.])

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


def lp_coupled(log_theta: jnp.array, dat: jnp.array, lam1: jnp.array, lam2: jnp.array, n: int) -> jnp.array:
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


def log_lik(params: np.array, dat_prim: jnp.array, dat_coupled: jnp.array, 
            n: int, penal: float, perc_met: float = None) -> np.array:
    """Calculates the negative log. likelihood 

    Args:
        params (np.array): n**2 + 2 dimensional array holding the parameters of the model 
        dat_prim (jnp.array): Dataset containing prim tumors only, that generated a mets. 
        dat_coupled (jnp.array): Dataset containing coupled prim-met. pairs
        n (int): Number of genomic events
        penal (float): Weight of L1-penalization
        perc_met (float): Weights to correct for sampling biases. Defaults to 0.5

    Returns:
        np.array: - log. likelihood
    """
    log_theta = params[0:-2]
    log_theta = log_theta.reshape((n+1, n+1))
    l1 = L1(log_theta)

    if perc_met == None:
        perc_met = n_met/(n_prim + n_met)

    log_theta = jnp.array(log_theta)
    lam1 = jnp.exp(params[-2])
    lam2 = jnp.exp(params[-1])

    n_prim =  jnp.max(jnp.array([1., dat_prim.shape[0]]))
    n_met =  jnp.max(jnp.array([1., dat_coupled.shape[0]]))

    score_prim = lp_prim_only(log_theta, dat_prim, lam1, n)
    score_met = lp_coupled(log_theta, dat_coupled, lam1, lam2, n)
    score = (1 - perc_met) * score_prim/n_prim+ perc_met * score_met/n_met
    return(-np.array(score) + penal * l1)




def grad_coupled(log_theta: jnp.array, theta_prim: jnp.array, dat: jnp.array, lam1: jnp.array, lam2: jnp.array, n: int) -> tuple[jnp.array, jnp.array]:
    """_summary_

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): Rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        tuple[jnp.array, jnp.array]: (likelihood, gradient)
    """
    # Unpack parameters
    g = jnp.zeros((n+1,n+1), dtype=float)
    score = 0.0
    dlam1 = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        n_prim = int(state.at[::2].get().sum())
        n_met = int(state.at[1::2].get().sum() + 1)      
        lik, dtheta, dlam = ssr._g_coupled(log_theta, theta_prim, state, n_prim, n_met, lam1, lam2)
        g += dtheta
        dlam1 += dlam
        score += lik
    return score, jnp.append(g.flatten(), jnp.array([dlam1, 0.0]))


def grad_prim_only(log_theta:jnp.array, dat: jnp.array, lam1: jnp.array, n: int) -> tuple[jnp.array, jnp.array]:
    """Gradient of lp_prim
    
    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        dat (jnp.array): Dataset
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): Rate \lambda_2 of seccond sampling
        n (int): Number of genomic events

    Returns:
        tuple[jnp.array, jnp.array]: (log(P(state), d log(P(state))/ d theta)
    """
    g = jnp.zeros((n+1,n+1))
    score = 0.
    dlam1 = 0.0
    for i in range(dat.shape[0]):
        state_obs = dat.at[i, 0:2*n+1:2].get()
        p0 = jnp.zeros(2**int(state_obs.sum()))
        p0 = p0.at[0].set(1.0)
        s, g_, dlam = ssr._grad_prim_obs(log_theta, state_obs, p0, lam1)
        g += g_
        dlam1 += dlam
        score += s
    g = g.at[:-1, -1].set(0.)
    return score, jnp.append(g.flatten(), jnp.array([dlam1, 0.0]))

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
    score = 0.0
    dlam1 = 0.0
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n+1].get()
        state_obs = jnp.append(state.at[1:2*n+1:2].get(), 1)
        p0 = jnp.zeros(2**state_obs.sum())
        p0 = p0.at[0].set(1.0)
        s, g_, dlam = ssr._grad_met_obs(log_theta, state_obs, p0, lam1, lam2)
        g += g_
        dlam1 += dlam
        score += s
    return score, jnp.append(g.flatten(), jnp.array([dlam1, 0.0]))


def value_grad(params: np.array, dat_prim_only: jnp.array, dat_coupled: jnp.array, dat_prim_met: jnp.array, dat_met_only: jnp.array, 
               n: int, penal1: float, penal2: float, perc_met: float = None) -> tuple[np.array, np.array]:
    """Calculates the gradient of log_lik wrt. to all log(\theta_ij) and wrt. \lambda_1

    Args:
        params (np.array): n**2 + 2 dimensional array holding the parameters of the model 
        dat_prim (jnp.arrray): dataset containing prim tumors only, that generated no mets.
        dat_coupled (jnp.array): dataset containing coupled prim-met. pairs
        n (int): Number of genomic events
        penal (float): Weight of L1-penalization
        perc_met (float): Weights to correct for sampling biases. Defaults to 0.5

    Returns:
        tuple[np.array, np.array]: (likelihood, gradient)
    """
    # Unpack parameters
    n_prim_only =  dat_prim_only.shape[0]
    n_coupled = dat_coupled.shape[0]
    n_prim_met = dat_prim_met.shape[0]
    n_met_only = dat_met_only.shape[0]
    n_met = n_coupled + n_prim_met + n_met_only
    log_theta = params[0:-2].reshape((n+1, n+1))
       
    # Penalties and their derivatives
    l1 = L1(log_theta)
    l1_ = L1_(log_theta)
    l2 = L2(log_theta, params[-2])
    l2_ = L2_(log_theta, params[-2])

    # Transfer theta to the device
    log_theta = jnp.array(log_theta)
    lam1 = jnp.exp(params[-2])
    lam2 = jnp.exp(params[-1])

    if perc_met == None:
        perc_met = 1 - n_prim_only/(n_prim_only + n_met)

    log_theta_prim = log_theta.copy()
    log_theta_prim = log_theta_prim.at[:-1,-1].set(0.)
    score_prim, g_prim = grad_prim_only(log_theta, dat_prim_only, lam1, n)
    score_coupled, g_coupled = grad_coupled(log_theta, log_theta_prim, dat_coupled, lam1, lam2, n)
    #score_prim_met, g_prim_met = grad_prim_only(log_theta_prim, dat_prim_met, lam1, n)
    #score_met, g_met  = grad_met_only(log_theta, dat_met_only, lam1, lam2, n)
    score = (1 - perc_met) * score_prim/n_prim_only + perc_met *\
        (score_coupled/n_coupled) #+ score_prim_met + score_met)/n_met
    g = (1 - perc_met) * g_prim/n_prim_only + perc_met *\
        (g_coupled/n_coupled) #+ g_prim_met + g_met)/n_met
    return np.array(-score + penal1 * l1 + penal2 * l2), np.array(-g + penal1 * l1_ + penal2 * l2_)
