import jax.numpy as jnp
import numpy as np
from jx import likelihood as ssr
from jx import vanilla as mhn
from jx.kronvec import obs_states, diagnosis_theta

def L1(theta: jnp.array, eps: float = 1e-05) -> float:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return jnp.sum(jnp.sqrt(theta_**2 + eps))


def L1_(theta: np.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_[np.diag_indices(theta.shape[0])] = 0.
    return theta_.flatten() / np.sqrt(theta_**2 + eps)


def lp_prim_only(log_theta: jnp.array, fd_effects, dat: jnp.array) -> jnp.array:
    """Calculates the marginal likelihood of observing unpaired primary tumors at t_1

    Args:
        log_theta (jnp.array):      Theta matrix with logarithmic entries
        fd_effects (jnp.array):     Effects of the PT on first diagnosis
        dat (jnp.array):            Dataset of unpaired primary tumors

    Returns:
        jnp.array: log(L(\theta; Dat_prim))
    """
    score = 0.0
    n_muts = log_theta.shape[0] - 1
    log_theta_prim = log_theta.copy()
    log_theta_prim = log_theta_prim.at[:-1,-1].set(0.)
    log_theta_prim_fd = diagnosis_theta(log_theta_prim, fd_effects)
    for i in range(dat.shape[0]):
        state_prim = dat.at[i, 0:2*n_muts+1:2].get()
        n_prim = int(state_prim.sum())
        score += ssr._lp_prim_obs(log_theta_prim_fd, state_prim, n_prim)
    return score


def lp_met_only(log_theta: jnp.array, fd_effects: jnp.array, sd_effects: jnp.array, dat: jnp.array) -> jnp.array:
    """Calculates the marginal likelihood of observing an unpaired MT at second sampling

    Args:
        log_theta (jnp.array):      Theta matrix with logarithmic entries
        fd_effects (jnp.array):     Effects of PT on first diagnosis
        sd_effects (jnp.array):     Effects of MT on second diagnosis
        dat (jnp.array):            Dataset

    Returns:
        jnp.array: log(P(state))
    """
    log_theta_fd = diagnosis_theta(log_theta, fd_effects)
    log_theta_sd = diagnosis_theta(log_theta, sd_effects)
    n_mut = log_theta.shape[0] - 1
    score = 0.0
    for i in range(dat.shape[0]):
        state_met = jnp.append(dat.at[i, 1:2*n_mut+1:2].get(), 1)
        n_met = int(jnp.sum(state_met))
        score += ssr._lp_met_obs(log_theta_fd, log_theta_sd, state_met, n_met)
    return score


def lp_coupled(log_theta: jnp.array, fd_effects: jnp.array, sd_effects: jnp.array, dat: jnp.array) -> jnp.array:
    """Calculates the log likelihood score of sequential observations of prim-met pairs

    Args:
        log_theta (jnp.array):      theta matrix with logarithmic entries
        fd_effects (jnp.array):     Effects of PT on first diagnosis
        sd_effects (jnp.array):     Effects of MT on second diagnosis
        dat (jnp.array):            Dataset

    Returns:
        jnp.array: log(L(\theta; Dat_coupled))
    """
    score = 0.0
    n_muts = log_theta.shape[0] - 1
    log_theta_fd = diagnosis_theta(log_theta, fd_effects)
    log_theta_sd = diagnosis_theta(log_theta, sd_effects)
    log_theta_prim_fd = log_theta_fd.copy()
    log_theta_prim_fd = log_theta_prim_fd.at[:-1,-1].set(-1.*fd_effects.at[-1].get())
    for i in range(dat.shape[0]):
        state_joint = dat.at[i, 0:2*n_muts+1].get()
        n_prim = int(state_joint.at[::2].get().sum())
        n_met = int(state_joint.at[1::2].get().sum() + 1)
        score += ssr._lp_coupled(log_theta_fd, log_theta_prim_fd, log_theta_sd, state_joint, n_prim, n_met)
    return score



def log_lik(params: np.array, dat_prim_only: jnp.array, dat_prim_met: jnp.array, dat_met: jnp.array, 
            dat_coupled: jnp.array, penal1: float, perc_met: float) -> np.array:
    """Calculates the negative log. likelihood 

    Args:
        params (np.array): n(n+2) dimensional array holding the parameters of the model 
        dat_prim (jnp.array): Dataset containing PT-genotypes only, where no MT was ever diagnosed
        dat_prim_met (jnp.array): Dataset containing MT-genotypes, where an MT was diagnosde but not sequenced
        dat_coupled (jnp.array): Dataset containing PT-MT pairs
        penal (float): Weight of L1-penalization
        perc_met (float): Weights to correct for sampling biases

    Returns:
        np.array: - log. likelihood
    """
    
    n_mut = (dat_prim_only.shape[1]-1)//2
    n_total = n_mut + 1
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    fd_effects = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    sd_effects = jnp.array(params[n_total*(n_total+1):])
    
    l1 = L1(log_theta) + L1(fd_effects) + L1(sd_effects)
    score_prim, score_coupled, score_met, score_prim_met = 0., 0., 0., 0.
    n_prim_only, n_met_only, n_coupled, n_prim_met = 0, 0, 0, 0
    
    if dat_prim_only != None:
        score_prim = lp_prim_only(log_theta, fd_effects, dat_prim_only)
        n_prim_only =  dat_prim_only.shape[0]
    
    if dat_prim_met != None:
        score_prim_met = lp_prim_only(log_theta, fd_effects, dat_prim_met)
        n_prim_met =  dat_prim_met.shape[0]
    
    if dat_met != None:
        score_met = lp_met_only(log_theta, fd_effects, sd_effects, dat_met)
        n_met_only =  dat_met.shape[0]

    if dat_coupled != None:
        score_coupled = lp_coupled(log_theta, fd_effects, sd_effects, dat_coupled)
        n_coupled = dat_coupled.shape[0]

    n_met = n_met_only + n_prim_met + n_coupled
    score = (1 - perc_met) * score_prim/n_prim_only + perc_met/n_met * (score_met + score_prim_met + score_coupled)
    # The optimizer needs np.arrays as input
    return np.array(-score + penal1 * l1)


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
    score = 0.0
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


def value_grad(params: np.array, dat_prim_only: jnp.array, dat_prim_met:jnp.array, dat_met: jnp.array, dat_coupled: jnp.array,
               n_mut: int, penal1: float, penal2: float, perc_met: float) -> tuple[np.array, np.array]:
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
    n_total = n_mut + 1
    log_theta = params[0:-2].reshape((n_total, n_total))

    lam1 = params[-2]
    lam2 = params[-1]

    # Penalties and their derivatives
    l1 = L1(log_theta)
    l1_ = L1_(log_theta)
    lam1 = jnp.exp(params[-2])
    lam2 = jnp.exp(params[-1])
    
    # Transfer theta to the device
    log_theta = jnp.array(log_theta)
    log_theta_prim = log_theta.copy()
    log_theta_prim = log_theta_prim.at[:-1,-1].set(0.)
    
    score_prim, score_coupled, score_met, score_prim_met = 0., 0., 0., 0.
    g_prim, g_coupled, g_met, g_prim_met = jnp.zeros((n_total, n_total)), jnp.zeros((n_total, n_total)), jnp.zeros((n_total, n_total)), jnp.zeros((n_total, n_total))
    n_prim_only, n_coupled, n_prim_met, n_prim_met = 0, 0, 0, 0
    if dat_prim_only != None:
        score_prim, g_prim = grad_prim_only(log_theta_prim, dat_prim_only, lam1, n_mut)
        n_prim_only =  dat_prim_only.shape[0]
    
    if dat_prim_met != None:
        score_prim_met, g_prim_met = grad_prim_only(log_theta_prim, dat_prim_met, lam1, n_mut)
        n_prim_met =  dat_prim_met.shape[0]
    
    if dat_met != None:
        score_met, g_met = grad_met_only(log_theta, dat_met, lam1, lam2, n_mut)
        n_met_only =  dat_met.shape[0]

    if dat_coupled != None:
        score_coupled, g_coupled = grad_coupled(log_theta, log_theta_prim, dat_coupled, lam1, lam2, n_mut)
        n_coupled = dat_coupled.shape[0]
    
    n_met = n_coupled + n_met_only + n_prim_met
    score = (1 - perc_met) * score_prim/n_prim_only + perc_met/n_met * (score_coupled + score_prim_met + score_met)
    g = (1 - perc_met) * g_prim/n_prim_only + perc_met/n_met * (g_coupled +  g_prim_met + g_met)
    return np.array(-score + penal1 * l1 + penal2 * l2), np.array(-g + penal1 * l1_ + penal2 * l2_)