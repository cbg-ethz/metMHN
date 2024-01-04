from metmhn.jx import likelihood as ssr
from metmhn.jx.kronvec import diagnosis_theta
from metmhn.jx.vanilla import R_inv_vec
import logging 
import jax.numpy as jnp
import numpy as np
import scipy.optimize as opt


def L1(theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Computes the L1 penalty
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return jnp.sum(jnp.sqrt(theta_**2 + eps))


def L1_(theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    """
    Derivative of the L1 penalty
    """
    theta_ = theta.copy()
    if theta.ndim == 2:
        theta_ = theta_.at[jnp.diag_indices(theta.shape[0])].set(0.)
    return theta_.flatten() / jnp.sqrt(theta_.flatten()**2 + eps)


def sym_penal(log_theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    n = log_theta.shape[0]
    theta_ = log_theta.at[jnp.diag_indices(n)].set(0.)
    penal = jnp.sum(jnp.sqrt(theta_**2 + theta_.T**2 - theta_ * theta_.T + eps))
    return 0.5*(penal - n*jnp.sqrt(eps))


def sym_penal_(log_theta: jnp.ndarray, eps: float = 1e-05) -> jnp.ndarray:
    n = log_theta.shape[0]
    theta_ = log_theta.at[jnp.diag_indices(n)].set(0.)
    penal_denom = 2*jnp.sqrt(theta_**2 + theta_.T**2 - theta_ * theta_.T + eps)
    penal_num = 2*theta_ - theta_.T
    return (penal_num/penal_denom).flatten()


def lp_prim_only(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, 
                 dat: jnp.ndarray) -> jnp.ndarray:
    """Calculates the marginal likelihood of observing unpaired primary tumors

    Args:
        log_theta (jnp.ndarray):   Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray):     Log. effects of muts in the PT on PT-diagnosis
        dat (jnp.ndarray):         Dataset of unpaired primary tumors

    Returns:
        jnp.ndarray: log(L(\theta; Dat_prim))
    """
    score = 0.0
    n_muts = log_theta.shape[0] - 1
    for i in range(dat.shape[0]):
        state_pt = dat[i, 0:2*n_muts+1:2]
        n_prim = int(state_pt.sum())
        score += ssr._lp_prim_obs(log_theta, log_d_p, state_pt, n_prim)
    return score


def lp_met_only(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, 
                log_d_m: jnp.ndarray, dat: jnp.ndarray) -> jnp.ndarray:
    """Calculates the marginal likelihood of observing an unpaired MT

    Args:
        log_theta (jnp.ndarray):   Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray):     Log. effects of muts prior to the seeding on diagnosis
        log_d_m (jnp.ndarray):     Effects of muts after seeding on diagnosis
        dat (jnp.ndarray):         Dataset

    Returns:
        jnp.ndarray: log(P(dat, theta))
    """
    n_mut = log_theta.shape[0] - 1
    score = 0.0
    for i in range(dat.shape[0]):
        state_mt = jnp.append(dat[i, 1:2*n_mut+1:2], 1)
        n_met = int(jnp.sum(state_mt))
        score += ssr._lp_met_obs(log_theta, log_d_p, log_d_m, state_mt, n_met)
    return score


def lp_coupled(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, 
               log_d_m: jnp.ndarray, dat: jnp.ndarray) -> jnp.ndarray:
    """Calculates the log likelihood score of coupled PT-MT pairs

    Args:
        log_theta (jnp.ndarray):   Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray):     Effects of muts in PT on its diagnosis
        log_d_m (jnp.ndarray):     Effects of muts in MT on its diagnosis
        dat (jnp.ndarray):         Dataset

    Returns:
        jnp.array: log(L(\theta; Dat_coupled))
    """
    score = 0.0
    n_muts = log_theta.shape[0] - 1
    for i in range(dat.shape[0]):
        state_joint = dat[i, 0:2*n_muts+1]
        n_prim = int(state_joint[::2].sum())
        n_met = int(state_joint[1::2].sum() + 1)
        order = dat[i,2*n_muts+1]
        if order == 0:
            score += ssr._lp_coupled_0(log_theta, log_d_p, log_d_m, state_joint,
                                        n_prim, n_met)
        elif order == 1:
            score += ssr._lp_coupled_1(log_theta, log_d_p, log_d_m, state_joint,
                                        n_prim, n_met)
        else:
            score += ssr._lp_coupled_2(log_theta, log_d_p, log_d_m, state_joint,
                                        n_prim, n_met)

    return score


def log_lik(params: np.ndarray, dat_prim_only: jnp.ndarray, dat_prim_met: jnp.ndarray,  
            dat_met:jnp.ndarray, dat_coupled: jnp.ndarray, penal1: float, 
            perc_met: float) -> np.ndarray:
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
    n_m_o, n_p_m, n_c = dat_met.shape[0], dat_prim_met.shape[0], dat_coupled.shape[0]
    n_p_o = dat_prim_only.shape[0]

    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    
    l1 = sym_penal(log_theta) + L1(log_d_p) + L1(log_d_m)

    score_prim = lp_prim_only(log_theta, log_d_p, dat_prim_only)
    score_prim_met = lp_prim_only(log_theta, log_d_p, dat_prim_met)
    score_met_only = lp_met_only(log_theta, log_d_p, log_d_m, dat_met)
    score_coupled = lp_coupled(log_theta, log_d_p, log_d_m, dat_coupled)
    
    #w_m_o = n_c/n_m_o
    #w_p_m = n_c/n_p_m
    #score_mets = score_coupled + w_p_m * score_prim_met + w_m_o*score_met_only
    score_mets = score_coupled + score_prim_met + score_met_only
    #n_mets = n_c + w_m_o * n_m_o + w_p_m * n_p_m
    n_mets = n_c + n_m_o + n_p_m
    
    w = (1-perc_met)*n_mets/(perc_met * n_p_o)
    n_dat = w * n_p_o + n_mets 
    score = np.array(-(w * score_prim + score_mets) + penal1 * n_dat * l1)

    logging.info(f"score {score}, score_prim {score_prim}, score_c {score_coupled}, score_pm {score_prim_met}, score_mo {score_met_only}")
    # The SciPy-Optimizer only takes np.arrays as input
    return score


def grad_prim_only(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, 
                   dat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Gradient of lp_prim
    
    Args:
        log_theta (jnp.ndarray):      Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray):     Effects of mutations on first diagnosis
        dat (jnp.ndarray):            Dataset with PTs only

    Returns:
        tuple:                      Likelihood, gradient wrt. model parameters
    """
    n_total = log_theta.shape[0]
    g = jnp.zeros((n_total, n_total))
    d_p = jnp.zeros(n_total)
    score = 0.0
    for i in range(dat.shape[0]):
        state_obs = dat[i, 0:2*n_total-1:2]
        n_prim = int(state_obs.sum())      
        s, g_, fd_ = ssr._grad_prim_obs(log_theta, log_d_p, state_obs, n_prim)
        g += g_
        d_p += fd_
        score += s
    return score, jnp.concatenate((g.flatten(), d_p, jnp.zeros(n_total)))


def grad_met_only(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  dat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """ Gradient of lp_met

      Args:
        log_theta (jnp.array):      Theta matrix with logarithmic entries
        fd_effects (jnp.array):     Effects of mutations in the PT on the rate of first diagnosis
        sd_effects (jnp.array):     Effects of mutation in the MT on the rate of second diagnosis
        dat (jnp.array):            Dataset of MT-genotypes without PT-sequences

    Returns:
        tuple:                      Marginal Likelihood, grad wrt. model parameters
    """
    # Unpack parameters
    n_mut = log_theta.shape[0] - 1
    
    g = jnp.zeros((n_mut+1,n_mut+1), dtype=float)
    d_p = jnp.zeros(n_mut+1)
    d_m = jnp.zeros(n_mut+1)
    score = 0.0
    
    for i in range(dat.shape[0]):
        state = dat.at[i, 0:2*n_mut+1].get()
        state_met = jnp.append(state.at[1:2*n_mut+1:2].get(), 1)
        n_met = int(state_met.sum())
        s, g_, fd_, sd_ = ssr._grad_met_obs(log_theta, log_d_p, log_d_m, state_met, n_met)
        score += s
        g += g_
        d_p += fd_
        d_m += sd_
    return score, jnp.concatenate((g.flatten(), d_p, d_m))


def grad_coupled(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                 dat: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns the likelihood and gradients for a dataset containing coupled genotypes of PTs and MTs

    Args:
        log_theta (jnp.array):      Theta matrix with logarithmic entries
        fd_effects (jnp.array):     Effects of mutations in the PT on the rate of first diagnosis
        sd_effects (jnp.array):     Effects of mutations in the MT on the rate of second diagnosis
        dat (jnp.array):            Dataset containing coupled genotypes of PTs and MTs

    Returns:
        tuple[jnp.array, jnp.array]: likelihood, gradient wrt. model parameters
    """
    # Unpack parameters
    n_mut = log_theta.shape[0] - 1
    g = jnp.zeros((n_mut+1, n_mut+1), dtype=float)
    d_p = jnp.zeros(n_mut+1)
    d_m = jnp.zeros(n_mut+1)
    score = 0.0

    for i in range(dat.shape[0]):
        state = dat[i, 0:2*n_mut+1]
        n_prim = int(state[::2].sum())
        n_met = int(state[1::2].sum() + 1)
        order = dat[i,2*n_mut+1]
        if order == 0:
            lik, d_th, d_d_p, d_d_m = ssr._g_coupled_0(log_theta, log_d_p, log_d_m, state,
                                                             n_prim, n_met)
        elif order == 1:
            lik, d_th, d_d_p, d_d_m = ssr._g_coupled_1(log_theta, log_d_p, log_d_m, state,
                                                            n_prim, n_met)
        else:
            lik, d_th, d_d_p, d_d_m = ssr._g_coupled_2(log_theta, log_d_p, log_d_m, state,
                                                            n_prim, n_met)
        
        score += lik
        g += d_th
        d_p += d_d_p
        d_m += d_d_m

    return score, jnp.concatenate((g.flatten(), d_p, d_m)) 


def grad(params: np.ndarray, dat_prim_only: jnp.ndarray, dat_prim_met:jnp.ndarray, 
         dat_met: jnp.ndarray, dat_coupled: jnp.ndarray, penal: float, 
         perc_met: float) -> tuple[np.array, np.array]:
    """Calculates the gradient and log_lik wrt. to all log(\theta_ij)

    Args:
        params (np.array):          Array of size n*(n+2), holding all parameters of the model 
        dat_prim_only (jnp.array):  Dataset containing only PT genotypes, that never spawned an MT
        dat_prim_met (jnp.array):   Dataset containing only PT genotypes, that spawned an MT
        dat_met (jnp.array):        Dataset containing only MT genotypes
        dat_coupled (jnp.array):    Dataset containing coupled PT-MT pairs
        penal (float):              Weight of L1-penalization
        perc_met (float):           Weights to correct for sampling biases

    Returns:
        np.array:                   Gradient wrt. params
    """
    # Unpack parameters
    n_dat = dat_coupled.shape[0] + dat_met.shape[0] + dat_prim_met.shape[0] + dat_prim_only.shape[0]
    n_mut = (dat_prim_only.shape[1]-1)//2
    n_total = n_mut + 1
    n_m_o, n_p_m, n_c = dat_met.shape[0], dat_prim_met.shape[0], dat_coupled.shape[0]
    n_p_o = dat_prim_only.shape[0]
    
    # Transfer parameters to the device
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])

    # Penalties and their derivatives
    l1_ = np.concatenate((sym_penal_(log_theta), L1_(log_d_p), L1_(log_d_m)))
    l1 = sym_penal(log_theta) + L1(log_d_p) + L1(log_d_m)
    
    # Scores and gradients of all datasets
    score_prim, g_prim = grad_prim_only(log_theta, log_d_p, dat_prim_only)
    score_prim_met, g_prim_met = grad_prim_only(log_theta, log_d_p, dat_prim_met)
    score_met_only, g_met_only = grad_met_only(log_theta, log_d_p, log_d_m, dat_met)
    score_coupled, g_coupled = grad_coupled(log_theta, log_d_p, log_d_m, dat_coupled)
    
    #w_m_o = n_c/n_m_o
    #w_p_m = n_c/n_p_m
    #g_mets = g_coupled + w_p_m * g_prim_met + w_m_o * g_met_only
    score_mets = score_prim_met + score_met_only + score_coupled
    g_mets = g_coupled + g_prim_met + g_met_only
    #n_mets = n_c + w_m_o * n_m_o + w_p_m * n_p_m
    n_mets = n_c + n_m_o + n_p_m
    w = (1-perc_met)*n_mets/(perc_met * n_p_o)
    n_dat = w*n_p_o + n_mets
    g = w*g_prim + g_mets
    score = w * score_prim + score_mets
    # The SciPy-Optimizer only takes np.arrays as input
    logging.info(f"score {score}, score_prim {score_prim}, score_c {score_coupled}, score_pm {score_prim_met}, score_mo {score_met_only}")
    return np.array(-score + penal * n_dat * l1), np.array(-g + penal * n_dat * l1_)


def learn_mhn(th_init: jnp.ndarray, dp_init: jnp.ndarray, dm_init: jnp.ndarray, 
              dat_prim_only: jnp.ndarray, dat_prim_met: jnp.ndarray, dat_met: jnp.ndarray, 
              dat_coupled: jnp.ndarray, perc_met: float, penal: float, opt_iter: int=1e05, opt_ftol: float=1e-04, 
              opt_v: bool=True) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Infer a metMHN from data

    Args:
        th_init (jnp.ndarray):        Initial estimate for the log-theta matrix
                                    Matrix of dimension (n_muts+1) x (n_muts+1)
        dp_init (jnp.ndarray):        Initial estimate for the effects of muts on first diagnosis.
                                    Vector of size n_muts+1
        dm_init (jnp.ndarray):        Inital estimate for the effects of muts on second diagnosis
                                    Vector of size n_muts+1 
        dat_prim_only (jnp.ndarray):  Dataset containing PTs without observed MTs. 
                                    Binary matrix of size n_prim x 2*n_muts+1
        dat_prim_met (jnp.ndarray):   Dataset containing PTs with observed but not sequenced MTs
                                    Binary matrix of size n_prim_met x 2*n_muts+1 
        dat_met (jnp.ndarray):   Dataset containing MTs without matching PTs
                                    Binary matrix of size n_met x 2*n_muts+1
        dat_coupled (jnp.ndarray):    Dataset containing PT and MT genotyps in the same patient
                                    Binary matrix of size n_coupled x 2*n_muts+1
        perc_met (float):           Correction factor for the sampling bias. Should be a float in the range[0, 1] 
                                    and corresponds to the epidemiological prevalence of MTs in a population of patients
        penal (float):              Weight of the L1-penalization
        opt_iter (int):             Maximal number of iterations for optimizer. Defaults to 1e05
        opt_ftol (float):           Tolerance for optimizer. Defaults to 1e-05
        opt_v (bool):               Print out optimizer progress. Defaults to TRUE
        

    Returns:s
        tuple[np.ndarray, np.ndarray, np.ndarray]: Estimated log. theta, log. d_p, log. d_m
    """
    n_total = th_init.shape[0]
    start_params = np.concatenate((th_init.flatten(), dp_init, dm_init))
    x = opt.minimize(fun=grad, jac=True, x0=start_params, method="L-BFGS-B",  
                     args=(dat_prim_only, dat_prim_met, dat_met, dat_coupled, penal, perc_met), 
                     options={"maxiter":opt_iter, "disp": opt_v, "ftol": opt_ftol})
    theta = jnp.array(x.x[:n_total**2]).reshape((n_total, n_total))
    d_p = jnp.array(x.x[n_total**2:n_total*(n_total+1)])
    d_m = jnp.array(x.x[n_total*(n_total+1):])
    return theta, d_p, d_m


def p_unobs_seeding(log_theta: jnp.ndarray, log_d_p: jnp.ndarray,  dat: jnp.ndarray) -> jnp.ndarray:
    """This calculates the probability, that a MT is present given the genotype of the PT at the time of PT-diagnosis

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log. effects of the muts in the PT on its diagnosis
        dat (jnp.ndarray): Dataset of PT-observations

    Returns:
        jnp.ndarray: P(Seeding = 1|PT)
    """
    dat_mod = dat[:,::2]
    dat_mod = dat_mod.at[:,-1].set(1)
    log_theta_pt = log_theta.at[:-1, -1].set(0.)
    log_theta_pt = diagnosis_theta(log_theta_pt, log_d_p)
    p_unobs = jnp.zeros(dat_mod.shape[0])
    for i in range(dat_mod.shape[0]):
        obs = dat_mod.at[i,:].get()
        m =  obs.sum()
        p0 = jnp.zeros(2**m)
        p0 = p0.at[0].set(1.)
        pth = R_inv_vec(log_theta_pt, p0, 1.,  obs, False)
        p_state = pth.reshape((-1, 2), order="F")[-1,:]
        p_unobs = p_unobs.at[i].set(p_state[1]/p_state.sum())
    return p_unobs