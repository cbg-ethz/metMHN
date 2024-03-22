from metmhn.jx import likelihood as ssr
import metmhn.jx.one_event as one
import logging 
import jax.numpy as jnp
from jax import vmap
import numpy as np
import scipy.optimize as opt
from typing import Callable


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


def symmetric_penal(params: np.array, n_total: int, eps=1e-05) -> tuple[jnp.ndarray, jnp.ndarray]:
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    penal = np.array(sym_penal(log_theta) + L1(log_d_p) + L1(log_d_m))
    penal_ = np.concatenate((sym_penal_(log_theta), L1_(log_d_p), L1_(log_d_m)))
    return penal, penal_


def score(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, dat: jnp.ndarray, 
          perc_met: float)-> jnp.ndarray:
    """Calculates the log. likelihood of the dataset dat

    Args:
        log_theta (jnp.ndarray): (n+1)x(n+1)-dimensional Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the PT on the rate of its observation event
        log_d_m (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the MT on the rate of its observation event
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset.
    
    Returns:
        jnp.ndarray: Log. likelihood
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    score, score_pt = 0., 0.
    for i in range(dat.shape[0]):
        if dat[i,-1] == 0:
            # Never metastasizing primary tumors
            state_obs = dat[i, 0:2*n_total-1:2]
            n_prim = int(state_obs.sum())
            score_pt += ssr._lp_prim_obs(log_theta, log_d_p, state_obs, n_prim)
        else:
            if dat[i,-1] == 1:
            # Metastasized primary tumors without sequenced metastasis
                state_obs = dat[i, 0:2*n_total-1:2]
                n_prim = int(state_obs.sum())      
                score += ssr._lp_prim_obs(log_theta, log_d_p, state_obs, n_prim)
            elif dat[i, -1] == 2:
                # Metastates without sequenced primary tumor
                state_obs = dat[i, 0:2*n_total-1]
                state_met = jnp.append(state_obs[1:2*n_total-1:2], 1)
                n_met = int(state_met.sum())
                score += ssr._lp_met_obs(log_theta, log_d_p, log_d_m, state_met, n_met)
            elif dat[i, -1] == 3:
                # Paired primary tumor and metastasis observation
                state_obs = dat[i, 0:2*n_mut+1]
                n_prim = int(state_obs[::2].sum())
                n_met = int(state_obs[1::2].sum() + 1)
                order = dat[i,-2]
                if order == 0:
                    if (n_prim + n_met-1) == 1:
                        score += one._lp_coupled_0(log_theta, log_d_p, log_d_m, state_obs)
                    else:
                        score += ssr._lp_coupled_0(log_theta, log_d_p, log_d_m, state_obs,
                                                           n_prim, n_met)
                elif order == 1:
                    if (n_prim + n_met-1) == 1:
                        score += one._lp_coupled_1(log_theta, log_d_p, log_d_m, state_obs)
                    else:
                        score += ssr._lp_coupled_1(log_theta, log_d_p, log_d_m, state_obs,
                                                           n_prim, n_met)
                else:
                    if (n_prim + n_met-1) == 1:
                        score += one._lp_coupled_2(log_theta, log_d_p, log_d_m, state_obs)
                    else:
                        score += ssr._lp_coupled_2(log_theta, log_d_p, log_d_m, state_obs,
                                                           n_prim, n_met)
    logging.info(f"NM-score: {score_pt}, EM-score: {score}")

    n_em = jnp.sum(dat[:,-3])
    n_nm = dat.shape[0] - n_em
    # Weight MTs relative to PTs to achieve the prespecified ratio perc_met 
    if n_em*n_nm != 0:
        w = perc_met * n_nm/((1-perc_met)*n_em)
    else:
        w = 1
    n_full = w*n_em + n_nm
    score = (w*score + score_pt)/n_full
    return score


def score_reg(params: np.ndarray, dat: jnp.ndarray, perc_met: float, penal: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], 
              w_penal: float) -> np.ndarray:
    """Calculates the negative log. likelihood and its gradient of the dataset dat with regularization penal

    Args:
        params (np.ndarray): (n+1)*(n+2)-dimensional vecor of parameters, the first (n+1)**2 entries correspond to log. Theta, 
            the next (n+1) to log_d_p and the last (n+1)-entries to log_d_m
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset
        penal (Callable[[np.ndarray, int], tuple[np.ndarray, np.ndaray]]): Penalization function, should take a parametervector params and total number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        w_penal (float): weight of the penalization

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative penalized log. likelihood, grad wrt. to all model parameters
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    # Transfer parameters to the device
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    sc = score(log_theta, log_d_p, log_d_m, dat, perc_met)
    pen, _ = penal(params, n_total)
    return np.array(-sc + w_penal*pen)


def score_and_grad(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, dat: jnp.ndarray, 
                   perc_met: float)->tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculates the log. likelihood and its gradient of the dataset dat

    Args:
        log_theta (jnp.ndarray): (n+1)x(n+1)-dimensional Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the PT on the rate of its observation event
        log_d_m (jnp.ndarray): (n+1)-dimensional vector with logarithmic effects of events in the MT on the rate of its observation event
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset.
    
    Returns:
        tuple[np.array, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Log. likelihood, grad wrt. theta, grad wrt. log_d_p, grad wrt. log_d_m
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    score, score_pt = 0., 0.
    d_th, d_th_pt = jnp.zeros((n_total, n_total)), jnp.zeros((n_total, n_total))
    d_d_p, d_d_p_pt = jnp.zeros(n_total), jnp.zeros(n_total) 
    d_d_m = jnp.zeros(n_total)

    # Never metastasizing primary tumors
    dat_po = dat[dat[:,-1]==0,:]
    n_active = jnp.unique(dat_po[:,:-2:2].sum(axis=1))
    for i in n_active:
        tmp = dat_po[dat_po[:,:-2:2].sum(axis=1)==i, :-2:2]
        lik, th_, dp_ = vmap(ssr._grad_prim_obs, (None, None, 0, None), out_axes=(0))(log_theta, log_d_p, tmp, int(i))
        score_pt += lik.sum()
        d_th_pt += th_.sum(axis=0)
        d_d_p_pt += dp_.sum(axis=0)

    # Metastasized primary tumors
    dat_pm = dat[dat[:,-1]==1,:]
    n_active = jnp.unique(dat_pm[:,:-2:2].sum(axis=1))
    for i in n_active:
        tmp = dat_pm[dat_pm[:,:-2:2].sum(axis=1)==i, :-2:2]
        lik, th_, dp_ = vmap(ssr._grad_prim_obs, (None, None, 0, None), out_axes=(0))(log_theta, log_d_p, tmp, int(i))
        score += lik.sum()
        d_th += th_.sum(axis=0)
        d_d_p += dp_.sum(axis=0)
    
    # Metastases
    dat_m = dat[dat[:,-1]==2,:]
    n_active = jnp.unique(dat_m[:,1:-2:2].sum(axis=1)) + 1
    for i in n_active:
        tmp = dat_m[dat_m[:,1:-2:2].sum(axis=1)+1==i, 1:-2:2]
        tmp = jnp.hstack((tmp, jnp.ones(tmp.shape[0], dtype=jnp.int8).reshape(-1,1)))
        lik, th_, dp_, dm_ = vmap(ssr._grad_met_obs, (None, None, None, 0, None), out_axes=(0))(log_theta, log_d_p, log_d_m, tmp, int(i))
        score += lik.sum()
        d_th += th_.sum(axis=0)
        d_d_p += dp_.sum(axis=0)
        d_d_m += dm_.sum(axis=0)
    
    # Paired primary tumors and metastases
    dat_c = dat[dat[:,-1]==3,:]
    for i in range(dat_c.shape[0]):
        state_obs = dat_c[i, 0:2*n_mut+1]
        n_prim = int(state_obs[::2].sum())
        n_met = int(state_obs[1::2].sum() + 1)
        order = dat_c[i,-2]
        if order == 0:
            if (n_prim + n_met-1) == 1:
                s, th_, d_p_, d_m_ = one._g_coupled_0(log_theta, log_d_p, log_d_m, state_obs)
            else:
                s, th_, d_p_, d_m_ = ssr._g_coupled_0(log_theta, log_d_p, log_d_m, state_obs,
                                                        n_prim, n_met)
        elif order == 1:
            if (n_prim + n_met-1) == 1:
                s, th_, d_p_, d_m_ = one._g_coupled_1(log_theta, log_d_p, log_d_m, state_obs)
            else:
                s, th_, d_p_, d_m_ = ssr._g_coupled_1(log_theta, log_d_p, log_d_m, state_obs,
                                                        n_prim, n_met)
        else:
            if (n_prim + n_met-1) == 1:
                s, th_, d_p_, d_m_ = one._g_coupled_2(log_theta, log_d_p, log_d_m, state_obs)
            else:
                s, th_, d_p_, d_m_ = ssr._g_coupled_2(log_theta, log_d_p, log_d_m, state_obs,
                                                      n_prim, n_met)
        score += s
        d_th += th_
        d_d_p += d_p_
        d_d_m += d_m_
    logging.info(f"NM-score: {score_pt}, EM-score: {score}")

    n_em = jnp.sum(dat[:,-3])
    n_nm = dat.shape[0] - n_em
    if n_em*n_nm != 0:
        w = perc_met * n_nm/((1-perc_met)*n_em)
    else:
        w = 1
    n_full = w*n_em + n_nm
    score = (w*score + score_pt)/n_full
    d_th = (w*d_th + d_th_pt)/n_full
    d_d_p = (w*d_d_p + d_d_p_pt)/n_full
    d_d_m = w*d_d_m/n_full
    return score, d_th, d_d_p, d_d_m


def score_and_grad_reg(params: np.ndarray, dat: jnp.ndarray, perc_met: float, penal: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], 
                       w_penal: float) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the negative log. likelihood and its gradient of the dataset dat with regularization penal

    Args:
        params (np.ndarray): (n+1)*(n+2)-dimensional vecor of parameters, the first (n+1)**2 entries correspond to log. Theta, 
            the next (n+1) to log_d_p and the last (n+1)-entries to log_d_m
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float): Expected percentage of metastasizing tumor in the Dataset
        penal (Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]): Penalty function, should take parametervector params and totoal number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        w_penal (float): weight of the penalization

    Returns:
        tuple[np.ndarray, np.ndarray]: Negative penalized log. likelihood, grad wrt. to all model parameters
    """
    n_mut = (dat.shape[1]-3)//2
    n_total = n_mut + 1
    # Transfer parameters to the device
    log_theta = jnp.array(params[0:n_total**2]).reshape((n_total, n_total))
    log_d_p = jnp.array(params[n_total**2:n_total*(n_total + 1)])
    log_d_m = jnp.array(params[n_total*(n_total+1):])
    score, d_th, d_d_p, d_d_m = score_and_grad(log_theta, log_d_p, log_d_m, dat, perc_met)
    grad_vec = np.concatenate((d_th.flatten(), d_d_p, d_d_m))
    pen, pen_ = penal(params, n_total)
    return np.array(-score + w_penal*pen), -grad_vec + w_penal*pen_ 


def learn_mhn(th_init: jnp.ndarray, dp_init: jnp.ndarray, dm_init: jnp.ndarray, dat: jnp.ndarray, perc_met: float, 
              penal: Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]], w_penal: float, opt_iter: int=1e05, opt_ftol: float=1e-04, 
              opt_v: bool=True) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """ Infer a metMHN from data

    Args:
        th_init (jnp.ndarray): Initial estimate for the log-theta matrix. Matrix of dimension (n_muts+1) x (n_muts+1)
        dp_init (jnp.ndarray): Initial estimate for the effects of muts on PT-observation. Vector of size n_muts+1
        dm_init (jnp.ndarray): Inital estimate for the effects of muts on MT-observation. Vector of size n_muts+1 
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        perc_met (float):  Expected percentage of metastasizing tumor in the Dataset
        penal (Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]): Penalty function, should take parametervector params and totoal number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        penal (float): Weight of the penalty
        opt_iter (int): Maximal number of iterations for optimizer. Defaults to 1e05
        opt_ftol (float): Tolerance for optimizer. Defaults to 1e-04
        opt_v (bool):  Print out optimizer progress. Defaults to TRUE
        

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Estimated log. theta, log. d_p, log. d_m
    """
    n_total = th_init.shape[0]
    start_params = np.concatenate((th_init.flatten(), dp_init, dm_init))
    x = opt.minimize(fun=score_and_grad_reg, jac=True, x0=start_params, method="L-BFGS-B",  
                     args=(dat, perc_met, penal, w_penal), 
                     options={"maxiter":opt_iter, "disp": opt_v, "ftol": opt_ftol})
    theta = jnp.array(x.x[:n_total**2]).reshape((n_total, n_total))
    d_p = jnp.array(x.x[n_total**2:n_total*(n_total+1)])
    d_m = jnp.array(x.x[n_total*(n_total+1):])
    return theta, d_p, d_m