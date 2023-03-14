import numpy as np

from ssr_kronvec_jax import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed, kronvec, kron_diag, obs_inds
import Utilityfunctions as utils
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
import vanilla as mhn


def f1(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = p[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z



def f2(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = m[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z



def f3(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s[:, 3].sum() + p[:, [1, 3]].sum() + m[:, [2, 3]].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z



def t12(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = s[:, 0].sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z



def t3(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s.sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z



def z3(s: jnp.array) -> tuple[jnp.array, float]:
    s = s.reshape((-1, 4), order="C")

    z = s[:, 3].sum()

    s = s.flatten(order="F")

    return s, z


def deriv_no_seed(i, val, x, y, log_theta, state, n):

    z_sync = jnp.multiply(x, kronvec_sync(log_theta=log_theta,
                                p=y, i=i, state=state))
    z_prim = jnp.multiply(x, kronvec_prim(log_theta=log_theta,
                                p=y, i=i, state=state))
    z_met = jnp.multiply(x, kronvec_met(log_theta=log_theta,
                            p=y, i=i, state=state))

    val = val.at[-1].set(z_met.sum())

    def body_fun(j, l_val):

        _z_sync, _z_prim, _z_met, _z = lax.switch(
            state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            [
                lambda s, p, m: (s, p, m, 0.),
                f1,
                f2,
                f3
            ],
            l_val[0],
            l_val[1],
            l_val[2],

        )
        return _z_sync, _z_prim, _z_met, l_val[3].at[j].set(_z)

    z_sync, z_prim, z_met, val = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=body_fun,
        init_val=(z_sync, z_prim, z_met, val)
    )

    z_sync, z_prim, z_met, _z = lax.switch(
        state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        [
            lambda s, p, m: (
                s, p, m, (jnp.sum(s) + jnp.sum(p) + jnp.sum(m))),
            t12,
            t12,
            t3
        ],
        z_sync,
        z_prim,
        z_met,
    )
    val = val.at[i].set(_z)

    z_sync, z_prim, z_met, val = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=body_fun,
        init_val=(z_sync, z_prim, z_met, val)
    )

    return val


def x_partial_Q_y(log_theta: jnp.array, x: jnp.array, y: jnp.array, state: jnp.array) -> jnp.array:
    """This function computes x \partial Q y with \partial Q the Jacobian of Q w.r.t. all thetas
    efficiently using the shuffle trick (sic!).

    Args:
        log_theta (jnp.array): Logarithmic theta values of the MHN 
        x (jnp.array): x vector to multiply with from the left. Length must equal the number of
        nonzero entries in the state vector.
        y (jnp.array): y vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        state (jnp.array): Binary state vector, representing the current sample's events.

    Returns:
        np.array: x \partial_(\Theta_{ij}) Q y for i, j = 1, ..., n+1
    """
    n = log_theta.shape[0] - 1
    z = jnp.zeros(shape=(n + 1, n + 1))

    z = z.at[:-1,:].set(vmap(deriv_no_seed, in_axes=(0,0, None, None, None, None, None), out_axes=0)(jnp.arange(n, dtype=int), z[:-1,:], x, y, log_theta, state, n))
    
    z_seed = jnp.multiply(x, kronvec_seed(log_theta=log_theta, p=y, state=state))

    z = z.at[-1, -1].set(z_seed.sum())

    def body_fun(j, val):

        _z_seed, _z = lax.switch(
            state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda s: (s, 0.),
                lambda s: (
                    s.reshape((-1, 2), order="C").flatten(order="F"), 0.),
                lambda s: (
                    s.reshape((-1, 2), order="C").flatten(order="F"), 0.),
                z3
            ],
            operand=val[0]
        )
        return _z_seed, val[1].at[-1, j].set(_z)

    z_seed, z = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=(z_seed, z)
    )

    return z


@partial(jit, static_argnames=["transpose"])
def R_i_inv_vec(log_theta: jnp.array, x: jnp.array, lam: jnp.array,  state: jnp.array, transpose: bool = False) -> jnp.array:
    """This computes R_i^{-1} x = (\lambda_i I - Q)^{-1} x

    Args:
        log_theta (np.array): Log values of the theta matrix
        x (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        lam (jnp.array): Rate \lambda_i of i-th sampling
        state (jnp.array): Binary state vector, representing the current sample's events.
        transpose (bool): calculate R^(-T)p else calc R^(-1)p
    Returns:
        jnp.array: R_i^{-1} x
    """
    n = log_theta.shape[0] - 1
    state_size = np.log2(x.shape[0]).astype(int)
    lidg = -1 / (kron_diag(log_theta=log_theta, state= state, p_in=x) - lam)
    y = lidg * x

    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=lambda _, val: lidg * (kronvec(log_theta=log_theta, p=val,
                                                state=state, diag=False, transpose=transpose) + x),
        init_val=y
    )

    return y

@jit
def _lp_coupled(log_theta: jnp.array, lam1: jnp.array, lam2: jnp.array, state: jnp.array, p0: jnp.array, 
                    latent_dist: jnp.array, latent_state: jnp.array) -> jnp.array:
    """
    Evaluates the log probability of seeing coupled genotype data in state "state"
    Args:
        log_theta (jnp.array): Logarithmic entries of theta
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): Rate \lambda_2 of second sampling
        state (jnp.array): Bitstring state of prim at t1 and met at t2 if prim_first else vice versa
        p0 (jnp.array): Starting distribution
        latent_dist (jnp.array): distribution of latent states (initially 0)
        latent_state: bitstring, tumor observed at t2
    Returns:
        jnp.array: log(P(state))
    """
    pTh1 = lam1 * R_i_inv_vec(log_theta, p0, lam1,  state)
    inds = obs_inds(pTh1, state, latent_dist, obs_prim = True)
    pTh1_obs = pTh1.at[inds].get()
    obs_sum = pTh1_obs.sum()
    pTh2 = lam2 * mhn.R_inv_vec(log_theta, pTh1_obs/obs_sum, lam2, latent_state)
    return jnp.log(obs_sum) + jnp.log(pTh2.at[-1].get())


@jit
def _lp_prim_obs(log_theta: jnp.array, lam1: jnp.array, state: jnp.array, p0: jnp.array) -> jnp.array:
    """Calculates the likelihood of seeing a primary tumor  at t_1 with genotype state
    
    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        lam1 (jnp.array): rate of first sampling
        state (jnp.array): bitstring of length sum(state), genotype of tumor
        p0 (jnp.array): starting distribution of size 2**sum(state)

    Returns:
        jnp.array: log(P(state))
    """
    log_theta_2 = log_theta.copy()
    log_theta_2 = log_theta.at[0:-1, -1].set(0.0)
    pTh = lam1 * mhn.R_inv_vec(log_theta_2, p0, lam1,  state, False)
    return jnp.log(pTh.at[-1].get())

@jit
def _lp_met_obs(log_theta: jnp.array, lam1: jnp.array, lam2: jnp.array, state_obs: jnp.array, p0: jnp.array):
    """Calculates the likelihood of seeing a met. tumor  at t_2 with genotype state
    
    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        lam1 (jnp.array): Rate \lambda_1 of first sampling
        lam2 (jnp.array): Rate \lambda_2 of second sampling
        state_obs (jnp.array): bitstring of length sum(state), genotype of tumor
        p0 (jnp.array): starting distribution of size 2**sum(state)

    Returns:
        jnp.array: log(P(state))
    """
    pTh1 = mhn.R_inv_vec(log_theta, p0, lam1,  state_obs, False)
    pTh2 = lam1 * lam2 * mhn.R_inv_vec(log_theta, pTh1, lam2, state_obs, False)
    return jnp.log(pTh2.at[-1].get())


@jit
def _grad_met_obs(log_theta: jnp.array, state: jnp.array, p0: jnp.array, lam1: jnp.array, lam2: jnp.array) -> jnp.array:
    """ gradient of lp_met_obs for a single sample

    Args:
        log_theta (jnp.array): Log values of the theta matrix.
        state (jnp.array): Binary state vector, representing the current sample's events.
        p_0 (jnp.array): Initial distribution
        lam1 (jnp.array): Rate \lambda_1 of first sampling.
        lam2 (jnp.array): Rate \lambda_2 of second sampling.

    Returns:
        jnp.array
    """
    R_1_inv_p_0 = mhn.R_inv_vec(
        log_theta=log_theta,
        x=p0,
        lam=lam1,
        state=state)

    R_2_inv_p_1 = mhn.R_inv_vec(
        log_theta=log_theta,
        x=R_1_inv_p_0,
        lam=lam2,
        state=state,
    )

    pTh = lam1 * lam2 * R_2_inv_p_1
    lhs = jnp.zeros_like(p0)
    lhs = lhs.at[-1].set(lam1 * lam2 / pTh.at[-1].get())
    lhs = mhn.R_inv_vec(log_theta, lhs, lam2, state, True)
    dTh = mhn.x_partial_Q_y(log_theta, lhs, R_2_inv_p_1, state)
    lhs = mhn.R_inv_vec(log_theta, lhs, lam1, state, True)
    dlam1 = 1/lam1 - jnp.dot(lhs, R_1_inv_p_0)
    dTh += mhn.x_partial_Q_y(log_theta, lhs, R_1_inv_p_0, state) 
    return dTh, dlam1 * lam1

def _grad_prim_obs(log_theta: jnp.array, state: jnp.array, p0: jnp.array, lam1: jnp.array) -> jnp.array:
    """ gradient of lp_prim_obs for a single sample

    Args:
        log_theta (jnp.array): Log values of the theta matrix.
        state (jnp.array): Binary state vector, representing the current sample's events.
        p_0 (jnp.array): Initial distribution
        lam1 (jnp.array): Rate \lambda_1 of first sampling.

    Returns:
        jnp.array
    """
    log_theta_2 = log_theta.copy()
    log_theta_2 = log_theta.at[0:-1, -1].set(0.0)
    R_1_inv_p_0 = mhn.R_inv_vec(
        log_theta=log_theta_2,
        x=p0,
        lam=lam1,
        state=state)
    
    p_theta = lam1 * R_1_inv_p_0
    lhs = jnp.zeros_like(p0)
    lhs = lhs.at[-1].set(1/p_theta.at[-1].get())
    lhs = mhn.R_inv_vec(log_theta_2, lhs, lam1, state, transpose = True)
    dlam1 = 1/lam1 - lam1*jnp.dot(lhs, R_1_inv_p_0)
    dth = lam1 * mhn.x_partial_Q_y(log_theta=log_theta_2, x=lhs, y=R_1_inv_p_0, state=state)
    dth = dth.at[:-1, -1].set(0.0)  # Derivative of constant is 0.
    return dth, dlam1 * lam1

@jit
def _g_coupled(log_theta: jnp.array, state: jnp.array, g3_lhs: jnp.array, obs_inds: jnp.array, pTh1: jnp.array, 
                nk: jnp.array, lam1: jnp.array) -> tuple:
    """Calculates the first, third and fourth summand of the gradient of pTh for a single prim-met pair

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        state (jnp.array): bitstring, genotypes of coupled sequencing data
        g3_lhs (jnp.array): lhs vector needed for calculating the third summand
        obs_inds (jnp.array): vector of observed states
        pTh1 (jnp.array): time marginal probility distribution at the point of first sampling
        nk (jnp.array): total probability mass assigned to observed states
        lam1 (jnp.array): Rate \lambda_1 of first sampling

    Returns:
        tuple 
    """
    #obs_inds_t = obs_inds.at[0:obs_inds.shape[0]//2].set(0)
    q = jnp.where(obs_inds == 1, 1/nk, 0.0)
    
    # gradient of first summand of the likelihood score
    q = R_i_inv_vec(log_theta, q, lam1, state, transpose = True)
    g_1 = x_partial_Q_y(log_theta, q, pTh1, state)
    
    # Partial derivative wrt. lambda1
    R1pth1 = R_i_inv_vec(log_theta, pTh1, lam1, state, transpose = False)/nk
    R1pth1 = jnp.where(obs_inds, R1pth1, 0.0)
    dlam1 = 1.0/lam1 - jnp.dot(q, pTh1) - jnp.dot(g3_lhs, R1pth1) + jnp.sum(R1pth1)    
    
    # 3rd summand of gradient
    q = R_i_inv_vec(log_theta, g3_lhs, lam1, state, transpose = True)
    g_3 = x_partial_Q_y(log_theta, q, pTh1, state)/nk
    
    # 4th summand of gradient
    q = obs_inds
    q = R_i_inv_vec(log_theta, q, lam1, state, transpose = True)
    g_4 = -1.0/nk * x_partial_Q_y(log_theta, q, pTh1, state)
    
    return g_1 + g_3 + g_4, dlam1 * lam1

@jit
def _g_3_lhs(log_theta: jnp.array, pTh1: jnp.array, pTh2: jnp.array, latent_inds: jnp.array, 
            second_obs: jnp.array, lam2:jnp.array) -> jnp.array:
    """Calculate the lhs needed for the third summand of the gradient

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        pTh1 (jnp.array): Joint probability distribution at first sampling
        pTh2 (jnp.array): Marginal probability distribution at second sampling
        latent_inds (jnp.array): Vector holding indices of observed states
        second_obs (jnp.array): Genotype of datapoint observed at second sampling
        lam2 (jnp.array): Rate \lambda_2 of second sampling

    Returns:
        jnp.array
    """
    q = jnp.zeros_like(pTh2)
    q = q.at[-1].set(1/pTh2.at[-1].get())
    q = mhn.R_inv_vec(log_theta, q, lam2, second_obs, transpose = True)
    q_big = jnp.zeros_like(pTh1)
    q_big =  q_big.at[latent_inds].set(q)
    return q_big