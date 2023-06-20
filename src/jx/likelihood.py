import numpy as np

from jx.kronvec import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed, kronvec, kron_diag, obs_states
import Utilityfunctions as utils
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from jx import vanilla as mhn


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
    # Here be OOM-errors
    #z = z.at[:-1,:].set(vmap(deriv_no_seed, in_axes=(0,0, None, None, None, None, None), out_axes=0)(jnp.arange(n, dtype=jnp.int64), z[:-1,:], x, y, log_theta, state, n))
    def init_z(j, val):
        val = val.at[j, :].set(deriv_no_seed(j, val[:, j], x, y, log_theta, state, n))
        return val
    z = lax.fori_loop(lower=0, upper=n, body_fun=init_z, init_val=z) 
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
        x (np.array): Vector to multiply with from the right. Length must equal the 2 to the number of
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
                poss_inds: jnp.array, latent_state: jnp.array) -> jnp.array:
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
    poss_states = obs_states(pTh1, state)
    poss_inds = jnp.where(poss_states == 1, size = poss_inds.shape[0])[0]
    pTh1_obs = pTh1.at[poss_inds].get()
    obs_sum = pTh1_obs.sum()
    pTh1_obs = jnp.append(jnp.zeros_like(pTh1_obs), pTh1_obs)
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
    log_theta_pt = log_theta.copy()
    log_theta_pt = log_theta.at[0:-1, -1].set(0.0)   # set all influences from seeding to 0 as it does not affect the pt
    pTh = lam1 * mhn.R_inv_vec(log_theta_pt, p0, lam1,  state, False)
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
    score = jnp.log(pTh.at[-1].get())
    
    lhs = jnp.zeros_like(p0)
    lhs = lhs.at[-1].set(lam1 * lam2 / pTh.at[-1].get())
    lhs = mhn.R_inv_vec(log_theta, lhs, lam2, state, True)
    dTh = mhn.x_partial_Q_y(log_theta, lhs, R_2_inv_p_1, state)
    lhs = mhn.R_inv_vec(log_theta, lhs, lam1, state, True)
    dlam1 = 1/lam1 - jnp.dot(lhs, R_1_inv_p_0)
    dTh += mhn.x_partial_Q_y(log_theta, lhs, R_1_inv_p_0, state) 
    return score, dTh, dlam1 * lam1

@jit
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
    score = jnp.log(p_theta.at[-1].get())
    
    lhs = jnp.zeros_like(p0)
    lhs = lhs.at[-1].set(1/p_theta.at[-1].get())
    lhs = mhn.R_inv_vec(log_theta_2, lhs, lam1, state, transpose = True)
    dlam1 = 1/lam1 - lam1*jnp.dot(lhs, R_1_inv_p_0)
    dth = lam1 * mhn.x_partial_Q_y(log_theta=log_theta_2, x=lhs, y=R_1_inv_p_0, state=state)
    dth = dth.at[:-1, -1].set(0.0)  # Derivative of constant is 0.
    return score, dth, dlam1 * lam1

@partial(jit, static_argnames=["n_prim", "n_met"])
def _g_coupled(log_theta: jnp.array, theta_prim: jnp.array, state_joint: jnp.array, n_prim: int, n_met: int, lam1: jnp.array, lam2: jnp.array) -> tuple:
    """Calculates the likelihood and gradient for a datapoint state_joint

    Args:
        log_theta (jnp.array): theta matrix with logarithmic entries
        state_joint (jnp.array): bitstring, genotypes of coupled sequencing data
        n_prim (int): number of events, that occured in the primary tumor
        n_met (int): number of events, that occured in the metastasis
        lam1 (jnp.array): Rate \lambda_1 of the first sampling
        lam2 (jnp.array): Rate \lambda_2 of the second sampling

    Returns:
        tuple: score, grad wrt. to theta, grad wrt. lam_1
    """
    prim = state_joint.at[0::2].get()
    met = jnp.append(state_joint.at[1::2].get(), 1)
    p0 = jnp.zeros(2**(n_prim + n_met - 1))
    p0 = p0.at[0].set(1.)
    
    # Joint and met-marginal distribution at first sampling
    pTh1_joint = lam1 * R_i_inv_vec(log_theta, p0, lam1, state_joint, transpose = False)
    p0 = jnp.zeros(2**n_prim)
    p0 = p0.at[0].set(1.)
    g_1, pTh1_marg = mhn.gradient(theta_prim, lam1, prim, p0)
    nk = pTh1_marg.at[-1].get()
    
    # Select the states where x = prim and z are compatible with met 
    poss_states = obs_states((n_prim + n_met -1), state_joint, True)
    poss_states_inds = jnp.where(poss_states, size=2**(n_met-1))[0]
    # Prim conditional distribution at first sampling, to be used as starting dist for second sampling
    pTh1_cond_obs = pTh1_joint.at[poss_states_inds].get()
    # States where the Seeding didn't happen aren't compatible with met and get probability 0
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_met-1)), pTh1_cond_obs)
    pTh1_cond_obs *= 1/nk
    
    # Derivative of pTh2 = M lam_2*(lam2*I-Q)^(-1)pth1_cond
    g_2, pTh2_marg = mhn.gradient(log_theta, lam2, met, pTh1_cond_obs)
    # lhs = (pD/pTh_2)^T M lam2*(lam2*I-Q)^(-1)
    q = jnp.zeros(2**n_met)
    q = q.at[-1].set(1/pTh2_marg.at[-1].get())
    q = lam2 * mhn.R_inv_vec(log_theta, q, lam2, met, transpose = True)
    lhs = jnp.zeros(2**(n_prim + n_met - 1))
    lhs =  lhs.at[poss_states_inds].set(q.at[2**(n_met - 1):].get())
    
    # Derivative of pth1_cond
    q = R_i_inv_vec(log_theta, lhs, lam1, state_joint, transpose = True)
    g_3 = x_partial_Q_y(log_theta, q, pTh1_joint, state_joint)/nk
    q = jnp.zeros(2**(n_prim))
    q = q.at[-1].set(1/nk)
    q = mhn.R_inv_vec(theta_prim, q, lam1, prim, transpose = True)
    g_4 = -mhn.x_partial_Q_y(theta_prim, q, pTh1_marg, prim)

    # Partial derivative wrt. lambda1
    R1pth1 = R_i_inv_vec(log_theta, pTh1_joint, lam1, state_joint, transpose = False)/nk
    R1pth1 = R1pth1 * poss_states
    dlam1 = 1/lam1 - jnp.dot(lhs, R1pth1) 
    
    score =  jnp.log(nk) + jnp.log(pTh2_marg.at[-1].get())
    return score, g_1 + g_2 + g_3 + g_4, dlam1 * lam1