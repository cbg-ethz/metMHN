import numpy as np

from ssr_kronvec_jax import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed, kronvec, kron_diag, obs_inds, marg_transp
import Utilityfunctions as utils
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
import vanilla as mhn


@jit
def f1(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = p[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


@jit
def f2(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = m[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


@jit
def f3(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s[:, 3].sum() + p[:, [1, 3]].sum() + m[:, [2, 3]].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


@jit
def t12(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = s[:, 0].sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


@jit
def t3(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s.sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


@jit
def z3(s: jnp.array) -> tuple[jnp.array, float]:
    s = s.reshape((-1, 4), order="C")

    z = s[:, 3].sum()

    s = s.flatten(order="F")

    return s, z


@jit
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

    def body_fun(i, val):

        z_sync = x * kronvec_sync(log_theta=log_theta,
                                  p=y, i=i, state=state)
        z_prim = x * kronvec_prim(log_theta=log_theta,
                                  p=y, i=i, state=state)
        z_met = x * kronvec_met(log_theta=log_theta,
                                p=y, i=i, state=state)

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
                    s, p, m, (sum(s) + sum(p) + sum(m))),
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

    z = z.at[:-1, :].set(vmap(body_fun, in_axes=(0, 0),
                         out_axes=0)(jnp.arange(n, dtype=int), z[:-1, :]))

    z_seed = x * kronvec_seed(log_theta=log_theta, p=y, state=state)

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
def R_i_inv_vec(log_theta: jnp.array, x: jnp.array, lam: float,  state: jnp.array, transpose: bool = False) -> jnp.array:
    """This computes R_i^{-1} x = (\lambda_i I - Q)^{-1} x

    Args:
        log_theta (np.array): Log values of the theta matrix
        x (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        lam (float): Value of \lambda_i
        state (np.array): Binary state vector, representing the current sample's events.

    Returns:
        np.array: R_i^{-1} x
    """
    n = log_theta.shape[0] - 1
    state_size = np.log2(x.shape[0]).astype(int)
    lidg = -1 / (kron_diag(log_theta=log_theta, state=state, p_in=x) - lam)
    y = lidg * x

    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=lambda _, val: lidg * (kronvec(log_theta=log_theta, p=val,
                                                state=state, diag=False, transpose=transpose) + x),
        init_val=y
    )

    return y


@partial(jit, static_argnames=["state_size"])
def gradient(log_theta: jnp.array, p_D: jnp.array, lam1: float, lam2: float, state: jnp.array, state_size: int) -> jnp.array:
    """This computes the gradient of the score function, which is the log-likelihood of a data vector p_D
    with respect to the log_theta matrix

    Args:
        log_theta (np.array): Log values of the theta matrix.
        p_D (np.array): Data vector.
        lam1 (float): Rate of the first sampling.
        lam2 (float): Rate of the second sampling.
        state (np.array): Binary state vector, representing the current sample's events.


    Returns:
        np.array: \partial_theta (p_D^T log p_theta)
    """

    # restricted = utils.ssr_to_fss(state)

    # reachable = utils.reachable_states(log_theta.shape[0] - 1)
    # if np.any(p_D[~reachable[restricted]] != 0):
    #     raise ValueError("The data vector contains unreachable states.")

    p_0 = jnp.zeros(2**state_size, dtype=float)
    p_0 = p_0.at[0].set(1.)
    lam = (lam1 * lam2 / (lam1 - lam2))
    R_1_inv_p_0 = R_i_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam1,
        state=state,)

    R_2_inv_p_0 = R_i_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam2,
        state=state,
    )

    p_theta = lam * (R_1_inv_p_0 - R_2_inv_p_0)

    # some states are not reachable and therefore have zero probability density
    # minuend = p_D * np.divide(lam, p_theta, where=reachable[restricted])
    minuend = p_D * jnp.divide(lam, p_theta)
    minuend = jnp.nan_to_num(minuend, posinf=0., neginf=0.)
    subtrahend = minuend.copy()
    minuend = R_i_inv_vec(log_theta=log_theta, x=minuend,
                          lam=lam2, state=state, transpose=True)
    subtrahend = R_i_inv_vec(log_theta=log_theta, x=minuend,
                             lam=lam1, state=state, transpose=True)
    minuend = x_partial_Q_y(log_theta=log_theta,
                            x=minuend, y=R_2_inv_p_0, state=state)
    subtrahend = x_partial_Q_y(
        log_theta=log_theta, x=subtrahend, y=R_1_inv_p_0, state=state)

    return minuend - subtrahend


@partial(jit, static_argnames=["prim_first"])
def _log_prob_coupled(log_theta: jnp.array, lam1: float, lam2: float, state: jnp.array, p0: jnp.array,
                      latent_dist: jnp.array, latent_state: jnp.array, prim_first: bool) -> float:
    """
    Evaluates the log probability of seeing coupled genotype data in state "state"
    Args:
        log_theta (jnp.array): Logarithmic entries of theta
        lam1 (float): Rate of first diagnosis
        lam2 (float): Rate of second diagnosis
        state (jnp.array): Bitstring state of prim at t1 and met at t2 if prim_first else vice versa
        m (int): Number of muations in state that happened
        prim_first (bool): If true: the primary tumor was first observed, else the met was first observed
    Returns:
        float: log(P(state))
    """
    n = log_theta.shape[0] - 1
    pTh1 = R_i_inv_vec(log_theta, p0, lam1,  state)
    pTh1 = obs_dist(pTh1, state, latent_dist, prim_first)
    obs_sum = pTh1.sum()
    log_theta = lax.cond(prim_first,
                         lambda x: x.at[0:n, -1].set(0.0),
                         lambda x: x,
                         operand=log_theta)
    pTh2 = lam2 * mhn.R_inv_vec(log_theta, pTh1/obs_sum, lam2, latent_state)
    return jnp.log(obs_sum) + jnp.log(pTh2.at[-1].get())


def log_prob_coupled(dat: jnp.array, log_theta: jnp.array, lam1: float, lam2: float):
    n = log_theta.shape[0]-1
    score = 0.0
    for i in range(dat.shape[0]):
        prim_first = bool(dat.at[i, -1].get())
        if prim_first:
            latent_state = jnp.append(
                dat.at[i, 1:2*n+1:2].get(), dat.at[i, -2].get())
        else:
            latent_state = dat.at[i, 0:2*n+1:2].get()
        latent_dist = jnp.zeros(2**int(latent_state.sum()))
        p0 = jnp.zeros(2**int(dat.at[i, 0:2*n+1].get().sum()))
        p0 = p0.at[0].set(1.0)
        score += _log_prob_coupled(log_theta, lam1, lam2, dat.at[i, 0:2*n+1].get(), p0,
                                   latent_dist, latent_state, prim_first)
    return score/dat.shape[0]


@jit
def _log_prob_single(log_theta: jnp.array, lam1: float, state: jnp.array, p0: jnp.array) -> float:
    pTh = lam1 * mhn.R_inv_vec(log_theta, p0, lam1,  state, False)
    return jnp.log(pTh.at[-1].get())


def log_prob_single(dat: jnp.array, log_theta: jnp.array, lam1: float):
    n = log_theta.shape[0] - 1
    score = 0.0
    for i in range(dat.shape[0]):
        marg_met = bool(dat.at[i, -1].get())
        if marg_met:
            state_obs = dat.at[i, 0:2*n+1:2].get()
            log_theta = log_theta.at[0:n, -1].set(0.0)
        else:
            state_obs = jnp.append(
                dat.at[i, 1:2*n+1:2].get(), dat.at[i, -2].get())
        p0 = jnp.zeros(2**int(state_obs.sum()))
        p0 = p0.at[0].set(1.0)
        score += _log_prob_single(log_theta, lam1, state_obs, p0)
    return score/dat.shape[0]


def grad_single(dat: jnp.array, log_theta: jnp.array, lam1: float):
    n = log_theta.shape[0] - 1
    g = jnp.zeros((n+1, n+1))
    for i in range(dat.shape[0]):
        marg_met = bool(dat.at[i, -1].get())
        if marg_met:
            state_obs = dat.at[i, 0:2*n+1:2].get()
            log_theta = log_theta.at[0:n, -1].set(0.0)
        else:
            state_obs = jnp.append(
                dat.at[i, 1:2*n+1:2].get(), dat.at[i, -2].get())
        p0 = jnp.zeros(2**int(state_obs.sum()))
        p0 = p0.at[0].set(1.0)
        g += mhn.gradient(log_theta, lam1, state_obs, p0)
    return g/dat.shape[0]


# @partial(jit, static_argnames=["prim_first"])
# def _grad_coupled(log_theta: jnp.array, lam1: float, lam2: float, state: jnp.array, p0: jnp.array,
# latent_dist: jnp.array, latent_state: jnp.array, prim_first: bool) -> jnp.array:
#
#    n = log_theta.shape[0] - 1
#    m = np.log2(p0.shape[0]).astype(int)
#    lm = np.log2(latent_dist.shape[0]).astype(int)
#
#    # Calculate joint distributiom at first sampling
#    pTh1 = R_i_inv_vec(log_theta, p0, lam1, state)
#    pTh1_obs = obs_dist(pTh1, state, latent_dist, prim_first)
#
#    # Perform marginalization over latent states
#    nk = pTh1_obs.sum()
#    q = jnp.zeros_like(p0)
#    pos = 2**(m - lm - 1) + 2**(m - 1) - 1
#    q = q.at[pos].set(1/nk)

    # Actual gradient
#    q = marg_transp(q, state, prim_first, False)
#    q = R_i_inv_vec(log_theta, q, lam1, state, transpose = True)
#    g_1 = x_partial_Q_y(log_theta, q, pTh1, state)
#    return g_1
