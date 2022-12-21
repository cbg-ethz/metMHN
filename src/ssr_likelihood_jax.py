import numpy as np

from ssr_kronvec_jax import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed, kronvec, kron_diag
import Utilityfunctions as utils
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


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


@partial(jit, static_argnames=["n"])
def x_partial_Q_y(log_theta: np.array, x: np.array, y: np.array, state: np.array, n: int) -> np.array:
    """This function computes x \partial Q y with \partial Q the Jacobian of Q w.r.t. all thetas
    efficiently using the shuffle trick (sic!).

    Args:
        log_theta (np.array): Logarithmic theta values of the MHN 
        x (np.array): x vector to multiply with from the left. Length must equal the number of
        nonzero entries in the state vector.
        y (np.array): y vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        state (np.array): Binary state vector, representing the current sample's events.

    Returns:
        np.array: x \partial_(\Theta_{ij}) Q y for i, j = 1, ..., n+1
    """
    z = jnp.zeros(shape=(n + 1, n + 1))

    for i in range(n):
        z_sync = x * kronvec_sync(log_theta=jnp.array(log_theta),
                                  p=jnp.array(y), i=i, n=n, state=jnp.array(state))
        z_prim = x * kronvec_prim(log_theta=jnp.array(log_theta),
                                  p=jnp.array(y), i=i, n=n, state=jnp.array(state))
        z_met = x * kronvec_met(log_theta=jnp.array(log_theta),
                                p=jnp.array(y), i=i, n=n, state=state)

        z = z.at[i, -1].set(z_met.sum())

        def body_fun(j, val):

            _z_sync, _z_prim, _z_met, _z = lax.switch(
                state.at[2*j].get() + 2 * state.at[2*j+1].get(),
                [
                    lambda s, p, m: (s, p, m, 0.),
                    f1,
                    f2,
                    f3
                ],
                val[0],
                val[1],
                val[2],

            )
            return _z_sync, _z_prim, _z_met, val[3].at[i, j].set(_z)

        z_sync, z_prim, z_met, z = lax.fori_loop(
            lower=0,
            upper=i,
            body_fun=body_fun,
            init_val=(z_sync, z_prim, z_met, z)
        )

        z_sync, z_prim, z_met, _z = lax.switch(
            state.at[2*i].get() + 2 * state.at[2*i+1].get(),
            [
                lambda s, p, m: (
                    s, p, m, (sum(s) + sum(p) + sum(m)).astype(float)),
                t12,
                t12,
                t3
            ],
            z_sync,
            z_prim,
            z_met,
        )
        z = z.at[i, i].set(_z)

        z_sync, z_prim, z_met, z = lax.fori_loop(
            lower=i+1,
            upper=n,
            body_fun=body_fun,
            init_val=(z_sync, z_prim, z_met, z)
        )

    z_seed = x * kronvec_seed(log_theta=jnp.array(log_theta),
                              p=jnp.array(y), n=n, state=jnp.array(state))

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


@partial(jit, static_argnames=["state_size", "transpose"])
def R_i_inv_vec(log_theta: np.array, x: np.array, lam: float,  state: np.array, state_size: int, transpose: bool = False) -> np.array:
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

    lidg = -1 / (kron_diag(log_theta=jnp.array(log_theta), n=n,
                 state=jnp.array(state), state_size=state_size) - lam)
    y = lidg * x

    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=lambda _, val: lidg * (kronvec(log_theta=jnp.array(log_theta), p=val, n=n,
                                                state=jnp.array(state), diag=False, transpose=transpose, state_size=state_size) + x),
        init_val=y
    )

    return y


@partial(jit, static_argnames=["state_size", "n"])
def gradient(log_theta: jnp.array, p_D: jnp.array, lam1: float, lam2: float, state: jnp.array, state_size: int, n: int) -> jnp.array:
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
        state=state,
        state_size=state_size)

    R_2_inv_p_0 = R_i_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam2,
        state=state,
        state_size=state_size
    )

    p_theta = lam * (R_1_inv_p_0 - R_2_inv_p_0)

    # some states are not reachable and therefore have zero probability density
    # minuend = p_D * np.divide(lam, p_theta, where=reachable[restricted])
    minuend = p_D * jnp.divide(lam, p_theta)
    minuend = jnp.nan_to_num(minuend, posinf=0., neginf=0.)
    subtrahend = minuend.copy()
    minuend = R_i_inv_vec(log_theta=log_theta, x=minuend,
                          lam=lam2, state=state, state_size=state_size, transpose=True)
    subtrahend = R_i_inv_vec(log_theta=log_theta, x=minuend,
                             lam=lam1, state=state, state_size=state_size, transpose=True)
    minuend = x_partial_Q_y(log_theta=log_theta,
                            x=minuend, y=R_2_inv_p_0, state=state, n=n)
    subtrahend = x_partial_Q_y(
        log_theta=log_theta, x=subtrahend, y=R_1_inv_p_0, state=state, n=n)

    return minuend - subtrahend


# def log_likelihood(log_theta: np.array, p_D: np.array, lam1: float, lam2: float, state: np.array) -> float:

#     reachable = utils.reachable_states(log_theta.shape[0] - 1)
#     if np.any(p_D[~reachable] != 0):
#         raise ValueError("The data vector contains unreachable states.")

#     p_0 = np.zeros(2 ** state.sum())
#     p_0[0] = 1
#     p_th = R_i_inv_vec(
#         log_theta=log_theta,
#         x=p_0,
#         lam=lam2,
#         state=state
#     ) - R_i_inv_vec(
#         log_theta=log_theta,
#         x=p_0,
#         lam=lam1,
#         state=state
#     )
#     p_th *= lam1 * lam2 / (lam1 - lam2)

#     return np.dot(
#         p_D[utils.ssr_to_fss(state) & reachable],
#         np.log(p_th[reachable[utils.ssr_to_fss(state)]])
#     )

if __name__ == "__main__":
    n = 4
    log_theta = utils.random_theta(n, 0.4)
    lam1 = np.random.exponential(10, 1)
    lam2 = np.random.exponential(10, 1)
    n_ss = 0
    while n_ss < 2:
        state = np.random.randint(2, size=2*n+1)
        n_ss = state.sum()

    p0 = np.zeros(1 << n_ss)
    p0[0] = 1
    p = R_i_inv_vec(log_theta=log_theta, x=p0,
                    lam=lam1, state=state, state_size=n_ss)
    q = R_i_inv_vec(log_theta=log_theta, x=p0,
                    lam=lam1, state=state, transpose=True, state_size=n_ss)
    x_partial_Q_y(
        log_theta=jnp.array(log_theta),
        x=jnp.array(p), y=jnp.array(q), state=jnp.array(state), n=n)
