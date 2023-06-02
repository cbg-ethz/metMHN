import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from .kronvec import k2d1t, k2ntt, k2dt0


@jit
def t(z: jnp.array) -> tuple[jnp.array, float]:
    z = z.reshape((-1, 2), order="C")
    val = z.sum()
    return z.flatten(order="F"), val


@jit
def f(z: jnp.array) -> tuple[jnp.array, float]:
    z = z.reshape((-1, 2), order="C")
    val = z[:, 1].sum()
    return z.flatten(order="F"), val


@partial(jit, static_argnames=["diag", "transpose"])
def _kronvec(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:

    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[j].get(),
            branches=[
                lambda x: x,
                lambda x: k2d1t(p=x, theta=jnp.exp(
                    log_theta.at[i, j].get()))
            ],
            operand=val
        )

        return val
    n = log_theta.shape[0]
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=i,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        index=state.at[i].get(),
        branches=[
            lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
            lambda x: k2ntt(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose),
        ],
        operand=p
    )

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=n,
                      body_fun=loop_body_diag, init_val=p)

    return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_i(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:
    return lax.cond(
        not diag and state[i] != 1,
        lambda: jnp.zeros_like(p),
        lambda: _kronvec(
            log_theta=log_theta,
            p=p,
            i=i,
            state=state,
            diag=diag,
            transpose=transpose
        ),
    )


@ partial(jit, static_argnames=["diag", "transpose"])
def kronvec(log_theta: jnp.array, p: jnp.array, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:
    """This computes the restricted version of the product of the rate matrix Q with a vector Q p.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        p (jnp.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q before multiplying. Defaults to False.

    Returns:
        jnp.array: Q p
    """

    def body_fun(i, val):

        val += kronvec_i(log_theta=log_theta, p=p, i=i,
                         state=state, diag=diag, transpose=transpose)

        return val
    n = log_theta.shape[0]
    state_size = np.log2(p.shape[0]).astype(int)
    return lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=jnp.zeros(shape=2**state_size)
    )


@jit
def kron_diag_i(
        log_theta: jnp.array,
        i: int,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:

    n = log_theta.shape[0]

    def loop_body(j, val):
        val = lax.switch(
            index=state.at[j].get(),
            branches=[
                lambda val: val,
                lambda val: k2d1t(val, jnp.exp(log_theta[i, j])),
            ],
            operand=val
        )
        return val

    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag
    )

    diag = lax.switch(
        index=state.at[i].get(),
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
        ],
        operand=diag
    )

    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    return diag


@jit
def kron_diag(
        log_theta: jnp.array,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:

    def body_fun(i, val):
        val += kron_diag_i(log_theta=log_theta, i=i, state=state, diag=diag)
        return val

    n = log_theta.shape[0]

    return lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=jnp.zeros_like(diag)
    )


@partial(jit, static_argnames=["transpose"])
def R_inv_vec(log_theta: jnp.array, x: jnp.array, lam: float,  state: jnp.array, transpose: bool = False) -> jnp.array:
    """This computes R^{-1} x = (\lambda I - Q)^{-1} x

    Args:
        log_theta (np.array): Log values of the theta matrix
        x (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        lam (float): Value of \lambda_i
        state (np.array): Binary state vector, representing the current sample's events.


    Returns:
        np.array: R_i^{-1} x
    """
    state_size = jnp.log2(x.shape[0]).astype(int)

    lidg = -1 / (kron_diag(log_theta=log_theta,
                 state=state, diag=jnp.ones_like(x)) - lam)
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
def x_partial_Q_y(
        log_theta: jnp.array,
        x: jnp.array,
        y: jnp.array,
        state: jnp.array) -> jnp.array:
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
    n = log_theta.shape[0]
    val = jnp.zeros(shape=(n, n))

    def body_fun(i, val):

        z = x * kronvec_i(log_theta=log_theta,
                          p=y, i=i, state=state)

        def body_fun(j, val):

            z, _val = lax.switch(
                state.at[j].get(),
                [
                    lambda x: (x, 0.),
                    lambda x: f(x)
                ],
                val[0],

            )
            return z, val[1].at[j].set(_val)

        z, val = lax.fori_loop(
            lower=0,
            upper=i,
            body_fun=body_fun,
            init_val=(z, val)
        )

        z, _val = lax.switch(
            state.at[i].get(),
            [
                lambda z: (z, z.sum()),
                lambda z: t(z)
            ],
            z,
        )
        val = val.at[i].set(_val)

        z, val = lax.fori_loop(
            lower=i+1,
            upper=n,
            body_fun=body_fun,
            init_val=(z, val)
        )

        return val

    val = vmap(body_fun, in_axes=(0, 0), out_axes=0)(
        jnp.arange(n, dtype=int), val)

    return val


@jit
def gradient(log_theta: jnp.array, lam: float, state: jnp.array, p_0: jnp.array) -> jnp.array:
    """This computes the gradient of the score function, which is the log-likelihood of a data vector p_D
    with respect to the log_theta matrix

    Args:
        log_theta (np.array): Log values of the theta matrix.
        p_D (np.array): Data vector.
        lam (float): Rate of the sampling.
        state (np.array): Binary state vector, representing the current sample's events.


    Returns:
        np.array: \partial_theta (p_D^T log p_theta)
    """
    p_theta = R_inv_vec(log_theta=log_theta, x=p_0, lam=lam,
                        state=state)
    x = jnp.zeros_like(p_theta)
    x = x.at[-1].set(1/p_theta.at[-1].get())
    x = R_inv_vec(log_theta=log_theta, x=x, lam=lam,
                  state=state, transpose=True)

    return x_partial_Q_y(log_theta=log_theta,
                         x=x, y=p_theta, state=state), p_theta
