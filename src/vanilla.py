import numpy as np
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial
from ssr_kronvec_jax import k2d1t, k2ntt, k2dt0


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


@partial(jit, static_argnames=["diag", "n", "transpose"])
def kronvec_i(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    n: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:

    @partial(jit, static_argnames=["diag", "n", "transpose"])
    def _kronvec(
        log_theta: jnp.array,
        p: jnp.array,
        i: int,
        n: int,
        state: jnp.array,
        diag: bool = True,
        transpose: bool = False
    ) -> jnp.array:

        y = p.copy()

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

        # Diagonal Kronecker factors
        y = lax.fori_loop(lower=0, upper=i,
                          body_fun=loop_body_diag, init_val=y)

        # Non-diagonal Kronecker factor
        y = lax.switch(
            index=state.at[i].get(),
            branches=[
                lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
                lambda x: k2ntt(p=x, theta=jnp.exp(
                    log_theta.at[i, i].get()), diag=diag, transpose=transpose),
            ],
            operand=y
        )

        # Diagonal Kronecker factors
        y = lax.fori_loop(lower=i+1, upper=n,
                          body_fun=loop_body_diag, init_val=y)

        return y

    return lax.cond(
        not diag and state[i] != 1,
        lambda: jnp.zeros_like(p),
        lambda: _kronvec(
            log_theta=log_theta,
            p=p,
            i=i,
            n=n,
            state=state,
            diag=diag,
            transpose=transpose
        ),
    )


@ partial(jit, static_argnames=["n", "diag", "transpose", "state_size"])
def kronvec(log_theta: jnp.array, p: jnp.array, n: int, state: jnp.array, state_size: int, diag: bool = True, transpose: bool = False) -> jnp.array:
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
                         n=n, state=state, diag=diag, transpose=transpose)

        return val

    return lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=jnp.zeros(shape=2**state_size)
    )


@ partial(jit, static_argnames=["n", "state_size"])
def kron_diag_i(
        log_theta: jnp.array,
        i: int,
        n: int,
        state: jnp.array,
        state_size: int) -> jnp.array:

    diag = jnp.ones(2 ** state_size)

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


@partial(jit, static_argnames=["n", "state_size"])
def kron_diag(
        log_theta: jnp.array,
        n: int,
        state: jnp.array,
        state_size: int) -> jnp.array:

    def body_fun(i, val):

        val += kron_diag_i(log_theta=log_theta, i=i, n=n,
                           state=state, state_size=state_size)
        return val

    return lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=jnp.zeros(shape=2**state_size)
    )


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
    n = log_theta.shape[0]

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


@partial(jit, static_argnames=["n"])
def x_partial_Q_y(
        log_theta: np.array,
        x: np.array,
        y: np.array,
        state: np.array,
        n: int) -> np.array:
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

    val = jnp.zeros(shape=(n, n))

    def body_fun(i, val):

        z = x * kronvec_i(log_theta=log_theta,
                          p=y, i=i, n=n, state=state)

        def body_fun(j, val):

            z, _val = lax.switch(
                state.at[j].get(),
                [
                    lambda x: (x, x.sum()),
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
