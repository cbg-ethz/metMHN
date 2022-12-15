import numpy as np
from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import jax
# Kronecker factors


@jit
def k2dt0(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([-theta, 0.])
    p = jax.vmap(lambda a, x: a * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k2d0t(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([0., theta])
    p = jax.vmap(lambda a, x: a * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k2dtt(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    p = jax.vmap(lambda x: -theta * x, 0, 0)(p)
    return p.flatten(order="F")


@jit
def k2d1t(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([1., theta])
    p = jax.vmap(lambda a, x: a * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k2d10(p: jnp.array) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([1., 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k2d01(p: jnp.array) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([0., 1.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k2d11(p: jnp.array) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    return p.flatten(order="F")


@jit
def k2ntt(p: jnp.array, theta: float, diag: bool = True, transpose: bool = False) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    p = lax.cond(
        diag,
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[-theta, 0.], [theta, 0.]]),
            lambda x: x @ jnp.array([[-theta, theta], [0., 0.]]),
            operand=p
        ),
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[0., 0.], [theta, 0.]]),
            lambda x: x @ jnp.array([[0., theta], [0., 0.]]),
            operand=p
        ),
        operand=p
    )
    return p.flatten(order="F")


@jit
def k4ns(p: jnp.array, theta: float, diag: bool = True, transpose: bool = False) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    p = lax.cond(
        diag,
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[-theta, 0., 0., 0.],
                                    [0., 0., 0., 0.], [0., 0., 0., 0.], [theta, 0., 0., 0.]]),
            lambda x: x @ jnp.array([[-theta, 0., 0., theta],
                                    [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
            operand=p),
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                    [0., 0., 0., 0.], [theta, 0., 0., 0.]]),
            lambda x: x @ jnp.array([[0., 0., 0., theta],
                                    [0., 0., 0., 0.], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
            operand=p
        ),
        operand=p
    )
    return p.flatten(order="F")


@jit
def k4np(p: jnp.array, theta: float, diag: bool = True, transpose: bool = False) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    p = lax.cond(
        diag,
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[-theta, 0., 0., 0.], [theta,
                                    0., 0., 0.], [0., 0., -theta, 0.], [0., 0., theta, 0.]]),
            lambda x: x @ jnp.array([[-theta, theta, 0., 0.],
                                    [0., 0., 0., 0.], [0., 0., -theta, theta], [0., 0., 0., 0.]]),
            operand=p
        ),
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[0., 0., 0., 0.], [theta,
                                    0., 0., 0.], [0., 0., 0., 0.], [0., 0., theta, 0.]]),
            lambda x: x @ jnp.array([[0., theta, 0., 0.],
                                    [0., 0., 0., 0.], [0., 0., 0., theta], [0., 0., 0., 0.]]),
            operand=p
        ),
        operand=p
    )
    return p.flatten(order="F")


@jit
def k4nm(p: jnp.array, theta: float, diag: bool = True, transpose: bool = False) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    p = lax.cond(
        diag,
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[-theta, 0., 0., 0.], [0., -
                                    theta, 0., 0.], [theta, 0., 0., 0.], [0., theta, 0., 0.]]),
            lambda x: x @ jnp.array([[-theta, 0., theta, 0.],
                                    [0., -theta, 0., theta], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
            operand=p
        ),
        lambda p: lax.cond(
            transpose,
            lambda x: x @ jnp.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                    [theta, 0., 0., 0.], [0., theta, 0., 0.]]),
            lambda x: x @ jnp.array([[0., 0., theta, 0.],
                                    [0., 0., 0., theta], [0., 0., 0., 0.], [0., 0., 0., 0.]]),
            operand=p
        ),
        operand=p
    )
    return p.flatten(order="F")


@jit
def k4d100t(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    # p = p.at[:, 3].multiply(theta).at[:, [1, 2]].set(0.)
    theta_slice = jnp.array([1., 0., 0., theta])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k4dt000(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, 0., 0., 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k4dtt00(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, -theta, 0., 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k4dt0t0(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, 0., -theta, 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k4d1t1t(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    # p = p.at[:, [1, 3]].multiply(theta)
    theta_slice = jnp.array([1., theta, 1., theta])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


@jit
def k4d11tt(p: jnp.array, theta: float) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., 1., theta, theta])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    # p = p.at[:, [2, 3]].multiply(theta)
    return p.flatten(order="F")


@partial(jit, static_argnames=["i", "diag", "n", "transpose"])
def kronvec_sync(log_theta: jnp.array, p: jnp.array, i: int, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:
    """This computes the restricted version of the product of the synchronized part of the ith Q summand
    Q_i with a vector Q_i p.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        p (jnp.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.array: Q_i p
    """

    return lax.cond(not diag and sum(state[2 * i: 2 * i + 2]) != 2,
                    lambda: jnp.zeros_like(p),
                    lambda: _kronvec_sync(
        log_theta=log_theta,
        p=p,
        i=i,
        n=n,
        state=state,
        diag=diag,
        transpose=transpose
    ),
    )


@partial(jit, static_argnames=["i", "diag", "n", "transpose"])
def _kronvec_sync(log_theta: jnp.array, p: jnp.array, i: int, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:

    y = p.copy()

    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d10(x),
                lambda x: k2d10(x),
                lambda x: k4d100t(p=x, theta=jnp.exp(log_theta.at[i, j].get()))
            ],
            operand=val
        )

        return val

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=0, upper=i, body_fun=loop_body_diag, init_val=y)

    # Non-diagonal Kronecker factor
    y = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
            lambda x: k2dt0(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k2dt0(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k4ns(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose)
        ],
        operand=y
    )

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=i+1, upper=n, body_fun=loop_body_diag, init_val=y)

    # Last Kronecker factor
    y = lax.cond(
        state[-1] == 1,
        lambda: k2d10(y),
        lambda: y
    )

    return y


@partial(jit, static_argnames=["i", "diag", "n", "transpose"])
def kronvec_prim(log_theta: jnp.array, p: jnp.array, i: int, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:
    """This computes the restricted version of the product of the asynchronous primary tumour
    part of the ith Q summand Q_i with a vector Q_i p.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        p (jnp.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.array: Q_i p
    """

    # there are no non-diagonal entries if event i is not mutated in prim
    return lax.cond(
        not diag and state[2 * i] == 0,
        lambda: jnp.zeros_like(p),
        lambda: lax.cond(
            state[-1] == 0,
            lambda: jnp.zeros_like(p),
            lambda: _kronvec_prim(
                log_theta=log_theta,
                p=p,
                i=i,
                n=n,
                state=state,
                diag=diag,
                transpose=transpose
            ),
        )
    )

    return y


@ partial(jit, static_argnames=["i", "diag", "n", "transpose"])
def _kronvec_prim(log_theta: jnp.array, p: jnp.array, i: int, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:

    y = p.copy()

    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d1t(x, theta=jnp.exp(log_theta.at[i, j].get())),
                lambda x: k2d11(x),
                lambda x: k4d1t1t(p=x, theta=jnp.exp(log_theta.at[i, j].get()))
            ],
            operand=val
        )
        return val

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=0, upper=i, body_fun=loop_body_diag, init_val=y)

    # Non-diagonal Kronecker factor
    y = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
            lambda x: k2ntt(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose),
            lambda x: k2dtt(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k4np(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose)
        ],
        operand=y
    )

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=i+1, upper=n, body_fun=loop_body_diag, init_val=y)

    # Last Kronecker factor
    y = k2d01(y)

    return y


@ partial(jit, static_argnames=["i", "diag", "n", "transpose"])
def kronvec_met(log_theta: jnp.array, p: jnp.array, i: int, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:
    """This computes the restricted version of the product of the asynchronous metastasis
    part of the ith Q summand Q_i with a vector Q_i p.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        p (jnp.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.array: Q_i p
    """

    y = p.copy()

    # there are no non-diagonal entries if event i is not mutated in both prim and met
    y = lax.cond(not diag and state[2 * i + 1] == 0,
                 lambda: jnp.zeros_like(y),
                 lambda: y,
                 )
    y = lax.cond(state[-1] == 0,
                 lambda: jnp.zeros_like(y),
                 lambda: y,
                 )

    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d11(x),
                lambda x: k2d1t(x, theta=jnp.exp(log_theta.at[i, j].get())),
                lambda x: k4d11tt(p=x, theta=jnp.exp(log_theta.at[i, j].get()))
            ],
            operand=val
        )

        return val

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=0, upper=i, body_fun=loop_body_diag, init_val=y)

    # Non-diagonal Kronecker factor
    y = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda x: x * -jnp.exp(log_theta.at[i, i].get()),
            lambda x: k2dtt(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k2ntt(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose),
            lambda x: k4nm(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose)
        ],
        operand=y
    )

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=i+1, upper=n, body_fun=loop_body_diag, init_val=y)

    # Last Kronecker factor
    y = k2d0t(y, theta=jnp.exp(log_theta.at[i, n].get()))

    return y


@ partial(jit, static_argnames=["n", "diag",
                                # "transpose"
                                ])
def kronvec_seed(log_theta: jnp.array, p: jnp.array, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:
    """This computes the restricted version of the product of the seeding summand of Q with a vector Q_M p.


    Args:
        log_theta (jnp.array): Log values of the theta matrix
        p (jnp.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_M (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_M before multiplying. Defaults to False.

    Returns:
        jnp.array: Q_seed p
    """

    y = p.copy()

    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d10(x),
                lambda x: k2d10(x),
                lambda x: k4d100t(p=x, theta=jnp.exp(log_theta.at[n, j].get()))
            ],
            operand=val
        )

        return val

    # Diagonal Kronecker factors
    y = lax.fori_loop(lower=0, upper=n, body_fun=loop_body_diag, init_val=y)

    # Non-diagonal Kronecker factor
    y = lax.cond(
        state[-1] == 1,
        lambda x: k2ntt(x, theta=jnp.exp(
            log_theta.at[n, n].get()), diag=diag, transpose=transpose),
        lambda x: x * -jnp.exp(log_theta.at[n, n].get()),
        operand=y
    )

    # there are no non-diagonal entries if event i is not mutated in both prim and met
    y = lax.cond(not diag and state[-1] == 0,
                 lambda: jnp.zeros_like(y),
                 lambda: y,
                 )

    return y


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
    y = jnp.zeros(shape=2**state_size)

    log_theta = jnp.array(log_theta)
    p = jnp.array(p)
    state = jnp.array(state)

    for i in range(n):
        y += kronvec_sync(log_theta=log_theta, p=p, i=i,
                          n=n, state=state, diag=diag, transpose=transpose)
        y += kronvec_prim(log_theta=log_theta, p=p, i=i,
                          n=n, state=state, diag=diag, transpose=transpose)
        y += kronvec_met(log_theta=log_theta, p=p, i=i,
                         n=n, state=state, diag=diag, transpose=transpose)

    y += kronvec_seed(log_theta=log_theta, p=p, n=n,
                      state=state, diag=diag, transpose=transpose)

    return y


@ partial(jit, static_argnames=["i", "n", "state_size"])
def kron_sync_diag(log_theta: jnp.array, i: int, n: int, state: jnp.array, state_size: int) -> jnp.array:
    """This computes the diagonal of the synchronized part of the ith Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.

    Returns:
        jnp.array: diag(Q_i_sync)
    """
    diag = jnp.ones(2 ** state_size)

    def loop_body(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda val: val,
                lambda val: k2d10(val),
                lambda val: k2d10(val),
                lambda val: k4d100t(val, jnp.exp(log_theta[i, j]))
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
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
            lambda val: k4dt000(val, jnp.exp(log_theta[i, i]))
        ],
        operand=diag
    )

    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    diag = lax.cond(
        state[-1] == 1,
        lambda x: k2d10(x),
        lambda x: x,
        operand=diag
    )

    return diag


@ partial(jit, static_argnames=["i", "n", "state_size"])
def kron_prim_diag(log_theta: jnp.array, i: int, n: int, state: jnp.array, state_size: int) -> jnp.array:
    """This computes the diagonal of the asynchronous primary tumour part of the ith
    Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.

    Returns:
        jnp.array: diag(Q_i_prim)
    """

    diag = lax.cond(
        state[-1] == 0,
        lambda: jnp.zeros(2 ** state_size),
        lambda: jnp.ones(2 ** state_size),
    )

    def loop_body(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda val: val,
                lambda val: k2d1t(val, jnp.exp(log_theta[i, j])),
                lambda val: k2d11(val),
                lambda val: k4d1t1t(val, jnp.exp(log_theta[i, j]))
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
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
            lambda val: k2dtt(val, jnp.exp(log_theta[i, i])),
            lambda val: k4dt0t0(val, jnp.exp(log_theta[i, i]))
        ],
        operand=diag
    )

    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    diag = k2d01(diag)

    return diag


@ partial(jit, static_argnames=["i", "n", "state_size"])
def kron_met_diag(log_theta: jnp.array, i: int, n: int, state: jnp.array, state_size: int) -> jnp.array:
    """This computes the diagonal of the asynchronous metastasis part of the ith
    Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.

    Returns:
        jnp.array: diag(Q_i_met)
    """
    diag = lax.cond(
        state[-1] == 0,
        lambda: jnp.zeros(2 ** state_size),
        lambda: jnp.ones(2 ** state_size),
    )

    def loop_body(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda val: val,
                lambda val: k2d11(val),
                lambda val: k2d1t(val, jnp.exp(log_theta[i, j])),
                lambda val: k4d11tt(val, jnp.exp(log_theta[i, j]))
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
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dtt(val, jnp.exp(log_theta[i, i])),
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
            lambda val: k4dtt00(val, jnp.exp(log_theta[i, i]))
        ],
        operand=diag
    )

    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    diag = k2d0t(diag, jnp.exp(log_theta[i, n]))

    return diag


@ partial(jit, static_argnames=["n", "state_size"])
def kron_seed_diag(log_theta: jnp.array, n: int, state: jnp.array, state_size: int) -> jnp.array:
    """This computes the diagonal of the seeding summand of Q.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.

    Returns:
        jnp.array: diag(Q_seed)
    """

    diag = jnp.ones(2 ** state_size)

    def loop_body(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda val: val,
                lambda val: k2d10(val),
                lambda val: k2d10(val),
                lambda val: k4d100t(val, jnp.exp(log_theta[n, j]))
            ],
            operand=val
        )
        return val

    diag = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    diag = lax.cond(
        state[-1] == 1,
        lambda x: k2dt0(x, jnp.exp(log_theta[-1, -1])),
        lambda x: x * -jnp.exp(log_theta[-1, -1]),
        operand=diag
    )

    return diag


@ partial(jit, static_argnames=["n", "state_size"])
def kron_diag(log_theta: jnp.array, n: int, state: jnp.array, state_size: int) -> jnp.array:
    """This computes diagonal of the rate matrix Q.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.

    Returns:
        jnp.array: diag(Q)
    """
    y = jnp.zeros(shape=2**state_size)

    for i in range(n):
        y += kron_sync_diag(log_theta=log_theta, i=i,
                            n=n, state=state, state_size=state_size)
        y += kron_prim_diag(log_theta=log_theta, i=i,
                            n=n, state=state, state_size=state_size)
        y += kron_met_diag(log_theta=log_theta, i=i,
                           n=n, state=state, state_size=state_size)
    y += kron_seed_diag(log_theta=log_theta, n=n,
                        state=state, state_size=state_size)

    return y
