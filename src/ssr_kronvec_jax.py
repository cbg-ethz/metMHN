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


@partial(jit, static_argnames=["diag", "transpose"])
def _kronvec_sync(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.array:

    def loop_body_diag(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d10(x),
                lambda x: k2d10(x),
                lambda x: k4d100t(p=x, theta=jnp.exp(
                    log_theta.at[i, j].get()))
            ],
            operand=val)
        return val

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=i,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
            lambda x: k2dt0(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k2dt0(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k4ns(p=x, theta=jnp.exp(
            log_theta.at[i, i].get()), diag=diag, transpose=transpose)
        ],
        operand=p
    )

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=log_theta.shape[0]-1,
                        body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = lax.cond(
        state[-1] == 1,
        lambda: k2d10(p),
        lambda: p
    )

    return p

@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_sync(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:
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
    return lax.cond(
        not diag and lax.dynamic_slice(state, [2*i], [2]).sum() != 2,
        lambda: p*0.0,
        lambda: _kronvec_sync(
            log_theta=log_theta,
            p=p,
            i=i,
            state=state,
            diag=diag,
            transpose=transpose
        ), 
    )

@partial(jit, static_argnames=["diag", "transpose"])
def _kronvec_prim(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.array:

    def loop_body_diag(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d1t(x, theta=jnp.exp(
                    log_theta.at[i, j].get())),
                lambda x: k2d11(x),
                lambda x: k4d1t1t(p=x, theta=jnp.exp(
                    log_theta.at[i, j].get()))
            ],
            operand=val)
        return val

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=i,
                        body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda x: -jnp.exp(log_theta.at[i, i].get()) * x,
            lambda x: k2ntt(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose),
            lambda x: k2dtt(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k4np(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose)
        ],
        operand=p)

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=log_theta.shape[0]-1,
                        body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = k2d01(p)

    return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_prim(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:
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
            lambda: p*0.0,
            lambda: _kronvec_prim(
                log_theta=log_theta,
                p=p,
                i=i,
                state=state,
                diag=diag,
                transpose=transpose
            ),
        )
    )


@ partial(jit, static_argnames=["diag", "transpose"])
def _kronvec_met(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:
    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d11(x),
                lambda x: k2d1t(x, theta=jnp.exp(
                    log_theta.at[i, j].get())),
                lambda x: k4d11tt(p=x, theta=jnp.exp(
                    log_theta.at[i, j].get()))
            ],
            operand=val
        )

        return val

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=i,
                        body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda x: x * -jnp.exp(log_theta.at[i, i].get()),
            lambda x: k2dtt(p=x, theta=jnp.exp(log_theta.at[i, i].get())),
            lambda x: k2ntt(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose),
            lambda x: k4nm(p=x, theta=jnp.exp(
                log_theta.at[i, i].get()), diag=diag, transpose=transpose)
        ],
        operand=p
    )
    n = log_theta.shape[0]-1
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=n,
                        body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = k2d0t(p, theta=jnp.exp(log_theta.at[i, n].get()))

    return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_met(
    log_theta: jnp.array,
    p: jnp.array,
    i: int,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:
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
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    return lax.cond(
        not diag and state[2 * i + 1] == 0,
        lambda: p*0.0,
        lambda: lax.cond(
            state[-1] == 0,
            lambda: p*0.0,
            lambda: _kronvec_met(
                log_theta=log_theta,
                p=p,
                i=i,
                state=state,
                diag=diag,
                transpose=transpose
            ),
        ),
    )

@ partial(jit, static_argnames=["diag","transpose"])
def _kronvec_seed(
        log_theta: jnp.array,
        p: jnp.array,
        state: jnp.array,
        diag: bool = True,
        transpose: bool = False
    ) -> jnp.array:
        def loop_body_diag(j, val):

            val = lax.switch(
                index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
                branches=[
                    lambda x: x,
                    lambda x: k2d10(x),
                    lambda x: k2d10(x),
                    lambda x: k4d100t(p=x, theta=jnp.exp(
                        log_theta.at[log_theta.shape[0], j].get()))
                ],
                operand=val
            )

            return val
        n = log_theta.shape[0]-1
        # Diagonal Kronecker factors
        p = lax.fori_loop(lower=0, upper=n,
                          body_fun=loop_body_diag, init_val=p)

        # Non-diagonal Kronecker factor
        p = lax.cond(
            state[-1] == 1,
            lambda x: k2ntt(x, theta=jnp.exp(
                log_theta.at[n, n].get()), diag=diag, transpose=transpose),
            lambda x: x * -jnp.exp(log_theta.at[n, n].get()),
            operand=p
        )

        return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_seed(
    log_theta: jnp.array,
    p: jnp.array,
    state: jnp.array,
    diag: bool = True,
    transpose: bool = False
) -> jnp.array:
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
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    return lax.cond(
        not diag and state[-1] == 0,
        lambda: p*0.0,
        lambda: _kronvec_seed(
            log_theta=log_theta,
            p=p,
            state=state,
            diag=diag,
            transpose=transpose
        ),
    )


@ partial(jit, static_argnames=["diag", "transpose"])
def kronvec(log_theta: jnp.array, p: jnp.array, state: jnp.array, 
            diag: bool = True, transpose: bool = False) -> jnp.array:
    """
    This computes the restricted version of the product of the rate matrix Q with a vector Q p.

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

        val += kronvec_sync(log_theta=log_theta, p=p, i=i,
                            state=state, diag=diag, transpose=transpose)
        val += kronvec_prim(log_theta=log_theta, p=p, i=i,
                            state=state, diag=diag, transpose=transpose)
        val += kronvec_met(log_theta=log_theta, p=p, i=i,
                           state=state, diag=diag, transpose=transpose)

        return val

    n = log_theta.shape[0]-1
    m = np.log2(p.shape[0]).astype(int)
    y = jnp.zeros(2**m)
    y = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=y
    )

    y += kronvec_seed(log_theta=log_theta, p=p,
                      state=state, diag=diag, transpose=transpose)

    return y


@jit
def kron_sync_diag(
        log_theta: jnp.array,
        i: int,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:
    """This computes the diagonal of the synchronized part of the ith Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (jnp.array): Vector of ones of size 2 ** (number of 1s in state)

    Returns:
        jnp.array: diag(Q_i_sync)
    """
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

    n = log_theta.shape[0] - 1

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


@jit
def _kron_prim_diag(
        log_theta: jnp.array,
        i: int,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:

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

    n = log_theta.shape[0] - 1

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

@jit
def kron_prim_diag(
        log_theta: jnp.array,
        i: int,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:
    """This computes the diagonal of the asynchronous primary tumour part of the ith
    Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (jnp.array): Vector of ones of size 2 ** (number of 1s in state)

    Returns:
        jnp.array: diag(Q_i_prim)
    """

    return lax.cond(
        state[-1] == 0,
        lambda: diag * 0.0,
        lambda: _kron_prim_diag(
            log_theta=log_theta,
            i=i,
            state=state,
            diag=diag, 
        ),
    )

@jit
def _kron_met_diag(
        log_theta: jnp.array,
        i: int,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:

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
    n = log_theta.shape[0] - 1

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


@jit
def kron_met_diag(
        log_theta: jnp.array,
        i: int,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:
    """This computes the diagonal of the asynchronous metastasis part of the ith
    Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (jnp.array): Vector of ones of size 2 ** (number of 1s in state)

    Returns:
        jnp.array: diag(Q_i_met)
    """
    return lax.cond(
        state[-1] == 0,
        lambda: diag * 0.0,
        lambda: _kron_met_diag(
            log_theta=log_theta,
            i=i,
            state=state,
            diag=diag,
        ),
    )


@jit
def kron_seed_diag(
        log_theta: jnp.array,
        state: jnp.array,
        diag: jnp.array) -> jnp.array:
    """This computes the diagonal of the seeding summand of Q.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        diag (jnp.array): Vector of ones of size 2 ** (number of 1s in state)

    Returns:
        jnp.array: diag(Q_seed)
    """
    n = log_theta.shape[0] - 1

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


@jit
def kron_diag(log_theta: jnp.array, state: jnp.array, p_in: jnp.array) -> jnp.array:
    """This computes diagonal of the rate matrix Q.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        p_in (jnp.array): Vector size 2 ** (number of 1s in state)

    Returns:
        jnp.array: diag(Q)
    """
    y = p_in * 0.0
    diag = y + 1
    def body_fun(i, val):

        val += kron_sync_diag(log_theta=log_theta, i=i,
                              state=state, diag=diag)
        val += kron_prim_diag(log_theta=log_theta, i=i,
                              state=state, diag=diag)
        val += kron_met_diag(log_theta=log_theta, i=i,
                             state=state, diag=diag)

        return val

    n = log_theta.shape[0] - 1

    y = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=y
    )

    y += kron_seed_diag(log_theta=log_theta,
                        state=state, diag=diag)

    return y


@jit
def marg0not1(p: jnp.array) -> jnp.array:
    p = p.reshape((-1,2), order="C")
    p = p @ jnp.array([[1, 1], [0, 0]])
    return p.ravel(order="F")


@jit
def marg_met_1and1(p: jnp.array) -> jnp.array:
    p = p.reshape((-1,4), order="C")
    p = p @ jnp.array([[1, 0, 1, 0], [0,1,0, 1], [0,0,0,0], [0,0,0, 0]])
    return p.ravel(order="F")


@jit
def marg_prim_1and1(p: jnp.array) -> jnp.array:
    p = p.reshape((-1,4), order="C")
    p = p @ jnp.array([[1, 1, 0, 0], [0,0,1, 1], [0,0,0,0], [0,0,0, 0]])
    return p.ravel(order="F")


@jit
def shuffle_stride2(p: jnp.array) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    return p.ravel(order="F")


@partial(jit, static_argnames=["marg_met", "marg_seeding"])
def marg_transp(p_in: jnp.array, state: jnp.array, marg_met: bool=True, marg_seeding: bool=False) -> jnp.array:
    """
    Calculates (p_in)^T M, where M is a 2^n x 2^(2*n) marginalization matrix implicitely  
    Args:
        p_in (jnp.array): probability distribution to marginalise
        n (int): total number of mutations
        state (jnp.array): bitsring, tumor sample of a single patient
        marg_met (bool): if true: marginalise over mets, else: marginalise over prims
        marg_seeding (bool): if true: marginalise over the seeding event as well 
    Returns:
        p: marginal distribution
    """
    def loop_body(i, p):
        ind = state.at[2*i].get() + 2*state.at[2*i+1].get() + (1 - marg_met)*4
        p = lax.switch(
            index = ind,
            branches = [
                lambda p: p,                        # 00 marg_met=1
                lambda p: shuffle_stride2(p),       # 10 marg_met=1
                lambda p: marg0not1(p),             # 01 marg_met=1
                lambda p: marg_met_1and1(p),        # 11 marg_met=1
                lambda p: p,                        # 00 marg_met=0
                lambda p: marg0not1(p),             # 10 marg_met=0
                lambda p: shuffle_stride2(p),       # 01 marg_met=0
                lambda p: marg_prim_1and1(p),       # 11 marg_met=0         
            ],
            operand = p
        )
        return p

    n = int((state.shape[0] - 1)/2)

    # The shape of the carry_over argument in a for_i loop has to remain constant
    p = lax.fori_loop(0, n, loop_body, p_in)
    p = lax.cond(
        (marg_seeding and state[-1] == 1),
        lambda x: marg0not1(x),
        lambda x: x.reshape((-1, 2), order="C").ravel(order="F"),
        operand = p
    )
    # JAX makes us jump through a lot of hoops here in order to jit this function
    #out_inds = marg_indices(jnp.ones_like(p), state, n, size_marg)
    return p


def keep_col2(p: jnp.array) -> jnp.array:
    p = p.reshape((-1, 2), order="C")
    p = p.at[:,0].set(0.)
    return p.ravel(order="F")


def keep_col1_3(p: jnp.array) -> jnp.array:
    p = p.reshape((-1, 4), order="C")
    p = p.at[:, (0, 2)].set(0.)
    return p.ravel(order="F")


def keep_col2_3(p: jnp.array) -> jnp.array:
    p = p.reshape((-1,4), order="C")
    p = p.at[:, (0,1)].set(0.)
    return p.ravel(order="F")


@partial(jit, static_argnames=["obs_prim"])
def obs_inds(p_in: jnp.array, state: jnp.array, latent_dist: jnp.array, obs_prim: bool=True) -> jnp.array:
    """
    Returns P(Prim = prim_obs, Met) or P(Prim, Met = met_obs), the joint distribution evaluated at either
    the observed metastasis state or the observed primary tumor state
    Args:
        p_in (jnp.array): Joint probability distribution of prims and mets
        state (jnp.array): bitstring, mutational state of prim and met of a patient
        n (int): total number of genomic events
    
        obs_prim (bool): If true return P(Prim = prim_obs, Met) else return P(Prim, Met = met_obs)
    Returns:
        jnp.array
    """
    def loop_body(i, p):
        ind = state.at[2*i].get() + 2*state.at[2*i+1].get() + (1 - obs_prim)*4
        p = lax.switch(
            index = ind,
            branches = [
                lambda p: p,                        # 00 obs_prim=1
                lambda p: keep_col2(p),             # 10 obs_primt=1
                lambda p: shuffle_stride2(p),       # 01 obs_prim=1
                lambda p: keep_col1_3(p),           # 11 obs_prim=1
                lambda p: p,                        # 00 obs_prim=0
                lambda p: shuffle_stride2(p),       # 10 obs_prim=0
                lambda p: keep_col2(p),             # 01 obs_prim=0
                lambda p: keep_col2_3(p),           # 11 obs_prim=0         
            ],
            operand = p
        )
        return p
    
    n = int((state.shape[0] - 1)/2)
    p = lax.fori_loop(0, n, loop_body, jnp.ones_like(p_in))
    latent_size = latent_dist.shape[0]
    # Jax makes us jump through a lot of hoops here, in order to jit this function
    inds = jnp.where(shuffle_stride2(p) == 1, size = latent_size)
    return inds[0] #Output of where is a tuple

#def keep_col1(p: jnp.array) -> jnp.array:
#    p = p.reshape((-1, 2), order="C")
#    p = p.at[:,1].set(0.)
#    return p.ravel(order="F")

#def keep_col0(p: jnp.array) -> jnp.array:
#    p = p.reshape((-1,4), order="C")
#    p = p.at[:, 1:4].set(0.)
#    return p.ravel(order="F")

#def keep_col0_1(p: jnp.array) -> jnp.array:
#    p = p.reshape((-1,4), order="C")
#    p = p.at[:, (2,3)].set(0.)
#    return p.ravel(order="F")

#def keep_col0_2(p: jnp.array) -> jnp.array:
#    p = p.reshape((-1,4), order="C")
#    p = p.at[:, (1,3)].set(0.)
#    return p.ravel(order="F")

#def copy_col0(p: jnp.array) -> jnp.array:
#    p = p.reshape((-1, 2), order="C")
#    p = p.at[:,1].set(p.at[:,0].get())
#    return p.ravel(order="F")


#@partial(jit, static_argnames=["obs_prim"])
#def marg_indices(p_in: jnp.array, state: jnp.array, size_marg: int, obs_prim: bool=True) -> jnp.array:
#    """
#    Returns indices of the states where the marginal probability is stored
#    Args:
#        p_in (jnp.array): Joint probability distribution of prims and mets
#        state (jnp.array): bitstring, mutational state of prim and met of a patient
#        n (int): total number of genomic events
#        size_marg (int): number of latent states
#        obs_prim (bool): If true return P(Prim = prim_obs, Met) else return P(Prim, Met = met_obs)
#    Returns:
#        jnp.array
#    """
#    def loop_body(i, p):
#        ind = state.at[2*i].get()*obs_prim + (1-obs_prim)*(state.at[2*i+1].get())+2
#        p = lax.switch(
#            index = ind,
#            branches = [
#                lambda p: keep_col0(p),
#                lambda p: keep_col0_1(p),           # 11 obs_prim=1
#                lambda p: keep_col0(p),                   # 00 obs_prim=0
#                lambda p: keep_col0_2(p),       # 10 obs_prim=0       
#            ],
#            operand = p
#        )
#        return p
#
#    n = state.shape[0] - 1
#
#    p = lax.fori_loop(0, n, loop_body, p_in)
#
#    return jnp.where(copy_col0(p) == 1, size = size_marg)