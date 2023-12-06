from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import jax


def diagnosis_theta(log_theta: jnp.array, log_diag_rates: jnp.array) -> jnp.array:
    """"Generate a log. theta that is used in the diagnosis formalism. scale all offdiagonals by their effect on the diagnosis

    Args:
        log_theta (jnp.array): theta matrix of size n+1 x n+1
        log_diag_rates (jnp.arrays): logarithmic effects of muts on diagnosis of size n+1
    Returns:
        jnp.array: scaled theta matrix of size n+1 x n+1
    """
    diagonal = jnp.diagonal(log_theta)
    scaled_theta = jnp.apply_along_axis(lambda x,y:(x-y), 1, log_theta, log_diag_rates)
    d_i = jnp.arange(0, scaled_theta.shape[0])
    scaled_theta = scaled_theta.at[d_i, d_i].set(diagonal)
    return scaled_theta


# Kronecker factors
def k2dt0(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([-theta, 0.])
    p = jax.vmap(lambda a, x: a * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k2d0t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([0., theta])
    p = jax.vmap(lambda a, x: a * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k2dtt(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    p = jax.vmap(lambda x: -theta * x, 0, 0)(p)
    return p.flatten(order="F")


def k2d1t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([1., theta])
    p = jax.vmap(lambda a, x: a * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k2d10(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([1., 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k2d01(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([0., 1.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k2d11(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    return p.flatten(order="F")


def k2ntt(p: jnp.ndarray, theta: float, 
          diag: bool = True, transpose: bool = False
          ) -> jnp.ndarray:
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


def k4ns(p: jnp.ndarray, theta: float, 
         diag: bool = True, transpose: bool = False
         ) -> jnp.ndarray:
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


def k4np(p: jnp.ndarray, theta: float, 
         diag: bool = True, transpose: bool = False
         ) -> jnp.ndarray:
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


def k4nm(p: jnp.ndarray, theta: float, 
         diag: bool = True, transpose: bool = False
         ) -> jnp.ndarray:
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


def k4d100t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., 0., 0., theta])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k4dt000(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, 0., 0., 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k4dtt00(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, -theta, 0., 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k4dt0t0(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, 0., -theta, 0.])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k4d1t1t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., theta, 1., theta])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def k4d11tt(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., 1., theta, theta])
    p = jax.vmap(lambda t, x: t * x, (None, 0), 0)(theta_slice, p)
    return p.flatten(order="F")


def _kronvec_sync(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:

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
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    """This computes the restricted version of the product of the synchronized part of the ith Q summand
    Q_i with a vector Q_i p.

    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        p (jnp.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        i (int): Index of the summand.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.ndarray: Q_i p
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


def _kronvec_prim(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:

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

    n = log_theta.shape[0]-1
    
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
    p = lax.fori_loop(lower=i+1, upper=n,
                      body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = k2d01(p)
    return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_prim(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    """This computes the restricted version of the product of the asynchronous primary tumour
    part of the ith Q summand Q_i with a vector Q_i p.

    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        p (jnp.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.ndarray: Q_i p
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


def _kronvec_met(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    def loop_body_diag(j, val):

        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                # ...|00|...
                lambda x: x,
                #...|10|...
                lambda x: k2d11(x), 
                #...|01|...
                lambda x: k2d1t(x, theta=jnp.exp(log_theta[i, j])),
                #...|11|...
                lambda x: k4d11tt(p=x, theta=jnp.exp(log_theta[i,j]))
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
            lambda x: k2dtt(p=x, theta=jnp.exp(log_theta[i,i])),
            lambda x: k2ntt(p=x, theta=jnp.exp(log_theta[i, i]), 
                            diag=diag, transpose=transpose),
            lambda x: k4nm(p=x, 
                           theta=jnp.exp(log_theta[i,i]), 
                           diag=diag, transpose=transpose)
        ],
        operand=p
    )
    n = log_theta.shape[0]-1
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=n,
                      body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = k2d0t(p, theta=jnp.exp(log_theta[i,n]))

    return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_met(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    """This computes the restricted version of the product of the asynchronous metastasis
    part of the ith Q summand Q_i with a vector Q_i p.

    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        p (jnp.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.ndarray: Q_i p
    """
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    return lax.cond(
        not diag and state.at[2 * i + 1].get() == 0,
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


def _kronvec_seed(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    
    def loop_body_diag(j, val):

        val = lax.switch(
            index=state[2*j] + 2 * state[2*j+1],
            branches=[
                lambda x: x,
                lambda x: k2d10(x),
                lambda x: k2d10(x),
                lambda x: k4d100t(p=x, theta=jnp.exp(
                    log_theta[log_theta.shape[0], j]))
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
            log_theta[n, n]), diag=diag, transpose=transpose),
        lambda x: x * -jnp.exp(log_theta[n, n]),
        operand=p
    )

    return p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_seed(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    """This computes the restricted version of the product of the seeding summand of Q with a vector Q_M p.
    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        p (jnp.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_M (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_M before multiplying. Defaults to False.

    Returns:
        jnp.ndarray: Q_seed p
    """
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    return lax.cond(
        not diag and state.at[-1].get() == 0,
        lambda: p*0.0,
        lambda: _kronvec_seed(
            log_theta=log_theta,
            p=p,
            state=state,
            diag=diag,
            transpose=transpose
        ),
    )


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec(log_theta: jnp.ndarray, p: jnp.ndarray, state: jnp.ndarray,
            diag: bool = True, transpose: bool = False
            ) -> jnp.ndarray:
    """
    This computes the restricted version of the product of the rate matrix Q with a vector p.

    Args:
        log_theta (jnp.ndarray): theta matrix with log. entries
        p (jnp.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
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
    y = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=jnp.zeros_like(p)
    )

    y += kronvec_seed(log_theta=log_theta, p=p,
                      state=state, diag=diag, transpose=transpose)

    return y


def diag_scal_p(log_d_p: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Multiplies a vector p with the diagonal matrix of PT-diagnosis effects d_p

    Args:
        d_p (jnp.ndarray): Vector of PT-diagnosis effects
        state (jnp.ndarray): Observed PT-MT tumor state
        p (jnp.ndarray): Vector to multiply with d_p

    Returns:
        jnp.ndarray: d_p*P
    """
    def sync_loop(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda x: x,
                lambda x: k2d1t(x,
                                theta=jnp.exp(log_d_p.at[j].get())),
                lambda x: k2d11(x),
                lambda x: k4d1t1t(p=x,
                                  theta=jnp.exp(log_d_p.at[j].get()))
            ],
            operand=val)
        return val

    n = log_d_p.shape[0]-1
    # Diagonal Kronecker factors
    pt_d_rates = lax.fori_loop(lower=0, upper=n,
                      body_fun=sync_loop, init_val=p)

    pt_d_rates = pt_d_rates.reshape((-1, 2), order="C").flatten(order="F")
    return pt_d_rates


def diag_scal_m(log_d_m: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    """Multiplies a vector p with the diagonal matrix of MT-diagnosis effects d_m

    Args:
        d_m (jnp.ndarray): Vector of MT-diagnosis effects
        state (jnp.ndarray): Observed PT-MT tumor state
        p (jnp.ndarray): Vector to multiply with d_m

    Returns:
        jnp.ndarray: d_m*p
    """
    def met_loop(j, val):
        val = lax.switch(
            index=state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                # ...|00|...
                lambda x: x,
                #...|10|...
                lambda x: k2d11(x),
                #...|01|...
                lambda x: k2d1t(x, 
                                theta=jnp.exp(log_d_m.at[j].get())),
                #...|11|...
                lambda x: k4d11tt(p=x,
                                  theta=jnp.exp(log_d_m.at[j].get()))
            ],
            operand=val
        )
        return val
    n = log_d_m.shape[0] - 1
    mt_d_rates = lax.fori_loop(lower=0, upper=n,
                           body_fun=met_loop, init_val=p)
    mt_d_rates = k2d0t(mt_d_rates, jnp.exp(log_d_m.at[-1].get()))
    return mt_d_rates


def kron_sync_diag(
        log_theta: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        n_state: int
        ) -> jnp.ndarray:
    """This computes the diagonal of the synchronized part of the ith Q summand Q_i.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        i (int): Index of the summand.
        state (jnp.array): Binary state vector, representing the current sample's events.
        n_state (int): Number of non zero bits in state

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
                lambda val: k4d100t(val, jnp.exp(log_theta.at[i, j].get()))
            ],
            operand=val
        )
        return val

    n = log_theta.shape[0] - 1
    diag = jnp.ones(2**n_state)

    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag)

    diag = lax.switch(
        index=state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        branches=[
            lambda val: -jnp.exp(log_theta.at[i, i].get()) * val,
            lambda val: k2dt0(val, jnp.exp(log_theta.at[i, i].get())),
            lambda val: k2dt0(val, jnp.exp(log_theta.at[i, i].get())),
            lambda val: k4dt000(val, jnp.exp(log_theta.at[i, i].get()))
        ],
        operand=diag)

    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag)

    diag = lax.cond(
        state[-1] == 1,
        lambda x: k2d10(x),
        lambda x: x,
        operand=diag)
    
    return diag


def _kron_prim_diag(
        log_theta: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        n_state : int
        ) -> jnp.ndarray:

    # Single diagonal Kronecker vector product
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
    diag = jnp.ones(2**n_state)
    
    # Diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag)

    # Non diagonal Kronecker factors
    diag = lax.switch(
        index=state[2*i] + 2 * state[2*i+1],
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dt0(val, jnp.exp(log_theta[i, i])),
            lambda val: k2dtt(val, jnp.exp(log_theta[i, i])),
            lambda val: k4dt0t0(val, jnp.exp(log_theta[i, i]))
        ],
        operand=diag)
    
    # Diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag)
    
    # Last Kronecker factor
    diag = k2d01(diag)
    return diag


def kron_prim_diag(
        log_theta: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        n_state: int
        ) -> jnp.ndarray:
    """This computes the diagonal of the asynchronous primary tumour part of the ith
    Q summand Q_i.

    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        i (int): Index of the summand.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        n_state(int): Number of non zero bits in state
    
    Returns:
        jnp.ndarray: diag(Q_i_prim)
    """

    return lax.cond(
        state[-1] == 0,
        lambda: jnp.zeros(2**n_state),
        lambda: _kron_prim_diag(
            log_theta=log_theta,
            i=i,
            state=state,
            n_state = n_state,
        ),
    )


def _kron_met_diag(
        log_theta: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        n_state: int
        ) -> jnp.ndarray:

    # Single Kronecker vector product
    def loop_body(j, val):
        val = lax.switch(
            index=state[2*j] + 2 * state[2*j+1],
            branches=[
                # ...|00|...
                lambda val: val,
                #...|10|...
                lambda val: k2d11(val),
                #...|01|...
                lambda val: k2d1t(val, jnp.exp(log_theta[i, j])),
                 #...|11|...
                lambda val: k4d11tt(val, theta=jnp.exp(log_theta[i,j]))
            ],
            operand=val
        )
        return val
    
    n = log_theta.shape[0] - 1
    diag =jnp.ones(2**n_state)

    # Diagonal Kronecker vector products
    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag)

    # Non diagonal Kronecker vector products
    diag = lax.switch(
        index=state[2*i] + 2 * state[2*i+1],
        branches=[
            lambda val: -jnp.exp(log_theta[i, i]) * val,
            lambda val: k2dtt(val, theta = jnp.exp(log_theta[i,i])),
            lambda val: k2dt0(val, jnp.exp(log_theta[i,i])),
            lambda val: k4dtt00(val, theta = jnp.exp(log_theta[i,i]))
        ],
        operand=diag
    )

    # diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    # Last Kronecker factor
    diag = k2d0t(diag, jnp.exp(log_theta[i,n]))

    return diag


def kron_met_diag(
        log_theta: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        n_state: int
        ) -> jnp.ndarray:
    """This computes the diagonal of the asynchronous metastasis part of the ith
        Q summand Q_i.

    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        i (int): Index of the summand.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        n_state: Number of nonzero bits in state

    Returns:
        jnp.ndarray: diag(Q_i_met)
    """
    return lax.cond(
        state[-1] == 0,
        lambda: jnp.zeros(2**n_state),
        lambda: _kron_met_diag(
            log_theta=log_theta,
            i=i,
            state=state,
            n_state = n_state,
        ),
    )


def kron_seed_diag(
        log_theta: jnp.ndarray,
        state: jnp.ndarray,
        n_state: int
        ) -> jnp.ndarray:
    """This computes the diagonal of the seeding summand of Q.

    Args:
        log_theta (jnp.array): Log values of the theta matrix
        n (int): Total number of events in the MHN.
        state (jnp.array): Binary state vector, representing the current sample's events.
        n_state (int): Number of non zero bits in state

    Returns:
        jnp.ndarray: diag(Q_seed)
    """
    n = log_theta.shape[0] - 1
    diag = jnp.ones(2**n_state)
    
    # Single Kronecker vector product
    def loop_body(j, val):
        val = lax.switch(
            index=state[2*j] + 2 * state[2*j+1],
            branches=[
                lambda val: val,
                lambda val: k2d10(val),
                lambda val: k2d10(val),
                lambda val: k4d100t(val, jnp.exp(log_theta[n, j]))
            ],
            operand=val
        )
        return val

    # Diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    # Last Kronecker factor
    diag = lax.cond(
        state[-1] == 1,
        lambda x: k2dt0(x, jnp.exp(log_theta[-1, -1])),
        lambda x: x * -jnp.exp(log_theta[-1, -1]),
        operand=diag
    )

    return diag

@partial(jit, static_argnames=["n_state"])
def kron_diag(log_theta: jnp.ndarray, state: jnp.ndarray, n_state: int) -> jnp.ndarray:
    """This computes diagonal of the rate matrix Q.

    Args:
        log_theta (jnp.ndarray): Theta matrix with log entries
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        n_state (int): Number of non zero bits in state

    Returns:
        jnp.ndarray: diag(Q)
    """

    def body_fun(i, val):

        val += kron_sync_diag(log_theta=log_theta, i=i,
                              state=state, n_state=n_state)
        val += kron_prim_diag(log_theta=log_theta, i=i,
                              state=state, n_state=n_state)
        val += kron_met_diag(log_theta=log_theta, i=i,
                             state=state, n_state=n_state)

        return val

    n = log_theta.shape[0] - 1
    y = jnp.zeros(2**n_state)
    y = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=y
    )

    y += kron_seed_diag(log_theta=log_theta, state=state, n_state=n_state)

    return y


def shuffle_stride2(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    return p.ravel(order="F")


def keep_col2(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    p = p.at[:, 0].set(0.)
    return p.ravel(order="F")


def keep_col1_3(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    p = p.at[:, (0, 2)].set(0.)
    return p.ravel(order="F")


def keep_col2_3(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    p = p.at[:, (0, 1)].set(0.)
    return p.ravel(order="F")

def keep_all(p: jnp.ndarray) -> jnp.ndarray:
    return p

@partial(jit, static_argnames=["n_joint", "pt_first"])
def obs_states(n_joint: int, state: jnp.ndarray, pt_first: bool = True) -> jnp.ndarray:
    """Selects all states that are compatible with state at first sampling. If obs_prim = true then primary part of state is
    assumed to be observed and the metastasis part is latent. Returns a binary array with entry at index i set to 1 if the 
    corresponfing state is compatible. 
    
    Args:
        n_joint (int): Number of non zero bits in state
        state (jnp.ndarray): Bitstring, mutational state of PT and MT states of a patient
        pt_first (bool): If true return the distribution of potential MT-states for a fixed PT observation: P(Prim = prim_obs, Met) 
            else return the distribution of potential PT-states for a fixed MT observation: P(Prim, Met = met_obs)
    
    Returns:
        jnp.ndarray
    """
    def loop_body(i, p):
        ind = state.at[2*i].get() + 2*state.at[2*i+1].get() + (1 - pt_first)*4
        p = lax.switch(
            index=ind,
            branches=[
                keep_all,              # 00 obs_prim=1
                keep_col2,             # 10 obs_prim=1
                shuffle_stride2,       # 01 obs_prim=1
                keep_col1_3,           # 11 obs_prim=1
                keep_all,              # 00 obs_prim=0
                shuffle_stride2,       # 10 obs_prim=0
                keep_col2,             # 01 obs_prim=0
                keep_col2_3,           # 11 obs_prim=0
            ],
            operand=p
        )
        return p

    n = (state.shape[0] - 1)//2
    p = lax.fori_loop(0, n, loop_body, jnp.ones(2**n_joint))
    p = lax.cond(state.at[-1].get() == 0,
                keep_all,
                keep_col2,
                operand = p)
    return p