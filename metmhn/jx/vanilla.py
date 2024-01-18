from metmhn.jx.kronvec import (k2d1t, 
                               k2ntt, 
                               k2dt0,
                               k2d10,
                               k2d0t
                               )
import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial


def k1d1(p: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
    return p


def k1dt(p: jnp.ndarray, theta: jnp.ndarray, 
         diag: bool = True, transpose: bool = True) -> jnp.ndarray:
    return -theta*p


def _kronvec(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:

    def loop_body_diag(j, val):
        val = lax.cond(
            state[j] == 0,
            k1d1, k2d1t,
            val, theta_i[j])
        return val
    
    n = log_theta.shape[0]
    theta_i = jnp.exp(log_theta[i,:])
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=i,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.cond(
        state[i] == 0,
        k1dt, k2ntt,
        p, theta_i[i], diag, transpose)

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=n,
                      body_fun=loop_body_diag, init_val=p)

    return p


def zero_fun(log_theta: jnp.ndarray, p: jnp.ndarray, i: int, state: jnp.ndarray, 
             diag: bool, transpose: bool) -> jnp.ndarray:
    return 0.*p


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec_i(
    log_theta: jnp.ndarray,
    p: jnp.ndarray,
    i: int,
    state: jnp.ndarray,
    diag: bool = True,
    transpose: bool = False
    ) -> jnp.ndarray:
    return lax.cond(
        not diag and state[i] != 1,
        zero_fun,
        _kronvec,
        log_theta, p, i, state, diag, transpose
        )


@partial(jit, static_argnames=["diag", "transpose"])
def kronvec(log_theta: jnp.ndarray, p: jnp.ndarray, state: jnp.ndarray, 
            diag: bool = True, transpose: bool = False) -> jnp.ndarray:
    """This computes the restricted version of the product of the rate matrix Q with a vector Q p.

    Args:
        log_theta (jnp.ndarray): Log values of the theta matrix
        p (jnp.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q before multiplying. Defaults to False.

    Returns:
        jnp.array: Q p or p^T Q
    """

    #def body_fun(i, val):
    #
    #    val += kronvec_i(log_theta=log_theta, p=p, i=i,
    #                     state=state, diag=diag, transpose=transpose)
    #
    #    return val
    
    n = log_theta.shape[0]
    return jnp.sum(
        vmap(kronvec_i, (None, None, 0, None, None, None), 0)(log_theta, p, jnp.arange(n, dtype=int), state, diag, transpose), 
        0)
    #return lax.fori_loop(
    #    lower=0,
    #    upper=n,
    #    body_fun=body_fun,
    #    init_val=jnp.zeros_like(p)
    #)


def _scal_p_d(x: tuple[jnp.ndarray, jnp.ndarray], d_p: jnp.ndarray, 
              d_m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (k2d1t(x[0], d_p), k2d1t(x[1], d_m))


def _scal_p_1(x: tuple[jnp.ndarray, jnp.ndarray], d_p: jnp.ndarray, 
              d_m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    return x


@jit
def scal_d_pt(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, state: jnp.ndarray, 
              vec: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    def loop_body_diag(j, val):
        val = lax.cond(state[j] == 1,
                       _scal_p_d,
                       _scal_p_1,
                       val, d_p[j], d_m[j]
                       )
        return val
    
    n = log_d_m.shape[0]
    d_p = jnp.exp(log_d_p)
    d_m = jnp.exp(log_d_m)
    p = lax.fori_loop(lower=0, upper=n-1, 
                      body_fun=loop_body_diag, init_val=(vec, vec))
    return k2d10(p[0]) , k2d0t(p[1], d_m[-1])


def _d_scal_d_pt(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, state: jnp.ndarray, 
              vec: jnp.ndarray, i: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    def loop_body_diag(j, val):
        val = lax.cond(state[j] == 1,
                       _scal_p_d,
                       _scal_p_1,
                       val, d_p[j], d_m[j]
                       )
        return val
    
    n = log_d_m.shape[0] - 1
    d_p = jnp.exp(log_d_p)
    d_m = jnp.exp(log_d_m)
    dp_dd = lax.fori_loop(lower=0, upper=i, 
                      body_fun=loop_body_diag, init_val=(vec, vec))
    
    dp_dd = (k2d0t(dp_dd[0], d_p[i]), k2d0t(dp_dd[1], d_m[i]))
    
    dp_dd = lax.fori_loop(lower=i+1, upper=n, 
                      body_fun=loop_body_diag, init_val=dp_dd)
    
    return k2d10(dp_dd[0]),  k2d0t(dp_dd[1], d_m[-1])


def zero_tuple_fun(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, state: jnp.ndarray, 
              vec: jnp.ndarray, i:int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (0. * vec, 0. * vec)


def tuple_0_ddm(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, state: jnp.ndarray, 
              vec: jnp.ndarray, i:int) -> tuple[jnp.ndarray, jnp.ndarray]:
    return (0*vec, scal_d_pt(log_d_p, log_d_m, state, vec)[1])


@jit 
def d_scal_d_pt(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, state: jnp.ndarray, 
              vec: jnp.ndarray, i:int) -> tuple[jnp.ndarray, jnp.ndarray]:
    n = log_d_p.shape[0]-1
    return lax.switch(state[i] + (i==n),
                      [zero_tuple_fun, _d_scal_d_pt, tuple_0_ddm],
                      log_d_p, log_d_m, state, vec, i
                    )


@jit
def x_partial_D_y(log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, state: jnp.ndarray, 
              x: jnp.ndarray, y:jnp.array) -> tuple[jnp.ndarray, jnp.ndarray]:
    
    def body_fun(i, log_d_p, log_d_m, state, x, y):
        d_dp, d_dm = d_scal_d_pt(log_d_p, log_d_m, state, y, i)
        a = jnp.dot(x, d_dp)
        b = jnp.dot(x, d_dm)
        return jnp.array([a,b])
    
    n = log_d_p.shape[0]
    res = vmap(body_fun, (0, None, None, None, None, None), 0)(jnp.arange(n), log_d_p, log_d_m, state, x, y)
    #d_p, d_m = lax.fori_loop(0, n, body_fun,(jnp.zeros_like(log_d_p), jnp.zeros_like(log_d_m)))
    return res[:,0], res[:,1]


def kron_diag_i(
        log_theta: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        diag: jnp.ndarray) -> jnp.ndarray:
    
    def loop_body(j, val):
        val = lax.cond(
            state[j] == 0,
            k1d1, 
            k2d1t,
            val, theta_i[j]
        )
        return val
    
    n = log_theta.shape[0]
    theta_i = jnp.exp(log_theta[i, :])

    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag
    )

    diag = lax.cond(
        state[i] == 0,
        k1dt,
        k2dt0,
        diag, theta_i[i]
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
        log_theta: jnp.ndarray,
        state: jnp.ndarray,
        diag: jnp.ndarray
        ) -> jnp.ndarray:

    #def body_fun(i, val):
    #    val += kron_diag_i(log_theta=log_theta, i=i, state=state, diag=diag)
    #    return val

    n = log_theta.shape[0]
    return jnp.sum(vmap(kron_diag_i, (None, 0, None, None), 0)(log_theta, jnp.arange(n), state, diag), axis=0)
    #return lax.fori_loop(
    #    lower=0,
    #    upper=n,
    #    body_fun=body_fun,
    #    init_val=jnp.zeros_like(diag)
    #)


@partial(jit, static_argnames=["transpose"])
def R_inv_vec(log_theta: jnp.ndarray, 
              x: jnp.ndarray, 
              state: jnp.ndarray,
              d_rates: jnp.ndarray = 1,
              transpose: bool = False,
              ) -> jnp.ndarray:
    """This computes R^{-1} x = (I - Q D^{-1})^{-1} x

    Args:
        log_theta (np.ndarray): Log values of the theta matrix
        x (np.ndarray): Vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        state (np.ndarray): Binary state vector, representing the current sample's events.
        transpose (bool): Logical flag, if true calculate x^T (I - Q D^{-1})^{-1}

    Returns:
        np.ndarray: R_i^{-1} x or x^T R_i^{-1}
    """

    def body_fun(j, val):
        return lidg * (kronvec(log_theta, val, state, False, transpose) + x)
    
    state_size = jnp.log2(x.shape[0]).astype(int)

    lidg = -1 / (kron_diag(log_theta=log_theta,
                 state=state, diag=jnp.ones_like(x))-d_rates)
    y = lidg * x

    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=body_fun,
        init_val=y
    )

    return y


def t_x_0(z: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    return (z, 0.)


def t(z: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    z = z.reshape((-1, 2), order="C")
    val = z.sum()
    return z.flatten(order="F"), val


def f(z: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    z = z.reshape((-1, 2), order="C")
    val = z[:, 1].sum()
    return z.flatten(order="F"), val


def t_x_sum(z: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    return (z, z.sum())


@jit
def x_partial_Q_y(
        log_theta: jnp.ndarray,
        x: jnp.ndarray,
        y: jnp.ndarray,
        state: jnp.ndarray, 
        ) -> jnp.ndarray:
    """This function computes x \partial Q y with \partial Q the Jacobian of Q w.r.t. all thetas
    efficiently using the shuffle trick (sic!).

    Args:
        log_theta (np.ndarray): Logarithmic theta values of the MHN
        x (np.ndarray): x vector to multiply with from the left. Length must equal the number of
            nonzero entries in the state vector.
        y (np.ndarray): y vector to multiply with from the right. Length must equal the number of
            nonzero entries in the state vector.
        state (np.ndarray): Binary state vector, representing the current sample's events.

    Returns:
        np.ndarray: x \partial_(\Theta_{ij}) Q y for i, j = 1, ..., n+1
    """
    n = log_theta.shape[0]
    val = jnp.zeros(shape=(n, n))

    def body_fun(i, val):
        
        def inner_fun(j, val):
            z, _val = lax.cond(
                state[j] == 0,
                t_x_0, 
                f,
                val[0]
                )
            return z, val[1].at[j].set(_val)

        z = x * kronvec_i(log_theta=log_theta,
                          p=y, i=i, state=state)

        z, val = lax.fori_loop(
            lower=0,
            upper=i,
            body_fun=inner_fun,
            init_val=(z, val)
        )

        z, _val = lax.cond(
            state[i] == 0,
            t_x_sum,
            t,
            z
            )
        
        val = val.at[i].set(_val)

        z, val = lax.fori_loop(
            lower=i+1,
            upper=n,
            body_fun=inner_fun,
            init_val=(z, val)
        )

        return val

    val = vmap(body_fun, (0,0),0) (jnp.arange(n, dtype=int), val)
    d_diag = -jnp.sum(val, axis=0) + jnp.diagonal(val)
    return val, d_diag


@jit
def gradient(log_theta: jnp.ndarray,
             state: jnp.ndarray, 
             p_0: jnp.ndarray
             ) -> jnp.ndarray:
    """This computes the gradient of the score function, which is the log-likelihood of a data vector p_D
    with respect to the log_theta matrix

    Args:
        log_theta (np.ndarray): Theta matrix with logarithmic entries.
        state (np.ndarray): Binary state vector, representing the current sample's events.
        p_0 (np.ndarray): Starting distribution

    Returns:
        jnp.ndarray: \partial_theta (p_D^T log p_theta)
    """
    p_theta = R_inv_vec(log_theta=log_theta, x=p_0, state=state)
    x = jnp.zeros_like(p_theta)
    x = x.at[-1].set(1/p_theta[-1])
    x = R_inv_vec(log_theta=log_theta, x=x,
                  state=state, transpose=True)
    d_th, d_diag = x_partial_Q_y(log_theta=log_theta, x=x, y=p_theta, state=state)
    return d_th, d_diag, p_theta