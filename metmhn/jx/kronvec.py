from functools import partial
from jax import jit, lax
import jax.numpy as jnp
import jax


def diagnosis_theta(log_theta: jnp.ndarray, log_diag_rates: jnp.ndarray) -> jnp.ndarray:
    """"Generate a log. theta that is used in the diagnosis formalism. 
    Columnwise scale all off diagonals entries by the mutations effect on the diagnosis.

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_diag_rates (jnp.ndarray): Log. effects of muts on diagnosis
    Returns:
        jnp.array: scaled theta matrix
    """
    diagonal = jnp.diagonal(log_theta)
    scaled_theta = jnp.apply_along_axis(lambda x,y:(x-y), 1, log_theta, log_diag_rates)
    d_i = jnp.arange(0, scaled_theta.shape[0])
    scaled_theta = scaled_theta.at[d_i, d_i].set(diagonal)
    return scaled_theta


def mult_t_x(t, x):
    return t*x

mult_t_x_vec = jax.vmap(mult_t_x, (None, 0), 0)

# Kronecker factors

def k1d00(p: jnp.ndarray, theta:float=1.) -> jnp.ndarray:
    return p


def k2dt0(p: jnp.ndarray, theta: float, 
          diag: bool = True, transpose: bool = True) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([-theta, 0.])
    p = mult_t_x_vec(theta_slice, p)
    return p.flatten(order="F")


def k2d0t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([0., theta])
    p = mult_t_x_vec(theta_slice, p)
    return p.flatten(order="F")


def k2dtt(p: jnp.ndarray, theta: float, diag: bool = True, 
          transpose: bool = True) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    return -theta * p.flatten(order="F")


def k2d1t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([1., theta])
    p = mult_t_x_vec(theta_slice, p)
    return p.flatten(order="F")


def k2d10(p: jnp.ndarray, theta: float=1.) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([1., 0.])
    p = mult_t_x_vec(theta_slice, p)
    return p.flatten(order="F")


def k2d01(p: jnp.ndarray) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    theta_slice = jnp.array([0., 1.])
    p = mult_t_x_vec(theta_slice, p)
    return p.flatten(order="F")


def k2d11(p: jnp.ndarray, theta: float=1.) -> jnp.ndarray:
    p = p.reshape((-1, 2), order="C")
    return p.flatten(order="F")


def k2ntt(p: jnp.ndarray, theta: float, 
          diag: bool = True, transpose: bool = False
          ) -> jnp.ndarray:
    t = lax.select_n(2*diag+transpose, 
                     jnp.array([[0., theta], [0., 0.]]),
                     jnp.array([[0., 0.], [theta, 0.]]),
                     jnp.array([[-theta, theta], [0., 0.]]),
                     jnp.array([[-theta, 0.], [theta, 0.]]),
                     )
    p = p.reshape((-1, 2), order="C")
    p = p @ t
    return p.flatten(order="F")


def k4ns(p: jnp.ndarray, theta: float, 
         diag: bool = True, transpose: bool = False
         ) -> jnp.ndarray:
    t = lax.select_n(2*diag+transpose,
                     jnp.array([[0., 0., 0., theta],[0., 0., 0., 0.], 
                                [0., 0., 0., 0.], [0., 0., 0., 0.]]),
                     jnp.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                [0., 0., 0., 0.], [theta, 0., 0., 0.]]),
                     jnp.array([[-theta, 0., 0., theta], [0., 0., 0., 0.],
                                [0., 0., 0., 0.], [0., 0., 0., 0.]]),
                     jnp.array([[-theta, 0., 0., 0.], [0., 0., 0., 0.],
                                [0., 0., 0., 0.], [theta, 0., 0., 0.]])
                    )
    p = p.reshape((-1, 4), order="C")
    p = p @ t
    return p.flatten(order="F")


def k4np(p: jnp.ndarray, theta: float, 
         diag: bool = True, transpose: bool = False
         ) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    t = lax.select_n(2*diag+transpose,
                     jnp.array([[0., theta, 0., 0.], [0., 0., 0., 0.], 
                                [0., 0., 0., theta], [0., 0., 0., 0.]]),
                     jnp.array([[0., 0., 0., 0.], [theta,0., 0., 0.],
                                [0., 0., 0., 0.], [0., 0., theta, 0.]]),
                     jnp.array([[-theta, theta, 0., 0.], [0., 0., 0., 0.],
                                [0., 0., -theta, theta], [0., 0., 0., 0.]]),
                     jnp.array([[-theta, 0., 0., 0.], [theta, 0., 0., 0.], 
                                [0., 0., -theta, 0.], [0., 0., theta, 0.]])
        )
    p = p @ t
    return p.flatten(order="F")


def k4nm(p: jnp.ndarray, theta: float, 
         diag: bool = True, transpose: bool = False
         ) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    t = lax.select_n(2*diag+transpose,
                     jnp.array([[0., 0., theta, 0.], [0., 0., 0., theta], 
                                [0., 0., 0., 0.], [0., 0., 0., 0.]]),
                     jnp.array([[0., 0., 0., 0.], [0., 0., 0., 0.],
                                [theta, 0., 0., 0.], [0., theta, 0., 0.]]),
                     jnp.array([[-theta, 0., theta, 0.], [0., -theta, 0., theta], 
                                [0., 0., 0., 0.], [0., 0., 0., 0.]]),
                     jnp.array([[-theta, 0., 0., 0.], [0., -theta, 0., 0.], 
                                [theta, 0., 0., 0.], [0., theta, 0., 0.]])
        )
    p = p @ t
    return p.flatten(order="F")


def k4d100t(p: jnp.ndarray, theta: float, diag:bool=True, transpose:bool=False) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., 0., 0., theta])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4dt000(p: jnp.ndarray, theta: float, diag: bool = True, 
            transpose: bool = True) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, 0., 0., 0.])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4dtt00(p: jnp.ndarray, theta: float, 
            diag: bool = True, transpose: bool= True) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, -theta, 0., 0.])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4d00tt(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([0, 0, theta, theta])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4dt0t0(p: jnp.ndarray, theta: float, diag: bool = True,
            transpose: bool = True) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([-theta, 0., -theta, 0.])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4d0t0t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([0, theta, 0, theta])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4d1t1t(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., theta, 1., theta])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k4d11tt(p: jnp.ndarray, theta: float) -> jnp.ndarray:
    p = p.reshape((-1, 4), order="C")
    theta_slice = jnp.array([1., 1., theta, theta])
    p = mult_t_x(theta_slice, p)
    return p.flatten(order="F")


def k1ns(p:jnp.ndarray, theta: float,
         diag: bool = True, transpose: bool = True):
    return -theta * p


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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d10, k2d10, k4d100t],
            val, theta_i[j])
        return val

    # Diagonal Kronecker factors
    theta_i = jnp.exp(log_theta[i,:])
    p = lax.fori_loop(lower=0, upper=i,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [k1ns, k2dt0, k2dt0, k4ns],
        p, theta_i[i], diag, transpose)

    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=log_theta.shape[0]-1,
                      body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = lax.cond(
        state[-1] == 1,
        k2d10, k1d00,
        p, 0.
        )

    return p


def zero_fun(log_theta, p, i, state, diag, transp):
        return p*0.0


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
        zero_fun,
        _kronvec_sync,
        log_theta, p, i, state, diag, transpose)


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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d1t, k2d11, k4d1t1t],
            val, theta_i[j])
        return val

    n = log_theta.shape[0]-1
    theta_i = jnp.exp(log_theta[i, :])
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=i,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [k1ns, k2ntt, k2dtt, k4np],
        p, theta_i[i], diag, transpose)

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
    flag = not diag and state[2 * i] == 0
    return lax.cond(
        flag + (state[-1] == 0)*(1-flag),
        zero_fun, 
        _kronvec_prim,
        log_theta, p, i, state, diag,transpose
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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d11, k2d1t, k4d11tt],
            val, theta_i[j]
        )
        return val
    
    # Diagonal Kronecker factors
    theta_i = jnp.exp(log_theta[i, :])
    p = lax.fori_loop(lower=0, upper=i,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [k1ns, k2dtt, k2ntt, k4nm],
        p, theta_i[i], diag, transpose
    )
    n = log_theta.shape[0]-1
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=i+1, upper=n,
                      body_fun=loop_body_diag, init_val=p)

    # Last Kronecker factor
    p = k2d0t(p, theta_i[n])

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
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

    Returns:
        jnp.ndarray: Q_i p
    """
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    flag = not diag and state[2 * i + 1] == 0
    return lax.cond(
        flag + (state[-1] == 0)*(1-flag),
        zero_fun,
        _kronvec_met,
        log_theta, p, i, state, diag, transpose
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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d10, k2d10, k4d100t],
            val, theta_i[j]
        )
        return val
    
    n = log_theta.shape[0]-1
    theta_i = jnp.exp(log_theta[n,:])
    # Diagonal Kronecker factors
    p = lax.fori_loop(lower=0, upper=n,
                      body_fun=loop_body_diag, init_val=p)

    # Non-diagonal Kronecker factor
    p = lax.cond(
        state[-1] == 1,
        k2ntt,
        k1ns,
        p, theta_i[n], diag, transpose
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
        state (jnp.ndarray): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_M (and not set it to 0). Defaults to True.
        transpose (bool, optional): Whether to transpose Q_M before multiplying. Defaults to False.

    Returns:
        jnp.ndarray: Q_seed p
    """
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    def seed_zero(log_theta, p, state, diag, transpose):
        return p * 0.
    
    return lax.cond(
        not diag and state[-1] == 0,
        seed_zero,
        _kronvec_seed,
        log_theta, p, state, diag, transpose
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


@jit
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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d1t, k2d11, k4d1t1t],
            val, d_p[j]
            )
        return val

    n = log_d_p.shape[0]-1
    d_p = jnp.exp(log_d_p)
    # Diagonal Kronecker factors
    pt_d_rates = lax.fori_loop(lower=0, upper=n,
                      body_fun=sync_loop, init_val=p)

    #pt_d_rates = pt_d_rates.reshape((-1, 2), order="C").flatten(order="F")
    pt_d_rates = k2d1t(pt_d_rates, d_p[-1])
    return pt_d_rates


def _partial_diag_scal_p(log_d_p: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray, i:int):
    def sync_loop(j, val):
        val = lax.switch(
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d1t, k2d11, k4d1t1t],
            val, d_p[j]
            )
        return val

    n = log_d_p.shape[0]-1
    d_p = jnp.exp(log_d_p)
    # Diagonal Kronecker factors
    pt_d_rates = lax.fori_loop(lower=0, upper=i, body_fun=sync_loop, init_val=p)
    pt_d_rates = lax.cond(state[2*i] + state[2*i+1]==1,
                          k2d0t,
                          k4d0t0t,
                          pt_d_rates, d_p[i])
    pt_d_rates = lax.fori_loop(lower=i+1, upper=n, body_fun=sync_loop, init_val=pt_d_rates)
    pt_d_rates = k2d1t(pt_d_rates, d_p[-1])
    return pt_d_rates


def scal_zero_fun(log_d_p, state, p, i):
    return p*0.

@jit
def partial_diag_scal_p(log_d_p: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray, 
                        i:int) -> jnp.ndarray:
    n = log_d_p.shape[0] - 1

    def partial_le(log_d_p, state, p, i):
        d_p_d = diag_scal_p(log_d_p, state, p)
        d_p_d = d_p_d.reshape((-1, 2), order="F")
        d_p_d = d_p_d.at[:,0].set(0.)
        return d_p_d.ravel(order="F")

    return lax.switch(state[2*i]+(i==n),
                    [scal_zero_fun, _partial_diag_scal_p, partial_le],
                    log_d_p, state, p, i
                    )
 
@jit
def diag_scal_m(log_d_m: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray, i: int = 0
                ) -> jnp.ndarray:
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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d11, k2d1t, k4d11tt],
            val, d_m[j]
        )
        return val
    n = log_d_m.shape[0] - 1
    d_m = jnp.exp(log_d_m)
    mt_d_rates = lax.fori_loop(lower=0, upper=n,
                           body_fun=met_loop, init_val=p)
    mt_d_rates = k2d0t(mt_d_rates, d_m[-1])
    return mt_d_rates


def _partial_diag_scal_m(log_d_m: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray, 
                         i:int) -> jnp.ndarray:
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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d11, k2d1t, k4d11tt],
            val, d_m[j]
        )
        return val
    n = log_d_m.shape[0] - 1
    d_m = jnp.exp(log_d_m)
    mt_d_rates = lax.fori_loop(lower=0, upper=i, body_fun=met_loop, init_val=p)
    mt_d_rates = lax.cond(state[2*i]+state[2*i+1]==1, 
                          k2d0t, 
                          k4d00tt,
                          mt_d_rates, d_m[i])
    mt_d_rates = lax.fori_loop(lower=i+1, upper=n, body_fun=met_loop, init_val=mt_d_rates)
    return k2d0t(mt_d_rates, d_m[-1])


@jit
def partial_diag_scal_m(log_d_m: jnp.ndarray, state: jnp.ndarray, p: jnp.ndarray, 
                        i:int) -> jnp.ndarray:
    n = log_d_m.shape[0] - 1
    return lax.switch((state[jnp.min(jnp.array([2*n, 2*i+1]))]) + (i==n),
                      [scal_zero_fun, _partial_diag_scal_m, diag_scal_m],
                      log_d_m, state, p, i)


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
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d10, k2d10, k4d100t],
            val, theta_i[j]
        )
        return val
    theta_i = jnp.exp(log_theta[i,:])
    n = log_theta.shape[0] - 1
    diag = jnp.ones(2**n_state)

    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag)

    diag = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [k1ns, k2dt0, k2dt0, k4dt000],
        diag, theta_i[i]
    )
    
    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag)

    diag = lax.cond(
        state[-1] == 1,
        k2d10,
        k1d00,
        diag, 1.)
    
    return diag


def _kron_prim_diag(
        log_theta: jnp.ndarray,
        diag: jnp.array,
        i: int,
        state: jnp.ndarray,
        ) -> jnp.ndarray:

    # Single diagonal Kronecker vector product
    def loop_body(j, val):
        val = lax.switch(
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d1t, k2d11, k4d1t1t],
            val, theta_i[j]
        )
        return val

    n = log_theta.shape[0] - 1
    theta_i = jnp.exp(log_theta[i, :])
    
    # Diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag)

    # Non diagonal Kronecker factors
    diag = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [k1ns, k2dt0, k2dtt, k4dt0t0],
        diag, theta_i[i])
    
    # Diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag)
    
    # Last Kronecker factor
    diag = k2d01(diag)
    return diag


def diag_zero_fun(log_theta, diag, i, state):
    return 0*diag


@partial(jit, static_argnames=["n_state"])
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
    p = jnp.ones(2**n_state)
    return lax.cond(
        state[-1] == 0,
        diag_zero_fun,
        _kron_prim_diag,
        log_theta, p, i, state
        )


def _kron_met_diag(
        log_theta: jnp.ndarray,
        diag: jnp.ndarray,
        i: int,
        state: jnp.ndarray,
        ) -> jnp.ndarray:

    # Single Kronecker vector product
    def loop_body(j, val):
        val = lax.switch(
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d11, k2d1t, k4d11tt],
            val, theta_i[j]
        )
        return val
    
    n = log_theta.shape[0] - 1
    theta_i = jnp.exp(log_theta[i, :])

    # Diagonal Kronecker vector products
    diag = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=loop_body,
        init_val=diag)

    # Non diagonal Kronecker vector products
    diag = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [k1ns, k2dtt, k2dt0, k4dtt00],
        diag, theta_i[i]
    )

    # diagonal Kronecker factors
    diag = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=loop_body,
        init_val=diag
    )

    # Last Kronecker factor
    diag = k2d0t(diag, theta_i[n])

    return diag


@partial(jit, static_argnames=["n_state"])
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
    diag = jnp.ones(2**n_state)
    return lax.cond(
        state[-1] == 0,
        diag_zero_fun,
        _kron_met_diag,
        log_theta, diag, i, state
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
    # Single Kronecker vector product
    def loop_body(j, val):
        val = lax.switch(
            state[2*j] + 2 * state[2*j+1],
            [k1d00, k2d10, k2d10, k4d100t],
            val, theta_i[j]
            )
        return val
    n = log_theta.shape[0] - 1
    diag = jnp.ones(2**n_state)
    theta_i = jnp.exp(log_theta[n,:])
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
        k2dt0,
        k1ns,
        diag, theta_i[n]
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
        ind = state[2*i] + 2*state[2*i+1] + (1 - pt_first)*4
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
    p = lax.cond(state[-1] == 0,
                keep_all,
                keep_col2,
                p)
    return p