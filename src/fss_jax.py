import jax.numpy as jnp
from jax import jit, lax, vmap
from functools import partial

# functions to calculate the Q_sync p
@partial(jit, static_argnums=4)
def kronvec_sync_jit(theta_i: jnp.array,
                     p: jnp.array,
                     start: int,
                     stop: int,
                     n: int) -> jnp.array:
    """
    Calculates the diagonal Kronecker vector product of Q_sync(theta_ij) for all j in range(start stop)
    Arguments
        theta_i (jnp.array): i-th row of theta matrix
        p (jnp.array): state vector p
        start (int): start index
        stop (int): stop index
        n (int): number of mutations
    Returns:
        jnp.array: Q_j^(sync_i) p for j in range(start, stop)
    """
    def loop_body(j, p):
        p = p.reshape((2 ** (2 * n - 1), 4), order="C")
        #p = p.at[:, (1, 2)].set(0.)
        #p = p.at[:, 3].multiply(theta_i[j])
        theta_slice = jnp.array([1, 0, 0, theta_i[j]])
        # Scale the columns of p by the entries of theta_slice and return in Fortran order
        p = vmap(lambda ti, p: ti*p, (None, 0), 1)(theta_slice, p)
        p = p.ravel()#flatten(order = "F")
        return p
    return lax.fori_loop(start, stop, loop_body, p)


@partial(jit, static_argnums=3)
def kronvec_sync(log_theta: jnp.array, p: jnp.array, i: int, n: int, diag: bool) -> jnp.array:
    """
    Calculates the i-th Kronecker vector product of Q^(sync) p
    Args:
        log_theta (jnp.array): Logarithmic theta matrix
        p (jnp.array): state vector
        i (int): Current index of sumation
        n (int): Number of mutations
        diag (bool): Flag indicating wether to build the diagonal of Q
    Returns :
        jnp.array: Q^(sync_i) p
    """
    theta_i = jnp.exp(log_theta.at[i, :].get())
    # Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, 0, i, n)

    # Non Diagonal Kronecker factors
    p = p.reshape((2**(2*n-1), 4), order="C")
    # These operations are done in place as long as the function is jitted
    # ToDo: Replace with jax.matmul as soon as it is optimized for sparse matrices
    p = p.at[:, (1, 2)].set(0.)
    p = p.at[:, 3].set(theta_i[i] * p.at[:, 0].get())
    p = lax.cond(diag,
                 lambda p: p.at[:, 0].set(-p.at[:, 3].get()),
                 lambda p: p.at[:, 0].set(0.), p)
    p = p.flatten(order="F")

    # Non Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, i+1, n, n)

    # Diagonal Kronecker factor for seeding
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 1].set(0.)
    p = p.flatten(order="F")
    return p


def sync_transp_q(p: jnp.array, tij: float) -> jnp.array:
    """
    Returns the product of the transposed non diagonal Kronecker factor for Q_sync and a vector p
    Args:
        p (jnp.array): state vector
        tij (float): i,j-th entry of theta
    Returns:
        jnp.array
    """
    p = p.at[:, 0].add(-p.at[:, 3].get())
    p = p.at[:, 0].multiply(-tij)
    return p


@partial(jit, static_argnums=3)
def kronvec_sync_transp(log_theta: jnp.array, p: jnp.array, i: int, n: int, diag:  bool = True) -> jnp.array:
    """
    Calculates the transpose i-th Kronecker vector product of Q^(sync) p
    Args:
        log_theta (jnp.array): Logarithmic theta matrix
        p (jnp.array): state vector
        i (int): Current index of sumation
        n (int): Number of mutations
        diag (bool): Flag indicating wether to build the diagonal of Q
    Returns :
        jnp.array: Q^(sync_i) p
    """
    theta_i = jnp.exp(log_theta.at[i, :].get())
    # Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, 0, i, n)

    # Non diagonal Kronecker factor
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = p.at[:, (1, 2)].set(0.)
    p = lax.cond(diag,
                 sync_transp_q,
                 lambda p, tij: p.at[:, 0].set(p.at[:, 3].get()*theta_i[i]), p, theta_i[i])
    p = p.at[:, 3].set(0.)
    p = p.flatten(order="F")

    # Diagonal Kronecker factors
    p = kronvec_sync_jit(theta_i, p, i+1, n, n)

    # Diagonal Kronecker factor for seeding
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 1].set(0.)
    p = p.flatten(order="F")
    return p


@partial(jit, static_argnums=4)
def kronvec_met_jit(theta_i: jnp.array,
                     p: jnp.array,
                     start: int,
                     stop: int,
                     n: int) -> jnp.array:
    """
    Calculates the diagonal Kronecker vector product of Q_met(theta_ij) for all j in range(start stop)
    Arguments
        theta_i (jnp.array): i-th row of theta matrix
        p (jnp.array): state vector p
        start (int): start index
        stop (int): stop index
        n (int): number of mutations
    Returns:
        jnp.array: Q_j^(sync_i) p for j in range(start, stop)
    """
    def loop_body(j, p):
        p = p.reshape((2 ** (2 * n - 1), 4), order="C")
        #p = p.at[:, (2, 3)].multiply(theta_i[j])
        #p = p.flatten(order="F")
        theta_slice = jnp.array([1, theta_i[j], theta_i[j], 1])
        p = vmap(lambda ti, p: ti * p, (None, 0), 1)(theta_slice, p)
        p = p.ravel()
        return p

    return lax.fori_loop(start, stop, loop_body, p)


@partial(jit, static_argnums=3)
def kronvec_met(theta: jnp.array, p: jnp.array, i: int, n: int, diag: bool = True) -> jnp.array:
    """
   Calculates the i-th Kronecker vector product of Q^(met) p
    Args:
          theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
    Returns:
          np.array: Q_met p
    """
    theta_i = jnp.exp(theta.at[i, :].get())
    # Diagonal 4x4 Kronecker factors j<i
    p = kronvec_met_jit(theta_i, p, 0, i, n)

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = lax.cond(diag,
                 lambda p: p.at[:, (0, 1)].set(-p.at[:, (2, 3)].get()),
                 lambda p: p.at[:, (0, 1)].set(0), p
                 )
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    p = kronvec_met_jit(theta_i, p, i+1, n, n)

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 0].set(0)
    p = p.at[:, 1].set(theta_i[n]*p.at[:, 1].get())
    p = p.flatten(order="F")
    return p


def met_lambda_diag(p: jnp.array, tij: float) -> jnp.array:
    """
     Returns the product of the transposed non diagonal Kronecker factor for Q_met and a vector p
     Args:
         p (jnp.array): state vector
         tij (float): i,j-th entry of theta
     Returns:
         jnp.array
     """
    p = p.at[:, 0].set((p.at[:, 2].get() - p.at[:, 0].get()) * tij)
    p = p.at[:, 1].set((p.at[:, 3].get() - p.at[:, 1].get()) * tij)
    p = p.at[:, (2, 3)].set(0)
    return p


def met_lambda_no_diag(p: jnp.array, tij: float):
    """
     Returns the product of the transposed non diagonal Kronecker factor for Q_met and a vector p
     if diag is set to false
     Args:
         p (jnp.array): state vector
         tij (float): i,j-th entry of theta
     Returns:
         jnp.array
     """
    p = p.at[:, 0].set(p.at[:, 2].get() * tij)
    p = p.at[:, 1].set(p.at[:, 3].get() * tij)
    p = p.at[:, (2, 3)].set(0)
    return p


@partial(jit, static_argnums=3)
def kronvec_met_transp(theta: jnp.array, p: jnp.array, i: int, n: int, diag: bool = True) -> jnp.array:
    """
    Calculates the transpose of the i-th Kronecker vector product of Q^(met) p
    Args:
          theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
    Returns:
          np.array: Q_met p
    """
    theta_i = jnp.exp(theta.at[i, :].get())
    # Diagonal 4x4 Kronecker factors j<i
    p = kronvec_met_jit(theta_i, p, 0, i, n)

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = lax.cond(diag, met_lambda_diag, met_lambda_no_diag, p, theta_i[i])
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    p = kronvec_met_jit(theta_i, p, i+1, n, n)

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 0].set(0)
    p = p.at[:, 1].set(theta_i[n]*p.at[:, 1].get())
    p = p.flatten(order="F")
    return p


@partial(jit, static_argnums=4)
def kronvec_prim_jit(theta_i: jnp.array,
                     p: jnp.array,
                     start: int,
                     stop: int,
                     n: int) -> jnp.array:
    """
    Calculates the diagonal Kronecker vector product of Q_prim(theta_ij) for all j in range(start stop)
    Arguments
        theta_i (jnp.array): i-th row of theta matrix
        p (jnp.array): state vector p
        start (int): start index
        stop (int): stop index
        n (int): number of mutations
    Returns:
        jnp.array: Q_j^(sync_i) p for j in range(start, stop)
    """
    def loop_body(j, p):
        p = p.reshape((2 ** (2 * n - 1), 4), order="C")
        #p = p.at[:, 1].set(theta_i[j]*p.at[:, 1].get())
        #p = p.at[:, 3].set(theta_i[j]*p.at[:, 3].get())
        theta_slice = jnp.array([1, theta_i[j], theta_i[j], 1])
        # Scale the columns of p by the entries of theta_slice and return in Fortran order
        p = vmap(lambda ti, p: ti*p, (None, 0), 1)(theta_slice, p)
        p = p.ravel()#flatten(order = "F")
        return p
    return lax.fori_loop(start, stop, loop_body, p)


@partial(jit, static_argnums=3)
def kronvec_prim(theta: jnp.array, p: jnp.array, i: int, n: int, diag: bool = True) -> jnp.array:
    """
    This function computes asynchroneous part of primary transitions of Q*p implicitly
    Args:
          theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
    Returns:
          np.array: Q_prim p
    """
    theta_i = jnp.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    p = kronvec_prim_jit(theta_i, p, 0, i, n)

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = p.at[:, (1, 3)].set(p.at[:, (0, 2)].get() * theta_i[i])
    p = lax.cond(diag,
                 lambda p: p.at[:, (0, 2)].set(-p.at[:, (1, 3)].get()),
                 lambda p: p.at[:, (0, 2)].set(0.), p)
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    p = kronvec_prim_jit(theta_i, p, i+1, n, n)

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 0].set(0.)
    p = p.flatten(order="F")
    return p


def prim_lambda_diag(p: jnp.array, tij: float) -> jnp.array:
    """
     Returns the product of the transposed non diagonal Kronecker factor for Q_prim and a vector p
     Args:
         p (jnp.array): state vector
         tij (float): i,j-th entry of theta
     Returns:
         jnp.array
     """
    p = p.at[:, 0].set((p.at[:, 1].get() - p.at[:, 0].get()) * tij)
    p = p.at[:, 2].set((p.at[:, 3].get()- p.at[:, 2].get()) * tij)
    p = p.at[:, (1, 3)].set(0.)
    return p


def prim_lambda_no_diag(p: jnp.array, tij: float):
    """
     Returns the product of the transposed non diagonal Kronecker factor for Q_prim and a vector p
     if no diagonal should be built
     Args:
         p (jnp.array): state vector
         tij (float): i,j-th entry of theta
     Returns:
         jnp.array
     """
    p = p.at[:, 0].set(p.at[:, 1].get() * tij)
    p = p.at[:, 2].set(p.at[:, 3].get() * tij)
    p = p.at[:, (1, 3)].set(0.)
    return p


@partial(jit, static_argnums=3)
def kronvec_prim_transp(theta: jnp.array, p: jnp.array, i: int, n: int, diag: bool = True) -> jnp.array:
    """
    This function computes the transpose of the asynchroneous part of the primary transitions of Q*p implicitly
    Args:
          theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
    Returns:
          np.array: Q_prim p
    """
    theta_i = jnp.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    p = kronvec_prim_jit(theta_i, p, 0, i, n)

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p = p.at[:, (1, 3)].set(p.at[:, (0, 2)].get() * theta_i[i])
    p = lax.cond(diag,
                 prim_lambda_diag,
                 prim_lambda_no_diag, p, theta_i[i])
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    p = kronvec_prim_jit(theta_i, p, i+1, n, n)

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 0].set(0.)
    p = p.flatten(order="F")
    return p


@partial(jit, static_argnums=2)
def kronvec_seed(theta: jnp.array, p: jnp.array, n: int, diag: bool = True) -> jnp.array:
    """
    This function computes the seeding part of metastatic transitions of Q p implicitly
    Args:
          theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
    Returns:
          jnp.array: Q_seed p
    """
    theta_n = jnp.exp(theta[n, :])
    # Diagonal 4x4 Kronecker factors j<i
    p = kronvec_prim_jit(theta_n, p, 0, n, n)

    # Non diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 1].set(theta_n[n] * p.at[:, 0].get())
    p = lax.cond(diag,
                 lambda p: p.at[:, 0].set(-1. * p.at[:, 1].get()),
                 lambda p: p.at[:, 0].set(0.), p)
    p = p.flatten(order="F")
    return p

@partial(jit, static_argnums=2)
def kronvec_seed_transp(theta: jnp.array, p: jnp.array, n: int, diag: bool = True) -> jnp.array:
    """
    This function computes the seeding part of metastatic transitions of Q^T p implicitly
    Args:
          theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
    Returns:
          jnp.array: Q_seed^T p
    """
    theta_n = jnp.exp(theta[n, :])
    # Diagonal 4x4 Kronecker factors j<i
    p = kronvec_prim_jit(theta_n, p, 0, n, n)

    # Non diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.at[:, 1].set(theta_n[n] * p.at[:, 0].get())
    p = lax.cond(diag,
                 lambda p, tnn: p.at[:, 0].set((p.at[:, 1].get() - p.at[:, 0].get())*tnn),
                 lambda p, tnn: p.at[:, 0].set(p.at[:, 1].get()*tnn), p, theta_n[n])
    p = p.at[:, 1].set(0.)
    p = p.flatten(order="F")
    return p


@partial(jit, static_argnums=3)
def qvec(log_theta: jnp.array, p: jnp.array, diag: bool, n: int) -> jnp.array:
    """
    This function computes Q*p implicitly
    Args:
          log_theta (jnp.array): Logarithmic theta matrix
          p (jnp.array): state vector
          diag (bool): Flag indicating whether to build the diagonal
          n (int): number of mutations
    Returns:
          np.array: Q p
    """
    def sum_func(i, old):
        old = old.at[:].add(kronvec_sync(log_theta, p, i, n, diag=diag))
        old = old.at[:].add(kronvec_prim(log_theta, p, i, n, diag=diag))
        old = old.at[:].add(kronvec_met(log_theta, p, i, n, diag=diag))
        return old

    tmp = lax.fori_loop(0, n, sum_func,  jnp.zeros_like(p))
    tmp = tmp.at[:].add(kronvec_seed(log_theta, p, n, diag=diag))
    return tmp
