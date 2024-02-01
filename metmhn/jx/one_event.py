import jax.numpy as jnp

# This contains the important functions for the case that state_size = 1
# This needs to be done separately, as otherwise reshape(-1, 4) throws an error
# in the tracing process of jax.


def small_Q(log_theta: jnp.ndarray) -> jnp.ndarray:
    """Explicitely build restricted Q-matrix

    Args:
        log_theta (jnp.ndarray): theta matrix with logarithmic effects

    Returns:
        jnp.ndarray: small Q matrix
    """
    base_r =  jnp.diagonal(log_theta)
    b_r = jnp.exp(base_r[:-1])
    e_seed = jnp.exp(log_theta[:-1, -1]) + 1.
    
    row1 = [-jnp.exp(base_r).sum(), 0.]
    row2 = [jnp.exp(log_theta[-1, -1]), -jnp.sum(b_r*e_seed)]
    return jnp.array([row1, row2])


def kronvec(log_theta: jnp.ndarray, p: jnp.ndarray,
            diag: bool = True, transpose: bool = False) -> jnp.ndarray:
    """Multiplies small Q with a vector p

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        p (jnp.ndarray): Probability vector
        diag (bool, optional): Calculate diagonal of Q. Defaults to True.
        transpose (bool, optional): Transpose Q p. Defaults to False.

    Returns:
        jnp.ndarray: Qp
    """

    if diag:
        Q = small_Q(log_theta=log_theta)
        if not transpose:
            return Q @ p
        else:
            return Q.T @ p
    else:
        if not transpose:
            return jnp.array([0., 
                              p[0] * jnp.exp(log_theta[-1, -1])])
        else:
            return jnp.array([p[1] * jnp.exp(log_theta[-1, -1]),0.])


def R_i_inv_vec(log_theta: jnp.ndarray, x: jnp.ndarray, d_p_le : jnp.ndarray, d_m_le: jnp.ndarray, 
                transpose: bool = False) -> jnp.ndarray:
    """returns (D-Q)^{-1} x  or x^T(D-Q)^{-1}

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        x (jnp.ndarray): State-vector
        d_r (jnp.ndarray): Dianosis-rate of the system
        transpose (bool, optional): If true return x^T(D-Q)^{-1}, else return (D-Q)^{-1}. 
            Defaults to False.

    Returns:
        jnp.ndarray: state vector
    """
    D = jnp.array([[1., 0.], [0., 1.]])
    D = D.at[-1, -1].set(d_p_le + d_m_le)
    R = D - small_Q(log_theta=log_theta)

    b = x.copy()

    if not transpose:
        b = b.at[0].divide(R[0, 0])
        b = b.at[1].add(-(b[0] * R[1, 0]))
        b = b.at[1].divide(R[1, 1])
    else:
        b = b.at[1].divide(R[1, 1])
        b = b.at[0].add(-(b[1] * R[1, 0]))
        b = b.at[0].divide(R[0, 0])
    
    return b


def x_partial_Q_y(log_theta: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """This calculates x^T \partial Q(theta) y wrt. theta

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmi entries
        x (jnp.ndarray): vector to be multiplied from the left
        y (jnp.ndarray): vector to be multiplied from the right

    Returns:
        jnp.ndarray: grad wrt. theta
    """
    z = jnp.zeros_like(log_theta)
    n = log_theta.shape[0]

    for i in range(n):
        theta_ii = jnp.exp(log_theta[i, i]) 
        theta_iM = jnp.exp(log_theta[i, -1])
        
        dQ = jnp.diag(jnp.array([1., 1 + theta_iM]))
        z = z.at[i, i].set(-theta_ii * x @ dQ @ y)
        
        dQ = jnp.diag(jnp.array([0., -theta_ii * theta_iM]))
        z = z.at[i, -1].set(x @ dQ @ y)
    theta_MM = jnp.exp(log_theta[-1, -1])
    z = z.at[-1, -1].set(x @ jnp.array([[-theta_MM, 0.], [theta_MM, 0.]]) @ y)
    return z


def q_inv_deriv_pth(log_theta: jnp.ndarray, d_p_le: jnp.ndarray, d_m_le: jnp.ndarray, q: jnp.ndarray, 
                    p: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate partial derivatives of z = q^T (D_{PM}-Q)^{-1} p_0 = q^T p wrt. theta, log_d_p and log_d_m

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        d_p_le (jnp.ndarray): Logarithmic effect of seeding in PT on its observation
        d_m_le(jnp.ndarray): Logarithmic effects of seeding in MT on its observation
        q (jnp.ndarray): Vector to multiply from the left
        p (jnp.ndarray): Vector to multiply from the right

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Partial derivatives of z wrt. theta, log_d_p, log_d_m
    """
    n = log_theta.shape[0]
    q = R_i_inv_vec(log_theta, q, d_p_le, d_m_le, True)
    g_2 = x_partial_Q_y(log_theta, q, p)
    d_dm_2 = jnp.zeros(n)
    d_dm_2 = d_dm_2.at[-1].set(jnp.dot(q*jnp.array([0, d_m_le]), p))
    d_dp_2 = jnp.zeros(n)
    d_dp_2 = d_dp_2.at[-1].set(jnp.dot(q*jnp.array([0, d_p_le]), p))
    return g_2, d_dp_2, d_dm_2