import numpy as np
import jax.numpy as jnp
import metmhn.jx.vanilla as mhn

# This contains the important functions for the case that state_size = 1
# This needs to be done separately, as otherwise reshape(-1, 4) throws an error
# in the tracing process of jax.


def small_Q(log_theta: jnp.array, d_e: jnp.array) -> jnp.array:
    base_r =  jnp.diagonal(log_theta)
    b_r = jnp.exp(base_r.at[:-1].get() - d_e)
    e_seed = jnp.exp(log_theta.at[:-1, -1].get()) + 1.
    
    row1 = [-jnp.exp(base_r).sum(), 0.]
    row2 = [jnp.exp(log_theta.at[-1, -1].get()), -jnp.sum(b_r*e_seed)]
    return jnp.array([row1, row2])


def kronvec(log_theta: jnp.array, p: jnp.array, d_e: jnp.array,
            diag: bool = True, transpose: bool = False) -> jnp.array:

    if diag:
        Q = small_Q(log_theta=log_theta, d_e=d_e)
        if not transpose:
            return Q @ p
        else:
            return Q.T @ p
    else:
        if not transpose:
            return jnp.array([0., 
                              p.at[0].get() * jnp.exp(log_theta.at[-1, -1].get())])
        else:
            return jnp.array([p.at[1].get() * jnp.exp(log_theta.at[-1, -1].get()), 
                              0.])


def R_i_inv_vec(log_theta: jnp.array, x: jnp.array, d_e: jnp.array, 
                transpose: bool = False) -> jnp.array:

    R = jnp.eye(2) - small_Q(log_theta=log_theta, d_e=d_e)

    b = x.copy()

    if not transpose:
        b = b.at[0].divide(R.at[0, 0].get())
        b = b.at[1].add(-(b.at[0].get() * R.at[1, 0].get()))
        b = b.at[1].divide(R.at[1, 1].get())
    else:
        b = b.at[1].divide(R.at[1, 1].get())
        b = b.at[0].add(-(b.at[1].get() * R.at[1, 0].get()))
        b = b.at[0].divide(R.at[0, 0].get())
    
    return b


def x_partial_Q_y(log_theta: jnp.array, x: jnp.array, y: jnp.array, 
                  d_e: jnp.array) -> jnp.array:

    z = jnp.zeros_like(log_theta)
    n = log_theta.shape[0]

    for i in range(n):
        theta_ii = jnp.exp(log_theta.at[i, i].get()) 
        theta_iM = jnp.exp(log_theta.at[i, -1].get())
        
        dQ = jnp.diag(jnp.array([1., jnp.exp(-d_e)*(1 + theta_iM)]))
        z = z.at[i, i].set(-theta_ii * x @ dQ @ y)
        
        dQ = jnp.diag(jnp.array([0., -theta_ii * theta_iM*jnp.exp(-d_e)]))
        z = z.at[i, -1].set(x @ dQ @ y)
    theta_MM = jnp.exp(log_theta.at[-1, -1].get())
    z = z.at[-1, -1].set(x @ jnp.array([[-theta_MM, 0.], [theta_MM, 0.]]) @ y)
    d_Q_de = small_Q(log_theta, d_e)
    d_Q_de = d_Q_de.at[:,0].set(0.)
    d_d_e = jnp.zeros(n)
    d_d_e = d_d_e.at[-1].set(-x @ d_Q_de @ y)
    return z, d_d_e


def _lp_coupled(log_theta: jnp.array, log_theta_sd: jnp.array,
                d_e: jnp.array) -> jnp.array:
    n = log_theta.shape[0]-1
    p = jnp.array([1., 0.])
    pTh1_joint = R_i_inv_vec(log_theta, p, d_e, transpose = False)

    # Prim conditional distribution at first sampling, to be used as starting dist for second sampling
    pTh1_cond_obs = pTh1_joint.copy()
    pTh1_cond_obs = pTh1_cond_obs.at[0].set(0.)

    met = jnp.array([0]*n+[1], dtype=jnp.int8)
    pTh2 = mhn.R_inv_vec(log_theta_sd, pTh1_cond_obs, 1., met)
    return jnp.log(pTh2.at[-1].get())


def _g_coupled(log_theta: np.array, log_theta_sd: jnp.array, 
               d_e: np.array) -> tuple[np.array, np.array, np.array, np.array]:
    n = log_theta.shape[0] - 1
    met = jnp.array([0]*n+[1], dtype=jnp.int8)
    p = jnp.array([1., 0.])
    
    # Joint and met-marginal distribution at first sampling
    pTh1_joint = R_i_inv_vec(log_theta, p, d_e, transpose = False)
    # Prim conditional distribution at first sampling, to be used as starting dist for second sampling
    pTh1_cond_obs = pTh1_joint.copy()
    pTh1_cond_obs = pTh1_cond_obs.at[0].set(0.)
    
    # Derivative of pTh2 = M(I-Q_sd)^(-1)pth1_cond
    g_1, d_sd, pTh2_marg = mhn.gradient(log_theta_sd, 1., met, pTh1_cond_obs)
    # q = (pD/pTh_2)^T M(I-Q_sd)^(-1)
    q = jnp.zeros(2)
    q = q.at[-1].set((1/pTh2_marg.at[-1].get()))
    q = mhn.R_inv_vec(log_theta_sd, q, 1., met, transpose = True)
    
    # Derivative of pth1_cond
    q = q.at[0].set(0.)
    p = R_i_inv_vec(log_theta, q, d_e, transpose = True)
    g_2, d_fd = x_partial_Q_y(log_theta, p, pTh1_joint, d_e)
    score = jnp.log(pTh2_marg.at[-1].get())
    return score, g_1 + g_2, d_fd, d_sd