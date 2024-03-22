import jax.numpy as jnp
import metmhn.jx.vanilla as mhn
from metmhn.jx.likelihood import diagnosis_theta

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



def _lp_coupled_0(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  state_joint:jnp.ndarray) -> jnp.ndarray:
    """This computes the log. prob to observe a PT and a PT in the same patient at the same time

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_joint (jnp.ndarray): Bitstring, genotypes of PT and MT
        n_prim (jnp.ndarray): Number of nonzero bits in PT-part of state_joint
        n_met (jnp.ndarray): Number of nonzero bit in MT-part of state_joint

    Returns:
        jnp.ndarray: log(P(state_joint|Theta, d_p, d_m))
    """
    p0 = jnp.zeros(2)
    p0 = p0.at[0].set(1.)
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p0, d_p_le, d_m_le) 
    pf_pTh1_cond_obs = pTh1_joint * jnp.array([0,d_p_le])
    mf_pTh1_cond_obs = pTh1_joint * jnp.array([0, d_m_le])
    met = jnp.append(state_joint[1::2], 1)
    log_theta_scal = diagnosis_theta(log_theta, log_d_m)
    pf_pTh2 = mhn.R_inv_vec(log_theta_scal, pf_pTh1_cond_obs, met)

    prim = state_joint[0::2]
    theta_pt = log_theta.at[:-1,-1].set(0.)
    theta_pt = diagnosis_theta(theta_pt, log_d_p)
    mf_pTh2 = mhn.R_inv_vec(theta_pt, mf_pTh1_cond_obs, prim)
    return jnp.log(pf_pTh2[-1] + mf_pTh2[-1])


def _lp_coupled_1(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  state_joint: jnp.ndarray) -> jnp.ndarray:
    """This computes the log. prob to first observe a PT and later a MT in the same patient

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_joint (jnp.ndarray): Bitstring, genotypes of PT and MT
        n_prim (jnp.ndarray): Number of nonzero bits in PT-part of state_joint
        n_met (jnp.ndarray): Number of nonzero bit in MT-part of state_joint

    Returns:
        jnp.ndarray: log(P(state_joint|Theta, d_p, d_m))
    """
    p0 = jnp.zeros(2)
    p0 = p0.at[0].set(1.)
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p0, d_p_le, d_m_le)
    pTh1_cond_obs = jnp.append(jnp.zeros(1), pTh1_joint[-1] * d_p_le) 
    met = jnp.append(state_joint[1::2], 1)
    log_theta_scal = diagnosis_theta(log_theta, log_d_m)
    pTh2 = mhn.R_inv_vec(log_theta_scal, pTh1_cond_obs, met)
    return jnp.log(pTh2[-1])


def _lp_coupled_2(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  state_joint: jnp.ndarray) -> jnp.ndarray:
    """This computes the log. prob to first observe a MT and later a PT in the same patient

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_joint (jnp.ndarray): Bitstring, genotypes of PT and MT
        n_prim (jnp.ndarray): Number of nonzero bits in PT-part of state_joint
        n_met (jnp.ndarray): Number of nonzero bit in MT-part of state_joint

    Returns:
        jnp.ndarray: log(P(state_joint|Theta, d_p, d_m))
    """
    p0 = jnp.zeros(2)
    p0 = p0.at[0].set(1.)
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p0, d_p_le, d_m_le)
    pTh1_cond_obs = jnp.append(jnp.zeros(1), pTh1_joint[-1]*d_m_le)
    prim = state_joint[0::2]
    theta_pt = log_theta.at[:-1,-1].set(0.)
    theta_pt = diagnosis_theta(theta_pt, log_d_p)
    pTh2 = mhn.R_inv_vec(theta_pt, pTh1_cond_obs, prim)
    return jnp.log(pTh2[-1])


def marginal_obs_pt_first(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, pTh1_joint: jnp.ndarray, state_joint: jnp.ndarray,
                          met: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
        This calculates partial derivatives of z = D_M(D_M-Q_M)^{-1} S(prim) pTh_1 =  D_M(D_M-Q_M)^{-1} pTh_{cond-obs} wrt theta, log_d_p and log_d_m
    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log effects of events in the PT on its observation
        log_d_m (jnp.ndarray): Log effects of events in the MT on its observation
        pTh1_joint (jnp.ndarray): Joint distribution of PTs and MTs at time of PT-observation
        state_joint (jnp.ndarray): Pair of observed PT and MT genotypes
        met (jnp.ndarray): Genotype of Metastasis
        n_joint (int): Number of non zero entries in state_joint
        n_met (int): Number of non zero entries in met

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Log. score, partial derivatives wrt to theta, log_d_p and log_d_m
    """
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_cond_obs = jnp.append(jnp.zeros(1), pTh1_joint[-1])*d_p_le

    # Derivative of pTh2 = M(I-Q_sd)^(-1)pth1_cond
    log_theta_dm = diagnosis_theta(log_theta, log_d_m)
    g_1, d_dm_1, pTh2 = mhn.gradient(log_theta_dm,  met, pTh1_cond_obs)
    exp_score = pTh2[-1]
    
    # q = (pD/score)^T (I-Q D_met^{-1})^{-1} S(x)
    q = jnp.zeros(2)
    q = q.at[-1].set(1/exp_score)
    q = mhn.R_inv_vec(log_theta_dm, q, met, transpose = True)

    p = q*jnp.array([0, d_p_le])
    d_dp_1 = jnp.zeros_like(log_d_p)
    d_dp_1 = d_dp_1.at[-1].set(jnp.dot(p, pTh1_joint))

    return exp_score, g_1, d_dp_1, d_dm_1, p


def marginal_obs_mt_first(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, pTh1_joint: jnp.ndarray, state_joint: jnp.ndarray,
                          prim: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
        This calculates partial derivatives of z = D_P(D_P-Q_P)^{-1} S(met) pTh_1 =  D_P(D_P-Q_P)^{-1} pTh_{cond-obs} wrt theta, log_d_p and log_d_m
    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log effects of events in the PT on its observation
        log_d_m (jnp.ndarray): Log effects of events in the MT on its observation
        pTh1_joint (jnp.ndarray): Joint distribution of PTs and MTs at time of MT-observation
        state_joint (jnp.ndarray): Pair of observed PT and MT genotypes
        prim (jnp.ndarray): Genotype of primary tumor
        n_joint (int): Number of non zero entries in state_joint
        n_prim (int): Number of non zero entries in prim

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: Log. score, partial derivatives wrt to theta, log_d_p and log_d_m
    """
    d_m_le = jnp.exp(log_d_m[-1])
    pTh1_cond_obs = jnp.append(jnp.zeros(1), pTh1_joint[-1]*d_m_le)

    # Derivative of pTh2 = D_P(D_P-Q_P)^(-1)pth1_cond
    log_theta_dp = log_theta.at[:-1,-1].set(0.)
    log_theta_pt = diagnosis_theta(log_theta_dp, log_d_p)

    g_1, d_dp_1, pTh2 = mhn.gradient(log_theta_pt, prim, pTh1_cond_obs)
    g_1 = g_1.at[:-1,-1].set(0.0)
    exp_score = pTh2[-1]
    
    # q = (pD/score)^T (I-Q D_met^{-1})^{-1} S(x)
    q = jnp.zeros(2)
    q = q.at[-1].set(1/exp_score)
    q = mhn.R_inv_vec(log_theta_pt, q, prim, transpose = True)
    
    p = q*jnp.array([0, d_m_le])
    d_dm_1 = jnp.zeros_like(log_d_m)
    d_dm_1 = d_dm_1.at[-1].set(jnp.dot(p, pTh1_joint))

    return exp_score, g_1, d_dp_1, d_dm_1, p



def _g_coupled_0(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
               state_joint: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """This computes the log. prob. to observe a PT and MT in unknown order in the same patient and 
    its gradients wrt to theta, d_p and d_m

    Args:
        log_theta (jnp.ndarray): Thetamatrix with logarithmic entries
        log_d_p (jnp.array): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_joint (jnp.ndarray): Bitstring, genotypes of coupled PT and MT
        n_prim (int): Number of nonzero entries in PT-part of state_joint
        n_met (int): Number of nonzero entries in MT-part of state_joint
    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: log_prob, grad wrt. theta,
            grad wrt. d_p, grad wrt. d_m
    """
    prim = state_joint[::2]
    met = jnp.append(state_joint[1::2], 1)
    p = jnp.zeros(2)
    p = p.at[0].set(1.)
    
    # Joint and met-marginal distribution at first sampling
    d_m_le = jnp.exp(log_d_m[-1])
    d_p_le = jnp.exp(log_d_p[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p, d_p_le, d_m_le)
    
    pf_exp_score, pf_g_1, pf_d_dp_1, pf_d_dm_1, pf_p = marginal_obs_pt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, met)
    mf_exp_score, mf_g_1, mf_d_dp_1, mf_d_dm_1, mf_p =  marginal_obs_mt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, prim)
    full_score = pf_exp_score + mf_exp_score

    g_2, d_dp_2, d_dm_2 = q_inv_deriv_pth(log_theta, d_p_le, d_m_le, (pf_p*pf_exp_score + mf_p*mf_exp_score)/full_score, pTh1_joint)
    
    d_dm = (pf_d_dm_1*pf_exp_score + mf_d_dm_1*mf_exp_score)/full_score - d_dm_2
    d_dp = (pf_d_dp_1*pf_exp_score + mf_d_dp_1*mf_exp_score)/full_score - d_dp_2
    grad_th = (pf_g_1*pf_exp_score + mf_g_1*mf_exp_score)/full_score + g_2
    
    return jnp.log(full_score), grad_th, d_dp, d_dm


def _g_coupled_1(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
               state_joint: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """This computes the log. prob to first observe a PT and then later a MT in the same patient and 
    its gradients wrt to theta, d_p and d_m

    Args:
        log_theta (jnp.ndarray): Thetamatrix with logarithmic entries
        log_d_p (jnp.array): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_joint (jnp.ndarray): Bitstring, genotypes of coupled PT and MT
        n_prim (int): Number of nonzero entries in PT-part of state_joint
        n_met (int): Number of nonzero entries in MT-part of state_joint
    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: log_prob, grad wrt. theta,
            grad wrt. d_p, grad wrt. d_m
    """
    met = jnp.append(state_joint.at[1::2].get(), 1)
    p = jnp.zeros(2)
    p = p.at[0].set(1.)
    
    d_p_le = jnp.exp(log_d_p[-1])
    d_m_le = jnp.exp(log_d_m[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p, d_p_le, d_m_le)

    exp_score, g_1, d_dp_1, d_dm_1, p = marginal_obs_pt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, met)
    g_2, d_dp_2, d_dm_2 = q_inv_deriv_pth(log_theta, d_p_le, d_m_le, p, pTh1_joint)
    d_dm = d_dm_1 - d_dm_2
    d_dp = d_dp_1 - d_dp_2

    return jnp.log(exp_score), g_1 + g_2, d_dp, d_dm



def _g_coupled_2(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
               state_joint: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """This computes the log. prob to first observe a MT and later PT in the same patient and 
    its gradients wrt to theta, d_p and d_m

    Args:
        log_theta (jnp.ndarray): Thetamatrix with logarithmic entries
        log_d_p (jnp.array): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_joint (jnp.ndarray): Bitstring, genotypes of coupled PT and MT
        n_prim (int): Number of nonzero entries in PT-part of state_joint
        n_met (int): Number of nonzero entries in MT-part of state_joint
    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]: log_prob, grad wrt. theta,
            grad wrt. d_p, grad wrt. d_m
    """
    prim = state_joint[::2]
    p = jnp.zeros(2)
    p = p.at[0].set(1.)
    
    d_p_le = jnp.exp(log_d_p[-1])
    d_m_le = jnp.exp(log_d_m[-1])
    pTh1_joint = R_i_inv_vec(log_theta, p, d_p_le, d_m_le)
    exp_score, g_1, d_dp_1, d_dm_1, p =  marginal_obs_mt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, prim)
    
    g_2, d_dp_2, d_dm_2 = q_inv_deriv_pth(log_theta, d_p_le, d_m_le, p, pTh1_joint)

    d_dp = d_dp_1 - d_dp_2 
    d_dm = d_dm_1 - d_dm_2

    return jnp.log(exp_score), g_1 + g_2, d_dp, d_dm