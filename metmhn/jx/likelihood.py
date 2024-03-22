from metmhn.jx.kronvec import (
                        kronvec_sync, 
                        kronvec_met, 
                        kronvec_prim, 
                        kronvec_seed, 
                        kronvec,
                        kron_diag,
                        diag_scal_p,
                        diag_scal_m, 
                        obs_states,
                        diagnosis_theta,
                        partial_diag_scal_p,
                        partial_diag_scal_m,
                        mto_kronvec,
                        mto_kron_diag
                        )

from metmhn.jx import vanilla as mhn
from metmhn.jx import one_event as one
import jax.numpy as jnp
from jax import jit, lax
from functools import partial


def f0(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
       ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    return s, p, m, 0.


def f1(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
       ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = p[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def f2(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
       ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = m[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def f3(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
       ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s[:, 3].sum() + p[:, [1, 3]].sum() + m[:, [2, 3]].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def t1(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
       ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    return s, p, m, jnp.sum(s) + jnp.sum(p) + jnp.sum(m)


def t12(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
        ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = s[:, 0].sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def t3(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray
       ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s.sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def z0(s: jnp.ndarray) -> tuple[jnp.ndarray, float]: 
    return s, 0.


def z1(s: jnp.ndarray) -> tuple[jnp.ndarray, float]: 
    return s.reshape((-1, 2), order="C").flatten(order="F"), 0.


def z3(s: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    s = s.reshape((-1, 4), order="C")
    z = s[:, 3].sum()
    s = s.flatten(order="F")
    return s, z


def deriv_no_seed(i: int, d_th_i: jnp.ndarray, x:jnp.ndarray, y: jnp.ndarray, 
                  log_theta: jnp.ndarray, state:jnp.ndarray, n:int) -> jnp.ndarray:

    z_sync = x * kronvec_sync(log_theta=log_theta, p=y, i=i, state=state)
    z_prim = x * kronvec_prim(log_theta=log_theta, p=y, i=i, state=state)
    z_met = x * kronvec_met(log_theta=log_theta, p=y, i=i, state=state)
    d_th_i = d_th_i.at[-1].set(z_met.sum())
    def body_fun(j, l_val):
        _z_sync, _z_prim, _z_met, _z = lax.switch(
            state[2*j] + 2 * state[2*j+1],
            [f0, f1, f2, f3],
            l_val[0], l_val[1], l_val[2],
        )
        return _z_sync, _z_prim, _z_met, l_val[3].at[j].set(_z)

    z_sync, z_prim, z_met, d_th_i = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=body_fun,
        init_val=(z_sync, z_prim, z_met, d_th_i)
        )

    z_sync, z_prim, z_met, _z = lax.switch(
        state[2*i] + 2 * state[2*i+1],
        [t1, t12, t12, t3],
        z_sync, z_prim, z_met,
        )

    d_th_i = d_th_i.at[i].set(_z)

    z_sync, z_prim, z_met, d_th_i = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=body_fun,
        init_val=(z_sync, z_prim, z_met, d_th_i)
        )
    return d_th_i

@jit
def x_partial_Q_y(log_theta: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, 
                  state: jnp.ndarray) -> jnp.ndarray:
    """This calculates x^T \partial Q(theta) y wrt. theta

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmi entries
        x (jnp.ndarray): vector to be multiplied from the left
        y (jnp.ndarray): vector to be multiplied from the right
        state (jnp.ndarray): Bitstring, genotypes of PT and MT

    Returns:
        jnp.ndarray: grad wrt. theta
    """
    
    n = log_theta.shape[0] - 1
    z = jnp.zeros(shape=(n + 1, n + 1))
   
    def init_z(j, val):
        val = val.at[j, :].set(deriv_no_seed(j, val[:, j], x, y, log_theta, state, n))
        return val
        
    z = lax.fori_loop(lower=0, upper=n, body_fun=init_z, init_val=z) 
    
    z_seed = x * kronvec_seed(log_theta=log_theta, p=y, state=state)
    z = z.at[-1, -1].set(z_seed.sum())

    def body_fun(j, val):

        _z_seed, _z = lax.switch(
            state[2*j] + 2 * state[2*j+1],
            branches=[z0, z1, z1, z3],
            operand=val[0]
        )
        return _z_seed, val[1].at[-1, j].set(_z)

    z_seed, z = lax.fori_loop(0, n, body_fun, (z_seed, z))

    return z


@jit
def x_partial_D_y(log_d_m: jnp.ndarray, log_d_p: jnp.ndarray, state: jnp.ndarray,
                  x: jnp.ndarray, y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """This calculates x^T \partial D(d_p, d_m) y wrt. d_p and d_m

    Args:
        log_d_m (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        log_d_p (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state (jnp.ndarray): Bitstring, genotypes of PT and MT
        x (jnp.ndarray): vector to be multiplied from the left
        y (jnp.ndarray): vector to be multiplied from the right

    Returns:
        tuple[jnp.ndarray, jnp.ndarray]: grad wrt. to d_p, grad wrt. to d_m
    """
    n = log_d_m.shape[0]
    
    def body_fun(i, carry):
        log_dp, log_dm = carry
        log_dp = log_dp.at[i].set(jnp.dot(x, partial_diag_scal_p(log_d_p, state, y, i)))
        log_dm = log_dm.at[i].set(jnp.dot(x, partial_diag_scal_m(log_d_m, state, y, i)))
        return (log_dp, log_dm)
    
    d_dp, d_dm = lax.fori_loop(0, n, body_fun, (jnp.zeros(n), jnp.zeros(n)))
    return d_dp, d_dm


@partial(jit, static_argnames=["transpose", "state_size"])
def R_i_inv_vec(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, x: jnp.ndarray, 
                state: jnp.ndarray, state_size: int, transpose: bool = False) -> jnp.ndarray:
    """This computes The inverse of the resolvent of Q times a vector x: (D-Q)^{-1}x

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        x (jnp.ndarray): Vector of size 2**state_size
        state (jnp.ndarray): Bitstring, genotype of observation
        state_size (int): Number of nonzero entries in state
        transpose (bool, optional): If true calculate vec^T (I - Q)^{-1}. Defaults to False.

    Returns:
        jnp.ndarray: (D-Q)^{-1}x
    """
    y = jnp.ones_like(x)
    lidg = -1. / (kron_diag(log_theta=log_theta, state=state, n_state=state_size) - 
                  (diag_scal_p(log_d_p, state, y) + diag_scal_m(log_d_m, state, y)))
    y = lidg * x
    
    def body_fun(index, carry):
       return lidg * (kronvec(log_theta=log_theta, p=carry, state=state, 
                              diag=False, transpose=transpose) + x)
    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=body_fun,
        init_val=y
    )
    return y


def cond_p_obs(pTh1_joint: jnp.ndarray, state_joint: jnp.ndarray, n_joint: int, n_single: int, pt_first: bool) -> jnp.ndarray:
    """Calculate the distribution of primary tumors/metastases conditioned on the observed genotype of 
    the other tumor at first observation

    Args:
        pTh1_joint (jnp.ndarray): Joint distribution of both tumors at the time of first observation
        state_joint (jnp.ndarray): Pair of primary tumor and metastasis genotypes
        n_joint (int): Number of non-zero entries in state_joint
        n_single (int): Number of non-zero entries in the unobserved tumor 
        pt_first (bool): Logical flag, if true then the primary tumor was first observed, if false the metastasis was first observed

    Returns:
        jnp.ndarray: Conditional distribution
    """
    compatible_states = obs_states(n_joint=n_joint, state=state_joint, pt_first=pt_first)
    poss_states_inds = jnp.where(compatible_states == 1., size=2**(n_single-1))[0]
    pTh1_cond_obs = pTh1_joint[poss_states_inds]
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_single-1)), pTh1_cond_obs)
    return pTh1_cond_obs


def _lp_coupled_0(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  state_joint:jnp.ndarray, n_prim:int, n_met:int) -> jnp.ndarray:
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
    n_joint = n_prim + n_met - 1
    p0 = jnp.zeros(2**n_joint)
    p0 = p0.at[0].set(1.)
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, n_joint)
    pf_pTh1_cond_obs = cond_p_obs(diag_scal_p(log_d_p, state_joint, pTh1_joint), state_joint, n_joint, n_met, True)
    mf_pTh1_cond_obs = cond_p_obs(diag_scal_m(log_d_m, state_joint, pTh1_joint), state_joint, n_joint, n_prim, False)
    
    met = jnp.append(state_joint[1::2], 1)
    log_theta_scal = diagnosis_theta(log_theta, log_d_m)
    pf_pTh2 = mhn.R_inv_vec(log_theta_scal, pf_pTh1_cond_obs, met)

    prim = state_joint[0::2]
    theta_pt = log_theta.at[:-1,-1].set(0.)
    theta_pt = diagnosis_theta(theta_pt, log_d_p)
    mf_pTh2 = mhn.R_inv_vec(theta_pt, mf_pTh1_cond_obs, prim)
    return jnp.log(pf_pTh2[-1] + mf_pTh2[-1])



def _lp_coupled_1(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  state_joint: jnp.ndarray, n_prim: int, n_met: int) -> jnp.ndarray:
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
    joint_size = n_prim + n_met - 1
    p0 = jnp.zeros(2**joint_size)
    p0 = p0.at[0].set(1.)
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, joint_size)
    pTh1_joint = diag_scal_p(log_d_p, state_joint, pTh1_joint)
    
    compatible_states = obs_states(n_joint=joint_size, state=state_joint, pt_first=True)
    poss_states_inds = jnp.where(compatible_states == 1., size=2**(n_met-1))[0]
    
    pTh1_cond_obs = pTh1_joint[poss_states_inds]
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_met-1)), pTh1_cond_obs)
    
    met = jnp.append(state_joint[1::2], 1)
    log_theta_scal = diagnosis_theta(log_theta, log_d_m)
    pTh2 = mhn.R_inv_vec(log_theta_scal, pTh1_cond_obs, met)
    return jnp.log(pTh2[-1])


def _lp_coupled_2(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                  state_joint: jnp.ndarray, n_prim: int, n_met: int) -> jnp.ndarray:
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
    joint_size = n_prim + n_met - 1
    p0 = jnp.zeros(2**joint_size)
    p0 = p0.at[0].set(1.)
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, joint_size)
    pTh1_joint = diag_scal_m(log_d_m, state_joint, pTh1_joint)
    
    compatible_states = obs_states(joint_size, state_joint, False)
    poss_states_inds = jnp.where(compatible_states == 1., size=2**(n_prim-1))[0]
    
    pTh1_cond_obs = pTh1_joint[poss_states_inds]
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_prim-1)), pTh1_cond_obs)
    
    prim = state_joint[0::2]
    theta_pt = log_theta.at[:-1,-1].set(0.)
    theta_pt = diagnosis_theta(theta_pt, log_d_p)
    pTh2 = mhn.R_inv_vec(theta_pt, pTh1_cond_obs, prim)
    return jnp.log(pTh2[-1])

    
def _lp_prim_obs(log_theta: jnp.ndarray, log_d_p: jnp.ndarray,
                 state_pt: jnp.ndarray, n_prim: int) -> jnp.ndarray:
    """This computes the log Prob. to observe an uncoupled primary tumor with genotype state_pt

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Logrithmic effects of muts in PT on its rate of diagnosis
        state_pt (jnp.ndarray): Bitstring, genotype of tumor 
        n_prim (int): Number of non-zero entries in state_prim

    Returns:
        jnp.ndarray: log(P(state_pt| \theta))
    """
    log_theta_pt = log_theta.at[:-1,-1].set(0.)
    log_theta_pt = diagnosis_theta(log_theta_pt, log_d_p)
    p0 = jnp.zeros(2**n_prim)
    p0 = p0.at[0].set(1.)
    pTh = mhn.R_inv_vec(log_theta_pt, p0, state_pt, jnp.ones_like(p0))
    return jnp.log(pTh[-1])


def _lp_met_obs(log_theta: jnp.ndarray, log_d_pt: jnp.ndarray, log_d_mt: jnp.ndarray, 
                state_mt: jnp.ndarray, n_met: int) -> jnp.ndarray:
    """This computes the log Prob. to observe an uncoupled metastatis with genotype state_mt

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_pt (jnp.ndarray): Effects of muts on diagnosis prior to seeding
        log_d_mt (jnp.ndarray): Effects of muts on diagnosis after seeding
        state_mt (jnp.ndarray): Bitstring, genotype of the met.
        n_met (int): Number of nonzero bits in state_mt

    Returns:
        jnp.ndarray: log(P(state_mt | \theta))
    """
    p0 = jnp.zeros(2**n_met)
    p0 = p0.at[0].set(1.)
    d_p, d_m = mhn.scal_d_pt(log_d_pt, log_d_mt, state_mt, jnp.ones(2**n_met))
    d_rates = d_p + d_m
    pTh = mhn.R_inv_vec(log_theta, p0, state_mt, d_rates, False)
    return jnp.log(pTh[-1] * d_rates[-1])

 
@partial(jit, static_argnames=["n_prim"])
def _grad_prim_obs(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, 
                   state_prim: jnp.ndarray, n_prim: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """This computes log prob to observe a PT and its gradients wrt. theta, d_p

    Args:
        log_theta (jnp.ndarray): Theta matrix with log. entries
        log_d_p (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        state_prim (jnp.ndarray): Bitstring, genotype of PT
        n_prim (int): Number of nonzero entries in PT

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: log prob, grad wrt. theta, grad wrt. d_p
    """
    p0 = jnp.zeros(2**n_prim)
    p0 = p0.at[0].set(1.)
    log_theta_pt = log_theta.at[:-1, -1].set(0.0)
    log_theta_pt = diagnosis_theta(log_theta_pt, log_d_p)
    d_th, d_dp, pTh2 = mhn.gradient(log_theta_pt, state_prim, p0)
    d_th = d_th.at[:-1, -1].set(0.0)
    return jnp.log(pTh2[-1]), d_th, d_dp


@partial(jit, static_argnames=["n_met"])
def _grad_met_obs(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
                   state_met: jnp.ndarray, n_met: int
                   ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """This computes the log. prob. to observe an MT and its gradients wrt. theta, d_p and d_m

    Args:
        log_theta (jnp.ndarray): Theta matrix with log. entries
        log_d_p (jnp.ndarray): Log. effects of muts in PT on PT-diagnosis
        log_d_m (jnp.ndarray): Log. effects of muts in MT on MT-diagnosis
        state_met (jnp.ndarray): bitstring, genotype of MT
        n_met (int): Number of nonzero bits in state_met

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: log prob, grad wrt. theta, 
            grad wrt. d_p, grad wrt. d_m
    """

    p0 = jnp.zeros(2**n_met)
    p0 = p0.at[0].set(1.)
    d_p, d_m = mhn.scal_d_pt(log_d_p, log_d_m, state_met, jnp.ones(2**n_met))
    d_rates = d_p + d_m

    pTh = mhn.R_inv_vec(log_theta, p0, state_met, d_rates, False)
    score = pTh[-1]
    q = jnp.zeros(2**n_met)
    q = q.at[-1].set(1/score)
    _, d_dm_1 = mhn.x_partial_D_y(log_d_p, log_d_m, state_met, q/d_rates[-1], pTh)
    q = mhn.R_inv_vec(log_theta, q, state_met, d_rates, True)
    d_dp, d_dm_2 = mhn.x_partial_D_y(log_d_p, log_d_m, state_met, q,pTh) 
    d_th, _ = mhn.x_partial_Q_y(log_theta, q, pTh, state_met)
    return jnp.log(score*d_rates[-1]), d_th, -d_dp, d_dm_1 - d_dm_2


#@partial(jit, static_argnames=["n_joint"])
def q_inv_deriv_pth(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, q: jnp.ndarray, p: jnp.ndarray, 
                    state_joint: jnp.ndarray, n_joint: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate partial derivatives of z = q^T (D_{PM}-Q)^{-1} p_0 = q^T p wrt. theta, log_d_p and log_d_m

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Logarithmic effects of events in PT on its observation
        log_d_m (jnp.ndarray): Logarithmic effects of events in MT on its observation
        q (jnp.ndarray): Vector to multiply from the left
        p (jnp.ndarray): Vector to multiply from the right
        state_joint (jnp.ndarray): Paired primary tumor and metastases state
        n_joint (int): Number of non zero entries in state_joint

    Returns:
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]: Partial derivatives of z wrt. theta, log_d_p, log_d_m
    """
    q = R_i_inv_vec(log_theta, log_d_p, log_d_m, q, state_joint, 
                    n_joint, transpose = True)
    g_2 = x_partial_Q_y(log_theta, q, p, state_joint)
    # Derivative wrt diagnosis effects
    d_dp_2, d_dm_2 = x_partial_D_y(log_d_m, log_d_p, state_joint, q, p)
    return g_2, d_dp_2, d_dm_2


def marginal_obs_pt_first(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, pTh1_joint: jnp.ndarray, state_joint: jnp.ndarray,
                          met: jnp.ndarray, n_joint:int, n_met:int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    pTh1_joint_scal = diag_scal_p(log_d_p, state_joint, pTh1_joint)
    # Select the states where x = prim and z are compatible with met
    p = obs_states(n_joint, state_joint, pt_first=True)
    poss_states_inds = jnp.where(p==1., size=2**(n_met-1))[0]
    pTh1_cond_obs = pTh1_joint_scal[poss_states_inds]
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_met-1)), pTh1_cond_obs)

    # Derivative of pTh2 = M(I-Q_sd)^(-1)pth1_cond
    log_theta_dm = diagnosis_theta(log_theta, log_d_m)
    g_1, d_dm_1, pTh2 = mhn.gradient(log_theta_dm,  met, pTh1_cond_obs)
    exp_score = pTh2[-1]
    
    # q = (pD/score)^T (I-Q D_met^{-1})^{-1} S(x)
    q = jnp.zeros(2**n_met)
    q = q.at[-1].set(1/exp_score)
    q = mhn.R_inv_vec(log_theta_dm, q, met, transpose = True)
    
    p = p*0.
    p = p.at[poss_states_inds].set(q[2**(n_met - 1):])
    d_dp_1, _ = x_partial_D_y(log_d_m, log_d_p, state_joint, p, pTh1_joint)

    return exp_score, g_1, d_dp_1, d_dm_1, p  


def marginal_obs_mt_first(log_theta:jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, pTh1_joint: jnp.ndarray, state_joint: jnp.ndarray,
                          prim: jnp.ndarray, n_joint:int, n_prim:int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    pTh1_joint_scal = diag_scal_m(log_d_m, state_joint, pTh1_joint)
    p = obs_states(n_joint, state_joint, pt_first=False)
    poss_states_inds = jnp.where(p==1., size=2**(n_prim-1))[0]
    pTh1_cond_obs = pTh1_joint_scal[poss_states_inds]
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_prim-1)), pTh1_cond_obs)

    # Derivative of pTh2 = D_P(D_P-Q_P)^(-1)pth1_cond
    log_theta_dp = log_theta.at[:-1,-1].set(0.)
    log_theta_pt = diagnosis_theta(log_theta_dp, log_d_p)

    g_1, d_dp_1, pTh2 = mhn.gradient(log_theta_pt, prim, pTh1_cond_obs)
    g_1 = g_1.at[:-1,-1].set(0.0)
    exp_score = pTh2[-1]
    
    # q = (pD/score)^T (I-Q D_met^{-1})^{-1} S(x)
    q = jnp.zeros(2**n_prim)
    q = q.at[-1].set(1/exp_score)
    q = mhn.R_inv_vec(log_theta_pt, q, prim, transpose = True)
    
    p = p * 0.
    p = p.at[poss_states_inds].set(q[2**(n_prim - 1):])
    _, d_dm_1 = x_partial_D_y(log_d_m, log_d_p, state_joint, p, pTh1_joint)
    return exp_score, g_1, d_dp_1, d_dm_1, p


def _g_coupled_0(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
               state_joint: jnp.ndarray, n_prim: int, n_met: int
               ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    n_joint = n_prim + n_met -1
    p = jnp.zeros(2**(n_prim + n_met - 1))
    p = p.at[0].set(1.)
    
    # Joint and met-marginal distribution at first sampling
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p, state_joint, 
                             n_joint, transpose = False)
    
    pf_exp_score, pf_g_1, pf_d_dp_1, pf_d_dm_1, pf_p = marginal_obs_pt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, met, n_joint, n_met)
    mf_exp_score, mf_g_1, mf_d_dp_1, mf_d_dm_1, mf_p =  marginal_obs_mt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, prim, n_joint, n_prim)
    full_score = pf_exp_score + mf_exp_score

    # Derivative of pth1_cond
    pf_p = diag_scal_p(log_d_p, state_joint, pf_p)*pf_exp_score/full_score
    mf_p = diag_scal_m(log_d_m, state_joint, mf_p)*mf_exp_score/full_score
    g_2, d_dp_2, d_dm_2 = q_inv_deriv_pth(log_theta, log_d_p, log_d_m, pf_p+mf_p, pTh1_joint, state_joint, n_joint)
    
    d_dm = (pf_d_dm_1*pf_exp_score + mf_d_dm_1*mf_exp_score)/full_score - d_dm_2
    d_dp = (pf_d_dp_1*pf_exp_score + mf_d_dp_1*mf_exp_score)/full_score - d_dp_2
    grad_th = (pf_g_1*pf_exp_score + mf_g_1*mf_exp_score)/full_score + g_2
    return jnp.log(full_score), grad_th, d_dp, d_dm


def _g_coupled_1(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
               state_joint: jnp.ndarray, n_prim: int, n_met: int
               ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    n_joint = n_prim + n_met - 1
    p = jnp.zeros(2**n_joint)
    p = p.at[0].set(1.)
    
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p, state_joint, 
                             n_joint, transpose = False)

    exp_score, g_1, d_dp_1, d_dm_1, p = marginal_obs_pt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, met, n_joint, n_met)
    # Derivative of pth1_cond
    p = diag_scal_p(log_d_p, state_joint, p)
    g_2, d_dp_2, d_dm_2 = q_inv_deriv_pth(log_theta, log_d_p, log_d_m, p, pTh1_joint, state_joint, n_joint)
    d_dm = d_dm_1 - d_dm_2
    d_dp = d_dp_1 - d_dp_2

    return jnp.log(exp_score), g_1 + g_2, d_dp, d_dm


def _g_coupled_2(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, 
               state_joint: jnp.ndarray, n_prim: int, n_met: int
               ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
    n_joint = n_prim + n_met - 1
    p = jnp.zeros(2**n_joint)
    p = p.at[0].set(1.)
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p, state_joint, 
                             n_joint, transpose = False)
    exp_score, g_1, d_dp_1, d_dm_1, p =  marginal_obs_mt_first(log_theta, log_d_p, log_d_m, pTh1_joint, state_joint, prim, n_joint, n_prim)
    
    # Derivative of pth1_cond
    p = diag_scal_m(log_d_m, state_joint, p)
    g_2, d_dp_2, d_dm_2 = q_inv_deriv_pth(log_theta, log_d_p, log_d_m, p, pTh1_joint, state_joint, n_joint)
    d_dp = d_dp_1 - d_dp_2 
    d_dm = d_dm_1 - d_dm_2

    return jnp.log(exp_score), g_1 + g_2, d_dp, d_dm