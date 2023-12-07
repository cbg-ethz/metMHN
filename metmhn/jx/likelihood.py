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
                        diagnosis_theta
                        )

from metmhn.jx import vanilla as mhn
import jax.numpy as jnp
from jax import jit, lax
from functools import partial



def f1(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = p[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def f2(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = m[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def f3(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s[:, 3].sum() + p[:, [1, 3]].sum() + m[:, [2, 3]].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def t12(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = s[:, 0].sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def t3(s: jnp.ndarray, p: jnp.ndarray, m: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s.sum() + p.sum() + m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z


def z3(s: jnp.ndarray) -> tuple[jnp.ndarray, float]:
    s = s.reshape((-1, 4), order="C")

    z = s[:, 3].sum()

    s = s.flatten(order="F")

    return s, z


def deriv_no_seed(i:int, g_row_i: jnp.array, g_m_row_i: jnp.array, x: jnp.array, y:jnp.array, 
                  log_theta: jnp.array,  diag_effects: jnp.array, state:jnp.array, n:int) -> tuple[jnp.array, jnp.array]:

    d_e = diag_effects.at[-1].get()
    diag_th = diagnosis_theta(log_theta, diag_effects)
    z_sync = jnp.multiply(x, kronvec_sync(log_theta=diag_th,
                                p=y, i=i, state=state))
    z_prim = jnp.multiply(x, kronvec_prim(diag_log_theta=diag_th,
                                p=y, i=i, d_e=d_e, state=state))
    z_met = jnp.multiply(x, kronvec_met(log_theta=log_theta, diag_eff = diag_effects,
                            p=y, i=i, state=state))
    g_m_row_i = g_m_row_i.at[-1].set(z_met.sum())

    def body_fun(j, l_val):

        _z_sync, _z_prim, _z_met, _z, _z_m = lax.switch(
            state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            [
                lambda s, p, m: (s, p, m, 0., 0.),
                f1,
                f2,
                f3
            ],
            l_val[0],
            l_val[1],
            l_val[2],

        )
        return _z_sync, _z_prim, _z_met, l_val[3].at[j].set(_z), l_val[4].at[j].set(_z_m)

    # Derivatives of off-diagonal entries in row i 
    z_sync, z_prim, z_met, g_row_i, g_m_row_i = lax.fori_loop(
        lower=0,
        upper=i,
        body_fun=body_fun,
        init_val=(z_sync, z_prim, z_met, g_row_i, g_m_row_i)
    )

    # Derivative of diagonal entry in row i
    z_sync, z_prim, z_met, _z, _z_m = lax.switch(
        state.at[2*i].get() + 2 * state.at[2*i+1].get(),
        [
            lambda s, p, m: (
                s, p, m, (jnp.sum(s) + jnp.sum(p)), jnp.sum(m)),
            t12,
            t12,
            t3
        ],
        z_sync,
        z_prim,
        z_met,
    )
    g_row_i = g_row_i.at[i].set(_z)
    g_m_row_i = g_m_row_i.at[i].set(_z_m)

    # Derivatives of off-diagonal entries in row i
    z_sync, z_prim, z_met, g_row_i, g_m_row_i = lax.fori_loop(
        lower=i+1,
        upper=n,
        body_fun=body_fun,
        init_val=(z_sync, z_prim, z_met, g_row_i, g_m_row_i)
    )

    return g_row_i, g_m_row_i


@jit
def x_partial_Q_y(log_theta: jnp.array, diag_effects: jnp.array, x: jnp.array, 
                  y: jnp.array, state: jnp.array) -> tuple[jnp.array, jnp.array]:
    """This function computes x \partial Q y with \partial Q the Jacobian of Q w.r.t. all thetas
    efficiently using the shuffle trick (sic!).

    Args:
        log_theta (jnp.array):  Logarithmic theta values of the MHN, if diagnosis == true, then the log theta entries 
                                are expected to be scaled with the diagnosis rates beforehand 
        x (jnp.array):          x vector to multiply with from the left. Length must equal the number of
                                nonzero entries in the state vector.
        y (jnp.array):          y vector to multiply with from the right. Length must equal the number of
                                nonzero entries in the state vector.
        state (jnp.array):      Binary state vector, representing the current sample's events.
        d_e (jnp.array):        Effect of the seeding on the diagnosis.
        diagnosis (bool):       Wether to calculate the derivative of the diagnosis effects.

    Returns:
        tuple: x \partial_(\Theta_{ij}) Q y,  x \partial_(\Theta_{dj}) Q yfor i, j = 1, ..., n+1
    """
    # Initialize all model parameters
    n = log_theta.shape[0] - 1
    diag_th = diagnosis_theta(log_theta, diag_effects)
    d_e = diag_effects.at[-1].get()
    
    # Calculate the derivatives of intergenomic effects
    # dQ_sync, dQ_prim and dQ_met are separated, 
    # necessary for the calculation of the derivs. of the diagnosis-effects
    z = jnp.zeros(shape=(n + 1, n + 1))
    z_m = jnp.zeros(shape=(n + 1, n + 1))

    def init_z(j, carry):
        g, g_m = deriv_no_seed(j, carry[0].at[:, j].get(), carry[1].at[:, j].get(), 
                               x, y, log_theta, diag_effects, state, n)
        return carry[0].at[j,:].set(g), carry[1].at[j,:].set(g_m)
    
    z, z_m = lax.fori_loop(lower=0, upper=n, body_fun=init_z, init_val=(z, z_m)) 
    
    # Calculate the derivatives of the effects of mutations on the seeding
    z_seed = jnp.multiply(x, kronvec_seed(diag_log_theta=diag_th, p=y, state=state))
    z = z.at[-1, -1].set(z_seed.sum())

    def body_fun(j, val):

        _z_seed, _z = lax.switch(
            state.at[2*j].get() + 2 * state.at[2*j+1].get(),
            branches=[
                lambda s: (s, 0.),
                lambda s: (
                    s.reshape((-1, 2), order="C").flatten(order="F"), 0.),
                lambda s: (
                    s.reshape((-1, 2), order="C").flatten(order="F"), 0.),
                z3
            ],
            operand=val[0]
        )
        return _z_seed, val[1].at[-1, j].set(_z)

    z_seed, z = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=body_fun,
        init_val=(z_seed, z)
    )

    # The seeding is allowed to influence the rate of diagnosis in the PT, 
    # this derivative has to be calculated explicitely
    q_prim_y = lax.fori_loop(
        lower=0,
        upper=n,
        body_fun=lambda j, val: val + kronvec_prim(diag_th, y, j, state, d_e, True, False),
        init_val = jnp.zeros_like(y))
    
    d_d_e = jnp.dot(x, -q_prim_y)
    
    # The derivative of the i-th diagnosis effect is given by the negative sum of the off-diagonal entries 
    # in the i-th column of d-theta
    d_diag = -jnp.sum(z, axis=0) + jnp.diagonal(z)
    d_diag = d_diag.at[-1].add(d_d_e)
    d_diag = d_diag.at[-1].add(jnp.sum(-z_m.at[:-1,-1].get()))
    def diag_loop_fun(j, carry):
        carry = carry.at[j].set(xdQmetddiy(log_theta, diag_effects, state, x, y, j))
        return carry
    d_m_diag = lax.fori_loop(lower=0, upper=n, body_fun=diag_loop_fun, init_val=jnp.zeros(n+1))
    d_diag += d_m_diag
    z += z_m 

    return z, d_diag


@partial(jit, static_argnames=["transpose", "state_size"])
def R_i_inv_vec(log_theta: jnp.ndarray, log_d_p: jnp.ndarray, log_d_m: jnp.ndarray, x: jnp.ndarray, state: jnp.ndarray, 
                state_size: int, transpose: bool = False) -> jnp.ndarray:
    """This computes R_i^{-1} x = ((D_dP + D_dM) - Q)^{-1} x

    Args:
        log_theta (np.array): Log values of the theta matrix
        x (np.array): Vector to multiply with from the right. Length must equal the 2 to the number of
        nonzero entries in the state vector.
        lam (jnp.array): Rate \lambda_i of i-th sampling. Has to be set to 1. if the diagnosis formalism is used.
        state (jnp.array): Binary state vector, representing the current sample's events.
        d_e (jnp.array): Effect of the seeding on diagnosis. Has to be 0 if diagnosis formalism is not used.
        transpose (bool): calculate R^(-T)p else calc R^(-1)p
    Returns:
        jnp.array: R_i^{-1} x
    """
    lidg = -1. / (kron_diag(log_theta=log_theta, state=state, n_state=state_size) - 
                  (diag_scal_p(log_d_p, state, jnp.ones_like(x)) + diag_scal_m(log_d_m, state, jnp.ones_like(x))))
    y = lidg * x
    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=lambda _, val: lidg * (kronvec(log_theta=log_theta, p=val,
                                                state=state, diag=False, transpose=transpose) + x),
        init_val=y
    )

    return y


def _lp_coupled(log_theta: jnp.ndarray, log_d_p:jnp.ndarray, log_d_m: jnp.ndarray, 
                state_joint: jnp.ndarray, n_prim: int, n_met: int, obs_order: int) -> jnp.ndarray:
    """Calculate the log prob to observe a datapoint state_joint

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Logarithmic effects of events on PT-observation
        log_d_m (jnp.ndarray): Logarithmic effects of events on MT-observation
        state_joint (jnp.ndarray): Observed datapoint
        n_prim (int): Number of active events (1s) in PT-part of state_joint
        n_met (int): Number of active events (1s) in MT-part of state_joint
        obs_order (int): Ordering of observations (0: same time, 1: PT->MT, 2: MT->PT)

    Returns:
        jnp.ndarray: log. prob of state_joint
    """
    theta_pt = log_theta.copy()
    theta_pt = theta_pt.at[:-1,-1].set(0.)
    joint_size = n_prim + n_met - 1
    p0 = jnp.zeros(2**joint_size)
    p0 = p0.at[0].set(1.)
    pTh1_joint = R_i_inv_vec(log_theta, log_d_p, log_d_m, p0, state_joint, joint_size)
    pTh1_joint = lax.switch(index = obs_order,
                            branches = [
                                lambda: diag_scal_p(log_d_p, state_joint, pTh1_joint) + diag_scal_m(log_d_m, state_joint, pTh1_joint),
                                lambda: diag_scal_p(log_d_p, state_joint, pTh1_joint),
                                lambda: diag_scal_m(log_d_m, state_joint, pTh1_joint)
                                ])
    
    # Select the states where x = prim and z are compatible with met 
    def pt_first():
        compatible_states = obs_states(n_joint=joint_size, state=state_joint, pt_first=True)
        poss_states_inds = jnp.where(compatible_states == 1., size=2**(n_met-1))[0]
        pTh1_cond_obs = pTh1_joint[poss_states_inds]
        #States where the Seeding didn't happen aren't compatible with met and get probability 0
        pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_met-1)), pTh1_cond_obs)
        met = jnp.append(state_joint[1::2], 1)
        log_theta_scal = diagnosis_theta(log_theta, log_d_m)
        pTh2 = mhn.R_inv_vec(log_theta_scal, pTh1_cond_obs, met)
        return jnp.log(pTh2[-1])
    
    def mt_first():
        compatible_states = obs_states(joint_size, state_joint, False)
        poss_states_inds = jnp.where(compatible_states == 1., size=2**(n_prim-1))[0]
        pTh1_cond_obs = pTh1_joint.at[poss_states_inds].get()
        #States where the Seeding didn't happen aren't compatible with met and get probability 0
        pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_prim-1)), pTh1_cond_obs)
        prim = state_joint[0::2]
        # Seeding has no effect in PTs
        log_theta_pt_scal = diagnosis_theta(theta_pt, log_d_p)
        pTh2 = mhn.R_inv_vec(log_theta_pt_scal, pTh1_cond_obs, prim)
        return jnp.log(pTh2[-1])
    
    log_prob = lax.switch(index = obs_order,
                          branches =[
                              lambda:jnp.log(pTh1_joint[-1]),
                              pt_first,
                              mt_first
                              ])
    return log_prob
    

    
def _lp_prim_obs(log_theta: jnp.ndarray, log_d_p: jnp.ndarray,
                 state_pt: jnp.ndarray, n_prim: int) -> jnp.ndarray:
    """Log Prob. to observe an uncoupled primary tumor with genotype state_pt

    Args:
        log_theta (jnp.ndarray): Theta matrix with logarithmic entries
        log_d_p (jnp.ndarray): Logrithmic effects of muts in PT on its rate of diagnosis
        state_pt (jnp.ndarray): Bitstring, genotype of tumor 
        n_prim (int): Number of non-zero entries in state_prim

    Returns:
        jnp.ndarray: log(P(state_pt| \theta))
    """
    log_theta_pt = log_theta.at[:-1,-1].set(0.)
    log_theta_fd = diagnosis_theta(log_theta_pt, log_d_p)
    p0 = jnp.zeros(2**n_prim)
    p0 = p0.at[0].set(1.)
    pTh = mhn.R_inv_vec(log_theta_fd, p0, state_pt)
    return jnp.log(pTh[-1])


def _lp_met_obs(log_theta: jnp.ndarray, log_d_pt: jnp.ndarray, log_d_mt: jnp.ndarray, 
                state_mt: jnp.ndarray, n_met: int) -> jnp.ndarray:
    """Log Prob. to observe an uncoupled metastatis with genotype state_mt

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
    d_rates = mhn.scal_d_pt(log_d_pt, log_d_mt, state_mt, jnp.ones(2**n_met))
    pTh = mhn.R_inv_vec(log_theta, p0, state_mt, d_rates,False) * d_rates
    return jnp.log(pTh[-1])


#def _grad_met_obs(log_theta_fd: jnp.array, log_theta_sd: jnp.array, state_met: jnp.array, 
#                  n_met: int) -> tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
#    """ Gradient of lp_met_obs for a single sample
#
#    Args:
#        log_theta_fd (jnp.array):   Theta matrix with logarithmic entries, scaled by first diagnosis effects (fd).
#        log_theta_sd (jnp.array):   Theta matrix with logarithmic entries, scaled by second diagnosis effects (sd).
#        state_met (jnp.array):      Bitstring, representing the current sample's genotype.
#        n_met (int):                Number of active events in current sample.
#
#    Returns:
#        tuple:                      Marginal likelihood, gradient wrt. theta, gradient wrt. fd, gradient wrt. sd
#    """
#    p0 = jnp.zeros(2**n_met)
#    p0 = p0.at[0].set(1.)
#    R_1_inv_p_0 = mhn.R_inv_vec(
#        log_theta=log_theta_fd,
#        x=p0,
#        state=state_met)
#    
#    R_1_inv_p_0_half = R_1_inv_p_0.at[:2**(n_met-1)].set(0)
#    
#    R_2_inv_p_1 = mhn.R_inv_vec(
#        log_theta=log_theta_sd,
#        x=R_1_inv_p_0_half,
#        state=state_met,
#    )
#
#    score = jnp.log(R_2_inv_p_1.at[-1].get())
#    
#    lhs = jnp.zeros_like(p0)
#    lhs = lhs.at[-1].set(1. / R_2_inv_p_1.at[-1].get())
#    # lhs = (pD/pTh)^T (I-Q_sd)^-1
#    lhs = mhn.R_inv_vec(log_theta_sd, lhs, state_met, True)
#    dTh_1, d_sd = mhn.x_partial_Q_y(log_theta_sd, lhs, R_2_inv_p_1, state_met)
#    
#    # lhs = (pD/pTh)^T (I-Q_sd)^-1 D (I-Q_fd)^(-1)
#    lhs = lhs.at[:2**(n_met-1)].set(0.)
#    lhs = mhn.R_inv_vec(log_theta_fd, lhs, state_met, True)
#    dTh_2, d_fd = mhn.x_partial_Q_y(log_theta_fd, lhs, R_1_inv_p_0, state_met) 
#    return score, dTh_1 + dTh_2, d_fd, d_sd

 
def _grad_prim_obs(log_theta_prim: jnp.array, fd_effects: jnp.array, 
                   state_prim: jnp.array, n_prim: int) -> tuple[jnp.array, jnp.array, jnp.array]:
    """ Gradient of lp_prim_obs for a single sample

    Args:
        log_theta_prim_sd (jnp.array):  Log values of the theta matrix.
        state_prim (jnp.array):         Binary state vector, representing the current sample's events.
        n_prim (int):                   Number of active events in the PT.
    Returns:
        tuple:                          Marginal Likelihood, gradient wrt. theta, gradient wrt. diagnosis effects.
    """
    p0 = jnp.zeros(2**n_prim)
    p0 = p0.at[0].set(1.)
    log_theta_fd = diagnosis_theta(log_theta_prim, fd_effects)
    p_theta= mhn.R_inv_vec(log_theta_fd, p0, state_prim)
    score = jnp.log(p_theta.at[-1].get())
    
    lhs = jnp.zeros_like(p0)
    lhs = lhs.at[-1].set(1/p_theta.at[-1].get())
    lhs = mhn.R_inv_vec(log_theta_fd, lhs, state_prim, transpose = True)
    dth, d_diag_effects = mhn.x_partial_Q_y(log_theta_fd, lhs, y=p_theta, state=state_prim)
    dth = dth.at[:-1, -1].set(0.0)  # Derivative of constant is 0.
    return score, dth, d_diag_effects 


def _g_coupled(log_theta: jnp.array, fd_effects: jnp.array, sd_effects: jnp.array, 
               state_joint: jnp.array, n_prim: int, n_met: int) -> tuple[jnp.array, jnp.array, jnp.array, jnp.array]:
    """Calculates the likelihood and gradients for a coupled datapoint (PT and MT genotype known)

    Args:
        log_theta_fd (jnp.array):       Theta matrix with logarithmic entries, scaled by effects on first diagnosis.
        log_theta_fd_prim (jnp.array):  Copy of log_theta_fd, with off diagonal entries of the last column set to 0.
        log_theta_sd (jnp.array):       Theta matrix with logarithmic entries, scaled by effects on second diagnosis.
        state_joint (jnp.array):        Bitstring, genotypes of PT and MT of the same patient.
        d_e (jnp.array):                Effect of seeding on the diagnosis. 
        n_prim (int):                   Number of active events in the PT.
        n_met (int):                    Number of active events in the MT.

    Returns:
        tuple: score, grad wrt. to theta, grad wrt. fd_effects, grad wrt. sd_effects
    """
    met = jnp.append(state_joint.at[1::2].get(), 1)
    p = jnp.zeros(2**(n_prim + n_met - 1))
    p = p.at[0].set(1.)
    
    # Joint and met-marginal distribution at first sampling
    pTh1_joint = R_i_inv_vec(log_theta, fd_effects, p, state_joint, 
                             n_prim+n_met-1, transpose = False)
    
    # Select the states where x = prim and z are compatible with met
    # Reuse the memory allocated for p 
    p = obs_states((n_prim + n_met - 1), state_joint, True)
    poss_states_inds = jnp.where(p==1., size=2**(n_met-1))[0]
    
    # Prim conditional distribution at first sampling, to be used as starting dist for second sampling
    pTh1_cond_obs = pTh1_joint.at[poss_states_inds].get()
    # States where the Seeding didn't happen aren't compatible with met and get probability 0
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_met-1)), pTh1_cond_obs)
    
    # Derivative of pTh2 = M(I-Q_sd)^(-1)pth1_cond
    log_theta_sd = diagnosis_theta(log_theta, sd_effects)
    g_1, d_sd, pTh2_marg = mhn.gradient(log_theta_sd, met, pTh1_cond_obs)
    # q = (pD/pTh_2)^T M(I-Q_sd)^(-1)
    q = jnp.zeros(2**n_met)
    q = q.at[-1].set(1/pTh2_marg.at[-1].get())
    q = mhn.R_inv_vec(log_theta_sd, q, met, transpose = True)
    
    # Reuse the memory allocated for p0
    p = p * 0. 
    p = p.at[poss_states_inds].set(q.at[2**(n_met - 1):].get())
    
    # Derivative of pth1_cond
    p = R_i_inv_vec(log_theta, fd_effects, p, state_joint, 
                    n_prim + n_met-1, transpose = True)
    g_2, d_fd = x_partial_Q_y(log_theta, fd_effects, p, pTh1_joint, state_joint)
    score = jnp.log(pTh2_marg.at[-1].get())
    return score, g_1 + g_2, d_fd, d_sd