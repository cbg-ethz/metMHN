from metmhn.jx.kronvec import (
                        diagnosis_theta,
                        kronvec_sync, 
                        kronvec_met, 
                        kronvec_prim, 
                        kronvec_seed, 
                        kronvec,
                        xdQmetddiy, 
                        kron_diag, 
                        obs_states
                        )

from metmhn.jx import vanilla as mhn
import jax.numpy as jnp
from jax import jit, lax
from functools import partial



def f1(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = p[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z, 0.


def f2(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = m[:, 1].sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, 0., z


def f3(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s[:, 3].sum() + p[:, [1, 3]].sum()
    z_m = m[:, [2, 3]].sum()
    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z, z_m


def t12(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float, float]:
    s = s.reshape((-1, 2), order="C")
    p = p.reshape((-1, 2), order="C")
    m = m.reshape((-1, 2), order="C")

    z = s[:, 0].sum() + p.sum()
    z_m = m.sum()
    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z, z_m


def t3(s: jnp.array, p: jnp.array, m: jnp.array) -> tuple[jnp.array, jnp.array, jnp.array, float, float]:
    s = s.reshape((-1, 4), order="C")
    p = p.reshape((-1, 4), order="C")
    m = m.reshape((-1, 4), order="C")

    z = s.sum() + p.sum()
    z_m =  m.sum()

    s = s.flatten(order="F")
    p = p.flatten(order="F")
    m = m.flatten(order="F")

    return s, p, m, z, z_m


def z3(s: jnp.array) -> tuple[jnp.array, float]:
    s = s.reshape((-1, 4), order="C")

    z = s[:, 3].sum()

    s = s.flatten(order="F")

    return s, z


def deriv_no_seed(i:int, g_row_i: jnp.array, g_m_row_i: jnp.array, x: jnp.array, y:jnp.array, 
                  log_theta: jnp.array,  diag_effects: jnp.array, state:jnp.array, n:int) -> tuple[jnp.array, jnp.array]:

    d_e = diag_effects.at[-1].get()
    diag_th = diagnosis_theta(log_theta, diag_effects)
    z_sync = jnp.multiply(x, kronvec_sync(diag_log_theta=diag_th,
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
def R_i_inv_vec(log_theta: jnp.array, diag_effects: jnp.array, x: jnp.array, state: jnp.array, 
                state_size: int, transpose: bool = False) -> jnp.array:
    """This computes R_i^{-1} x = (\lambda_i I - Q)^{-1} x

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
    lidg = -1. / (kron_diag(log_theta=log_theta, diag_effects = diag_effects, 
                           state= state, n_state=state_size) - 1.)
    y = lidg * x
    y = lax.fori_loop(
        lower=0,
        upper=state_size+1,
        body_fun=lambda _, val: lidg * (kronvec(log_theta=log_theta, diag_effects=diag_effects, p=val,
                                                state=state, diag=False, transpose=transpose) + x),
        init_val=y
    )

    return y


def _lp_coupled(log_theta: jnp.array, fd_effects:jnp.array, sd_effects: jnp.array, 
                state_joint: jnp.array, n_prim: int, n_met: int) -> jnp.array:
    """
    Evaluates the log probability of seeing coupled genotype data in state "state"
    Args:
        log_theta_fd (jnp.array):       Theta_matrix with logarithmics entries, scaled by effects on second diagnosis.
        log_theta_sd (jnp.array):       Theta_matrix with logarithmic entries, scaled by effects on first diagnosis.
        log_theta_prim (jnp.array):     Copy of log_theta_fd,but  with off-diagonal entries in the 
                                        last column set to 0.
        state_joint (jnp.array):        Bitstring, genotypes of PT and MT. 
        d_e (jnp.array):                Effect of the seeding on diagnosis. 
        n_prim (int):                   Number of active events in PT.
        n_met (int):                    Number of active events in MT.
    Returns:
        jnp.array: log(P(state))
    """
    p = jnp.zeros(2**(n_prim + n_met - 1))
    p = p.at[0].set(1.)
    pTh1_joint = R_i_inv_vec(log_theta, fd_effects, p, state_joint, 
                             n_prim+n_met-1, transpose = False)
   
    # Select the states where x = prim and z are compatible with met 
    p = obs_states((n_prim + n_met - 1), state_joint, True)
    poss_states_inds = jnp.where(p == 1., size=2**(n_met-1))[0]
    # Prim conditional distribution at first sampling, to be used as starting dist for second sampling
    pTh1_cond_obs = pTh1_joint.at[poss_states_inds].get()
    # States where the Seeding didn't happen aren't compatible with met and get probability 0
    pTh1_cond_obs = jnp.append(jnp.zeros(2**(n_met-1)), pTh1_cond_obs)

    met = jnp.append(state_joint.at[1::2].get(), 1)
    log_theta_sd = diagnosis_theta(log_theta, sd_effects)
    pTh2 = mhn.R_inv_vec(log_theta_sd, pTh1_cond_obs, met)
    return jnp.log(pTh2.at[-1].get())


def _lp_prim_obs(log_theta_prim: jnp.array, fd_effects: jnp.array,
                 state_prim: jnp.array, n_prim: int) -> jnp.array:
    """Calculates the marginal likelihood of only observing a PT at first sampling with genotype state_prim
    
    Args:
        log_theta_prim_fd (jnp.array):      Theta matrix for diagnosis formalism with logarithmic entries. 
                                            The off-diagonal entries of the last column are set to 0.
        state_prim (jnp.array):             Bitstring of length 2*n+1, observed genotype of tumor(pair)
        n_prim (int):                       Number of active events in the PT.
        
    Returns:
        jnp.array: log(P(state_prim; theta))
    """
    log_theta_fd = diagnosis_theta(log_theta_prim, fd_effects)
    p0 = jnp.zeros(2**n_prim)
    p0 = p0.at[0].set(1.)
    pTh = mhn.R_inv_vec(log_theta_fd, p0, state_prim, False)
    return jnp.log(pTh.at[-1].get())


#def _lp_met_obs(log_theta_fd: jnp.array, log_theta_sd: jnp.array, 
#                state_met: jnp.array, n_met: int) -> jnp.array:
#    """Calculates the marginal likelihood of only observing a MT at second sampling with genotype state_met
#    
#    Args:
#        log_theta_fd (jnp.array):   Theta matrix with logarithmic entries, scaled by fd_effects.
#        log_theta_sd (jnp.array):   Theta matrix with logarithmic entries, scaled by sd_effects. 
#        state_met (jnp.array):      Bitstring, genotype of MT.
#        n_met (int):                Number of active events in the MT.
#
#    Returns:
#        jnp.array: log(P(state_met; theta))
#    """
#    p0 = jnp.zeros(2**n_met)
#    p0 = p0.at[0].set(1.)
#    pTh1 = mhn.R_inv_vec(log_theta_fd, p0, state_met, False)
#    pTh1 = pTh1.at[:2**(n_met-1)].set(0.)
#    pTh2 = mhn.R_inv_vec(log_theta_sd, pTh1, state_met, False)
#    return jnp.log(pTh2.at[-1].get())


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