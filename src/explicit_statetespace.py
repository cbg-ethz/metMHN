import numpy as np


# Some helper functions
def dia2(tij: float) -> np.array:
    """
    Args:
        tij (float): ij-th entry of theta
    Returns:
         np.array: non diagonal kronecker factor
    """
    ret = np.diag([-tij, 0])
    ret[1, 0] = tij
    return ret


def multi_kron_prods(diag_fac, off_diag_fac: np.array,
                     last: np.array, exp_theta: np.array, i: int, n: int,
                     deriv_fac: np.array = None, k: int = None) -> np.array:
    """
    Calculates a series of Kronecker products
    Args:
        diag_fac: function, that builds diagonal Kroneckerfactors
        off_diag_fac (np.array): Non diagonal Kroneckerfactor
        last (np.array): Last kronecker factor to be multiplied
        exp_theta (np.array): Exponential theta matrix
        i (int): index of the current summand
        n (int): Total number of mutations
        deriv_fac (function):  if not None: Kroneckerfactor for derivative of Q wrt theta_ik
        k (int): if not None: column index of derivative wrt theta
    Returns:
         np.array: Q explicit matrix
    """
    Q = np.eye(1)
    for j in range(i):
        if k is None:
            Q = np.kron(diag_fac(exp_theta[i, j]), Q)
        elif k != j:
            Q = np.kron(diag_fac(exp_theta[i, j]), Q)
        else:
            Q = np.kron(deriv_fac(exp_theta[i, j]), Q)

    Q = np.kron(off_diag_fac, Q)
    for j in range(i+1, n):
        if k is None:
            Q = np.kron(diag_fac(exp_theta[i, j]), Q)
        elif k != j:
            Q = np.kron(diag_fac(exp_theta[i, j]), Q)
        else:
            Q = np.kron(deriv_fac(exp_theta[i, j]), Q)

    return np.kron(last, Q)


def diag_sync(tij: float) -> np.array:
    """
    Explicitely builds diagonal Kronecker factors for Q_sync
    Args:
        tij (float): ij-th entry of theta
    Returns:
        np.array, 4x4 dimensional diagonal Kroneckerfactor
    """
    return np.diag([1, 0, 0, tij])


def diag_met(tij: float) -> np.array:
    """
    Explicitely builds diagonal Kronecker factors for Q_met
    Args:
        tij (float): ij-th entry of theta
    Returns:
        np.array, 4x4 dimensional diagonal Kroneckerfactor
    """
    return np.diag([1, 1, tij, tij])


def diag_prim(tij: float) -> np.array:
    """
    Explicitly builds diagonal Kronecker factors for Q_prim
    Args:
        tij (float): ij-th entry of theta
    Returns:
        np.array, 4x4 dimensional diagonal Kroneckerfactor
    """
    return np.diag([1, tij, 1, tij])


def build_q(log_theta: np.array, deriv_inds = None) -> np.array:
    """
    Explicitly builds Q matrix or \partial Q \partial theta_deriv_inds
    Args:
        log_theta (np.array): logarithmic theta matrix
        deriv_inds (tuple): Either None if Q should be built or tuple of indices
    Returns:
        np.array: Q or \partial Q \partial theta_ij, explicit Q_matrix
    """
    n = log_theta.shape[0]-1
    theta = np.exp(log_theta)
    Q = np.zeros([2**(2*n+1), 2**(2*n+1)])
    for i in range(n):
        sync_off = np.diag([-theta[i, i], 0., 0., 0.])
        sync_off[3, 0] = theta[i, i]
        sync = multi_kron_prods(diag_sync, sync_off,
                                np.diag([1., 0.]), theta, i, n)
        prim_off = np.diag([-1., 0., -1., 0.])
        prim_off[1, 0] = 1.
        prim_off[3, 2] = 1.
        prim = multi_kron_prods(diag_prim, prim_off *
                                theta[i, i], np.diag([0, 1]), theta, i, n)
        met_off = np.diag([-1., -1., 0., 0.])
        met_off[2, 0] = 1.
        met_off[3, 1] = 1.
        met = multi_kron_prods(
            diag_met, met_off*theta[i, i], np.diag([0., theta[i, n]]), theta, i, n)
        Q = Q + sync + prim + met
    off_seed = np.diag([-theta[n, n], 0.])
    off_seed[1, 0] = theta[n, n]
    Q = Q + multi_kron_prods(diag_sync, off_seed, 1.0*np.eye(1), theta, n, n)
    return Q


def diag_sync_deriv(tij: float) -> np.array:
    """
    Explicitely builds diagonal Kronecker factors for Q_sync
    Args:
        tij (float): ij-th entry of theta
    Returns:
        np.array, 4x4 dimensional diagonal Kroneckerfactor
    """
    return np.diag([0, 0, 0, tij])


def diag_met_deriv(tij: float) -> np.array:
    """
    Explicitely builds diagonal Kronecker factors for Q_met
    Args:
        tij (float): ij-th entry of theta
    Returns:
        np.array, 4x4 dimensional diagonal Kroneckerfactor
    """
    return np.diag([0, 0, tij, tij])


def diag_prim_deriv(tij: float) -> np.array:
    """
    Explicitly builds diagonal Kronecker factors for Q_prim
    Args:
        tij (float): ij-th entry of theta
    Returns:
        np.array, 4x4 dimensional diagonal Kroneckerfactor
    """
    return np.diag([0, tij, 0, tij])


def build_q_deriv_ik(log_theta: np.array, i: int, k: int ) -> np.array:
    """
    Explicitly builds \partial Q \partial log(theta_kl)
    Args:
        log_theta (np.array): logarithmic theta matrix
        k (int): row index of theta to take derivative
        l (int): column index of theta to take derivative
    Returns:
        np.array: \partial Q \partial log(theta_kl)
    """
    n = log_theta.shape[0]-1
    theta = np.exp(log_theta)
    Q = np.zeros([2**(2*n+1), 2**(2*n+1)])

    sync_off = np.diag([-theta[i, i], 0., 0., 0.])
    sync_off[3, 0] = theta[i, i]
    sync = multi_kron_prods(diag_sync, sync_off, np.diag([1., 0.]), theta, i, n, diag_sync_deriv, k)

    prim_off = np.diag([-1., 0., -1., 0.])
    prim_off[1, 0] = 1.
    prim_off[3, 2] = 1.
    prim = multi_kron_prods(diag_prim, prim_off*theta[i, i], np.diag([0, 1]), theta, i, n, diag_prim_deriv, k)

    met_off = np.diag([-1., -1., 0., 0.])
    met_off[2, 0] = 1.
    met_off[3, 1] = 1.
    met = multi_kron_prods(diag_met, met_off*theta[i, i], np.diag([0., theta[i, n]]),
                           theta, i, n, diag_met_deriv, k)

    if i < n and k < n:
        Q = sync + prim + met
    elif i < n and k == n:
        Q = met
    else:
        off_seed = np.diag([-theta[n, n], 0.])
        off_seed[1, 0] = theta[n, n]
        Q = multi_kron_prods(diag_sync, off_seed, 1.0*np.eye(1), theta, n, n, diag_sync_deriv, k)
    return Q


def build_q_grad_p(theta: np.array, q: np.array, p: np.array) -> np.array:
    """
    Calculates q \partial Q \partial log(theta_ij) p for all i,j explicitly
    Args:
        theta (np.array): logarithmic theta matrix
        q (np.array): vector to multiply from the left to dQ/d theta
        p (np.array): vector to multiply from the right to dQ/d theta

    returns:
        np.array: q d Q/d Theta p
    """
    n = theta.shape[0] - 1
    g = np.zeros_like(theta)
    for i in range(n+1):
        for k in range(n+1):
            g[i, k] = q @ build_q_deriv_ik(theta, i, k) @ p
    return g


# Methods for explicit calculation of the SSR version of Q
def sync_ssr_q(mut: list, theta: np.array, i: int, n: int) -> np.array:
    """
    Builds explicit truncated Q_matrix for synchroneous transitions
    Args:
        mut (list): list  of length n, entry at index k: 0 (no mutation k), 1 (k only present in primary t.),
                    2 (k only present in met), 3 (k present in both)
        theta (np.array): Logarithmic theta matrix
        i (int): Index of current summand
        n (int): Total number of mutations
    Returns:
         np.array: Q_sync_ssr
    """
    syncL = np.diag([-1, 0, 0, 0])
    syncL[3, 0] = 1
    diag10 = np.diag([1, 0])
    Q = np.diag([1])
    # Diagonal Kronecker factors
    for j in range(i):
        if mut[j] == 3:
            Q = np.kron(np.diag([1, 0, 0, theta[i, j]]), Q)
        elif mut[j] == 1 or mut[j] == 2:
            Q = np.kron(diag10, Q)
    # Non diagonal Kronecker factors
    if mut[i] == 3:
        Q = np.kron(syncL * theta[i, i], Q)
    elif mut[i] == 1 or mut[i] == 2:
        Q = np.kron(diag10 * (-theta[i, i]), Q)
    else:
        Q = Q * (-theta[i, i])
    # Diagonal Kronecker factors
    for j in range(i + 1, n):
        if mut[j] == 3:
            Q = np.kron(np.diag([1, 0, 0, theta[i, j]]), Q)
        elif mut[j] == 1 or mut[j] == 2:
            Q = np.kron(diag10, Q)
    if mut[-1] == 1:
        Q = np.kron(diag10, Q)
    return Q


def met_ssr_q(mut: list, theta: np.array, i: int, n: int) -> np.array:
    """
    Builds explicit truncated Q_matrix for asynchroneous metastasis transitions
    Args:
        mut (list): list  of length n, entry at index k: 0 (no mutation k), 1 (k only present in primary t.),
                    2 (k only present in met), 3 (k present in both)
        theta (np.array): Logarithmic theta matrix
        i (int): Index of current summand
        n (int): Total number of mutations
    Returns:
         np.array: Q_met_ssr
    """
    if mut[-1] == 0:
        return np.zeros((2 ** (np.count_nonzero(mut > 0))) * (2 ** np.count_nonzero(mut == 3)))
    metL = np.diag([-1, -1, 0, 0])
    metL[2, 0] = 1
    metL[3, 1] = 1
    diag11 = np.eye(2)
    diag_slice = np.diag([-theta[i, i], 0])
    diag_slice[1, 0] = theta[i, i]
    Q = np.diag([1])
    # Diagonal Kronecker factors
    for j in range(i):
        if mut[j] == 1:
            Q = np.kron(diag11, Q)  # ...|10|...
        elif mut[j] == 2:
            Q = np.kron(np.diag([1, theta[i, j]]), Q)  # ...|01|...
        elif mut[j] == 3:
            # ...|11|...
            Q = np.kron(np.diag([1, 1, theta[i, j], theta[i, j]]), Q)
    # Non diagonal Kronecker factors
    if mut[i] == 0:
        Q = Q * (-theta[i, i])  # ...|00|...
    elif mut[i] == 1:
        Q = np.kron(diag11 * (-theta[i, i]), Q)
    elif mut[i] == 2:
        Q = np.kron(diag_slice, Q)
    elif mut[i] == 3:
        Q = np.kron(metL * theta[i, i], Q)
    # Diagonal Kronecker factors
    for j in range(i + 1, n):
        if mut[j] == 1:
            Q = np.kron(diag11, Q)
        elif mut[j] == 2:
            Q = np.kron(np.diag([1, theta[i, j]]), Q)
        elif mut[j] == 3:
            Q = np.kron(np.diag([1, 1, theta[i, j], theta[i, j]]), Q)
    Q = np.kron(np.diag([0, theta[i, n]]), Q)
    return Q

# Builds Q_i for the primary part of Q after seeding


def prim_ssr_q(mut: list, theta: np.array, i: int, n: int) -> np.array:
    """
    Builds explicit truncated Q_matrix for asynchroneous primary transitions
    Args:
        mut (list): list  of length n, entry at index k: 0 (no mutation k), 1 (k only present in primary t.),
                    2 (k only present in met), 3 (k present in both)
        theta (np.array): Logarithmic theta matrix
        i (int): Index of current summand
        n (int): Total number of mutations
    Returns:
         np.array: Q_prim_ssr
    """
    if mut[-1] == 0:
        return np.zeros((2 ** (np.count_nonzero(mut > 0))) * (2 ** np.count_nonzero(mut == 3)))
    primL = np.diag([-1, 0, -1, 0])
    primL[1, 0] = 1
    primL[3, 2] = 1
    diag_slice = np.diag([-theta[i, i], 0])
    diag_slice[1, 0] = theta[i, i]
    diag11 = np.eye(2)
    Q = np.diag([1])
    for j in range(i):
        if mut[j] == 1:
            Q = np.kron(np.diag([1, theta[i, j]]), Q)
        elif mut[j] == 2:
            Q = np.kron(diag11, Q)
        elif mut[j] == 3:
            Q = np.kron(np.diag([1, theta[i, j], 1, theta[i, j]]), Q)
    if mut[i] == 0:
        Q = Q * (-theta[i, i])
    elif mut[i] == 1:
        Q = np.kron(diag_slice, Q)
    elif mut[i] == 2:
        Q = np.kron(diag11 * (-theta[i, i]), Q)
    elif mut[i] == 3:
        Q = np.kron(primL * theta[i, i], Q)
    # Case j > i
    for j in range(i + 1, n):
        if mut[j] == 1:
            Q = np.kron(np.diag([1, theta[i, j]]), Q)
        elif mut[j] == 2:
            Q = np.kron(diag11, Q)
        elif mut[j] == 3:
            Q = np.kron(np.diag([1, theta[i, j], 1, theta[i, j]]), Q)
    Q = np.kron(np.diag([0, 1]), Q)
    return Q

# Adds the entries of the seeding event


def seeding_ssr_q(mut: list, theta: np.array, n: int) -> np.array:
    """
    Builds explicit truncated Q_matrix for synchroneous transitions
    Args:
        mut (list): list  of length n, entry at index k: 0 (no mutation k), 1 (k only present in primary t.),
                    2 (k only present in met), 3 (k present in both)
        theta (np.array): Logarithmic theta matrix
        n (int): Total number of mutations
    Returns:
         np.array: Q_sync_ssr
    """
    diag10 = np.diag([1, 0])
    Q = np.diag([1])
    for j in range(n):
        if mut[j] == 3:
            Q = np.kron(np.diag([1, 0, 0, theta[n, j]]), Q)
        elif mut[j] > 0:
            Q = np.kron(diag10, Q)
    if mut[-1] == 1:
        Q = np.kron(dia2(theta[n, n]), Q)
    else:
        Q = Q * (-theta[n, n])
    return Q


def ssr_build_q(dpoint: list, log_theta: np.array) -> np.array:
    """
    Builds explicit truncated Q_matrix for synchroneous transitions
    Args:
        dpoint (list): Bitstring, genotype(s) of tumors(s) of a single patient
        log_theta (np.array): Logarithmic theta matrix
    Returns:
         np.array: Q_sync_ssr
    """
    n = log_theta.shape[0] - 1
    theta = np.exp(log_theta)
    mut = [dpoint[j] + 2 * dpoint[j + 1] for j in range(0, 2 * n, 2)]
    mut.append(dpoint[-1])
    mut = np.array(mut)
    Q = np.zeros(2 ** (sum(dpoint)))
    for i in range(n):
        Q = Q + sync_ssr_q(mut, theta, i, n) + met_ssr_q(mut, theta, i, n) \
            + prim_ssr_q(mut, theta, i, n)
    return Q + seeding_ssr_q(mut, theta, n)
