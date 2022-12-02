import numpy as np
from explicit_statetespace import *


def kronvec_sync(theta: np.array, p: np.array, i: int, n: int, diag: bool = True, transp: bool = False) -> np.array:
    """
    This function computes the synchroneous part of Q*p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
          transp (bool): indicates wether to transpose Qp or not

    Returns:
          np.array: Q_sync p
    """
    theta_i = np.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, (1, 2)] = 0
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, (1, 2)] = 0
    if not transp:
        p[:, 3] = theta_i[i] * p[:, 0]
        if diag:
            p[:, 0] = -p[:, 3]
        else:
            p[:, 0] = 0
    else:
        if diag:
            p[:, 0] = (p[:, 3]-p[:, 0])*theta_i[i]
            p[:, 3] = 0
        else:
            p[:, 0] = p[:, 3]*theta_i[i]
            p[:, 3] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, (1, 2)] = 0
        p[:, 3] = p[:, 3]*theta_i[j]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def kronvec_met(theta: np.array, p: np.array, i: int, n: int, diag: bool = True, transp: bool = False) -> np.array:
    """
    This function computes asynchroneous part of metastatic transitions of Q*p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
          transp (bool): indicates wether to transpose Qp or not
    Returns:
          np.array: Q_met p
    """
    theta_i = np.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta_i[j]*p[:, 2]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    if not transp:
        p[:, 2] = theta_i[i] * p[:, 0]
        p[:, 3] = theta_i[i] * p[:, 1]
        if diag:
            p[:, 0] = -p[:, 2]
            p[:, 1] = -p[:, 3]
        else:
            p[:, 0] = 0
            p[:, 1] = 0
    else:
        if diag:
            p[:, 0] = (p[:, 2] - p[:, 0])*theta_i[i]
            p[:, 1] = (p[:, 3] - p[:, 1])*theta_i[i]
            p[:, (2, 3)] = 0
        else:
            p[:, 0] = p[:, 2] * theta_i[i]
            p[:, 1] = p[:, 3] * theta_i[i]
            p[:, (2, 3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta_i[j]*p[:, 2]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p[:, 1] = theta_i[n]*p[:, 1]
    p = p.flatten(order="F")
    return p


def kronvec_prim(theta: np.array, p: np.array, i: int, n: int, diag: bool = True, transp: bool = False) -> np.array:
    """
    This function computes asynchroneous part of primary transitions of Q*p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
          transp (bool): indicates wether to transpose Qp or not
    Returns:
          np.array: Q_prim p
    """
    theta_i = np.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 1] = theta_i[j]*p[:, 1]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    if not transp:
        p[:, 1] = theta_i[i] * p[:, 0]
        p[:, 3] = theta_i[i] * p[:, 2]
        if diag:
            p[:, 0] = -p[:, 1]
            p[:, 2] = -p[:, 3]
        else:
            p[:, (0, 2)] = 0
    else:
        if diag:
            p[:, 0] = (p[:, 1] - p[:, 0])*theta_i[i]
            p[:, 2] = (p[:, 3] - p[:, 2])*theta_i[i]
            p[:, (1, 3)] = 0
        else:
            p[:, 0] = p[:, 1]*theta_i[i]
            p[:, 2] = p[:, 3]*theta_i[i]
            p[:, (1, 3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 1] = theta_i[j]*p[:, 1]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p = p.flatten(order="F")
    return p


def kronvec_seed(theta: np.array, p: np.array, n: int, diag: bool = True, transp: bool = False) -> np.array:
    """
    This function computes the seeding part of metastatic transitions of Q*p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          n (int): number of genomic events
          diag (bool): Flag indicating wether to build the diagonal
          transp (bool): indicates wether to transpose Qp or not
    Returns:
          np.array: Q_seed p
    """
    theta_n = np.exp(theta[n, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, (1, 2)] = 0
        p[:, 3] = theta_n[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    if not transp:
        p[:, 1] = theta_n[n] * p[:, 0]
        if diag:
            p[:, 0] = -p[:, 1]
        else:
            p[:, 0] = 0
    else:
        if diag:
            p[:, 0] = (p[:, 1] - p[:, 0])*theta_n[n]
            p[:, 1] = 0
        else:
            p[:, 0] = p[:, 1]*theta_n[n]
            p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def qvec(theta: np.array, p: np.array, diag: bool, transp: bool = False) -> np.array:
    """
    This function computes Q*p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          diag (bool): Flag indicating whether to build the diagonal
          transp (bool):  Flag indicating whether to transpose Q
    Returns:
          np.array: Q p
    """
    n = theta.shape[0] - 1
    tmp = 1.*np.zeros(p.shape[0])
    for i in range(n):
        tmp += kronvec_sync(theta, p.copy(), i, n, diag=diag, transp=transp) +\
              kronvec_prim(theta, p.copy(), i, n, diag=diag, transp=transp) +\
              kronvec_met(theta, p.copy(), i, n, diag=diag, transp=transp)
    tmp += kronvec_seed(theta, p, n, diag=diag, transp=transp)
    return tmp


def kronvec_sync_diag(theta: np.array, p: np.array, i: int, n: int) -> np.array:
    """
    This function computes synchroneous part of primary transitions of diag(Q) p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events

    Returns:
          np.array: diag(Q_sync) p
    """
    theta_i = np.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, (1, 2)] = 0
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 0] = theta_i[i]*p[:, 0]
    p[:, (1, 2, 3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, (1, 2)] = 0
        p[:, 3] = p[:, 3]*theta_i[j]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2**(2 * n), 2), order="C")
    p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def kronvec_met_diag(theta: np.array, p: np.array, i: int, n: int) -> np.array:
    """
    This function computes diag(Q_met) p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events

    Returns:
          np.array: diag(Q_met) p
    """
    theta_i = np.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta_i[j]*p[:, 2]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 0] = theta_i[i]*p[:, 0]
    p[:, 1] = theta_i[i]*p[:, 1]
    p[:, (2, 3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta_i[j]*p[:, 2]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2**(2*n), 2), order="C")
    p[:, 0] = 0
    p[:, 1] = theta_i[n]*p[:, 1]
    p = p.flatten(order="F")
    return p


def kronvec_prim_diag(theta: np.array, p: np.array, i: int, n: int) -> np.array:
    """
    This function computes diag(Q_prim) p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          i (int): index of the current summand
          n (int): number of genomic events

    Returns:
          np.array: diag(Q_prim) p
    """
    theta_i = np.exp(theta[i, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 1] = theta_i[j]*p[:, 1]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 0] = theta_i[i]*p[:, 0]
    p[:, 2] = theta_i[i]*p[:, 2]
    p[:, (1, 3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 1] = theta_i[j]*p[:, 1]
        p[:, 3] = theta_i[j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p = p.flatten(order="F")
    return p


def kronvec_seed_diag(theta: np.array, p: np.array, n: int) -> np.array:
    """
    This function computes diag(Q_seed) p implicitly
    Args:
          theta (np.array): Logarithmic theta matrix
          p (np.array): state vector
          n (int): number of genomic events

    Returns:
          np.array: diag(Q_seed) p
    """
    theta_n = np.exp(theta[n, :])
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, (1, 2)] = 0
        p[:, 3] = theta_n[j]*p[:, 3]
        p = p.flatten(order="F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2**(2*n), 2), order="C")
    p[:, 0] = theta_n[n]*p[:, 0]
    p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def diag_q(theta: np.array) -> np.array:
    """
    This function computes diag(Q) implicitly
    Args:
          theta (np.array): Logarithmic theta matrix

    Returns:
          np.array: diag(Q)
    """
    n = theta.shape[0] - 1
    p = np.ones(2 ** (2 * n + 1))
    tmp = 1.*np.zeros(p.shape[0])
    for i in range(n):
        tmp = tmp + kronvec_met_diag(theta, p.copy(), i, n) + \
              kronvec_prim_diag(theta, p.copy(), i, n) +\
              kronvec_sync_diag(theta, p.copy(), i, n)
    return tmp + kronvec_seed_diag(theta, p, n)


def diag_diagnosis(n: int) -> np.array:
    """
    This function returns the modified diagonal of Q in the diagnosisMHN formalism
    Args:
        n (int): number of mutations
    Returns:
         np.array: p diagonal of Q_jump
    """
    p = np.ones(2**(2*n+1))
    # Diagonal 4x4 Kronecker factors j<i

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2 * n-1), 4), order="C")
    p[:, (0, 1, 2)] = 0
    p = p.flatten(order="F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p = p.flatten(order="F")
    return p


def q_partialQ_pth(theta: np.array, q: np.array, pTh: np.array, n: int) -> np.array:
    """
    calculates q \partial Q \partial theta_ij p for all i,j
    Args:
        theta (np.array): Logarithmic theta matrix
        q (np.array): vector multiplied from the left to partial Q partial theta
        pTh (np.array): vector multiplied from the right to partial Q partial theta
        n (int): number of mutations

    Returns:
        np.array: g
    """
    g = np.zeros_like(theta)
    for i in range(n):
        z_sync = q * kronvec_sync(theta, pTh.copy(), i, n)
        z_prim = q * kronvec_prim(theta, pTh.copy(), i, n)
        z_met = q * kronvec_met(theta, pTh.copy(), i, n)
        for j in range(n):
            z_sync = z_sync.reshape((2**(2*n-1), 4), order="C")
            z_prim = z_prim.reshape((2**(2*n-1), 4), order="C")
            z_met = z_met.reshape((2**(2*n-1), 4), order="C")

            g[i, j] = np.sum(z_sync[:, 3]) +\
                      np.sum(z_prim[:, (1, 3)]) +\
                      np.sum(z_met[:, (2, 3)])
            if i == j:
                g[i, j] = np.sum(z_sync) + np.sum(z_prim) + np.sum(z_met)#np.sum(z_sync[:, 0]) + np.sum(z_prim[:, (0, 2)]) + np.sum(z_met[:, (0, 1)])

            z_sync = z_sync.flatten(order="F")
            z_prim = z_prim.flatten(order="F")
            z_met = z_met.flatten(order="F")
        g[i, n] = np.sum(z_met)

    z_seed = q * kronvec_seed(theta, pTh, n)
    for j in range(n):
        z_seed = z_seed.reshape((2**(2*n-1), 4), order="C")
        g[n, j] = np.sum(z_seed[:, 3])
        z_seed = z_seed.flatten(order="F")

    g[n, n] = np.sum(z_seed)
    return g
