import numpy as np
from explicit_statetespace import *


def kronvec_sync(theta, p, i, n, diag=True):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,(1, 2)] = 0
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 3] = theta[i,i]*p[:, 0]
    p[:,(0,1,2)] = 0
    if diag:
        p[:, 0] = -p[:, 3]
    else:
        p[:, 0] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,(1, 2)] = 0
        p[:, 3] = p[:, 3]*theta[i, j]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def kronvec_met(theta, p, i, n, diag=True):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta[i, j]*p[:, 2]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 2] = theta[i,i]*p[:, 0]
    p[:, 3] = theta[i, i] * p[:, 1]
    if diag:
        p[:, 0] = -p[:, 2]
        p[:, 1] = -p[:, 3]
    else:
        p[:, (0,1)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta[i, j]*p[:, 2]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p[:, 1] = theta[i, n]*p[:, 1]
    p = p.flatten(order="F")
    return p


def kronvec_prim(theta, p, i, n, diag=True):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,1] = theta[i,j] * p[:, 1]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 1] = theta[i,i]*p[:, 0]
    p[:, 3] = theta[i,i]*p[:, 2]
    if diag:
        p[:, 0] = -p[:, 1]
        p[:, 2] = -p[:, 3]
    else:
        p[:, (0,2)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,1] = theta[i,j] * p[:, 1]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p = p.flatten(order="F")
    return p


def kronvec_seed(theta, p, n, diag=True):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,(1, 2)] = 0
        p[:, 3] = theta[n, j]*p[:, 3]
        p = p.flatten(order="F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 1] = theta[n,n]*p[:,0]
    if diag:
        p[:, 0] = -p[:,1]
    else:
        p[:, 0] = 0
    p = p.flatten(order="F")
    return p


def qvec(theta, p, diag):
    n = theta.shape[0] - 1
    tmp = 1.*np.zeros(p.shape[0])
    for i in range(n):
        tmp = tmp + kronvec_sync(theta, p.copy(), i, n, diag=diag) +\
              kronvec_prim(theta, p.copy(), i, n, diag=diag) + \
              kronvec_met(theta, p.copy(), i, n, diag=diag)
    tmp = tmp + kronvec_seed(theta, p, n, diag=diag)
    return tmp


def kronvec_sync_diag(theta, p, i, n):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,(1, 2)] = 0
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 0] = theta[i,i]*p[:, 0]
    p[:, (1,2,3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,(1, 2)] = 0
        p[:, 3] = p[:, 3]*theta[i, j]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def kronvec_met_diag(theta, p, i, n):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta[i, j]*p[:, 2]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 0] = theta[i,i]*p[:, 0]
    p[:, 1] = theta[i, i] * p[:, 1]
    p[:, (2,3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:, 2] = theta[i, j]*p[:, 2]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p[:, 1] = theta[i, n]*p[:, 1]
    p = p.flatten(order="F")
    return p


def kronvec_prim_diag(theta, p, i, n):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(i):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,1] = theta[i,j] * p[:, 1]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten(order="F")

    # Non diagonal 4x4 Kronecker factors j=i
    p = p.reshape((2**(2*n-1), 4), order="C")
    p[:, 0] = theta[i,i]*p[:, 0]
    p[:, 2] = theta[i,i]*p[:, 2]
    p[:, (1,3)] = 0
    p = p.flatten(order="F")

    # Diagonal 4x4 Kronecker factors j>i
    for j in range(i+1, n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,1] = theta[i,j] * p[:, 1]
        p[:, 3] = theta[i, j]*p[:, 3]
        p = p.flatten("F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = 0
    p = p.flatten(order="F")
    return p


def kronvec_seed_diag(theta, p, n):
    # Diagonal 4x4 Kronecker factors j<i
    for j in range(n):
        p = p.reshape((2**(2*n-1), 4), order="C")
        p[:,(1, 2)] = 0
        p[:, 3] = theta[n, j]*p[:, 3]
        p = p.flatten(order="F")

    # Diagonal 2x2 Kronecker factor for j=n
    p = p.reshape((2 ** (2 * n), 2), order="C")
    p[:, 0] = theta[n,n]*p[:,0]
    p[:, 1] = 0
    p = p.flatten(order="F")
    return p


def diag_q(theta):
    n = theta.shape[0] - 1
    p = np.ones(2 ** (2 * n + 1))
    tmp = 1.*np.zeros(p.shape[0])
    for i in range(n):
        tmp = tmp + kronvec_met_diag(theta, p.copy(), i, n) + \
              kronvec_prim_diag(theta, p.copy(),i,n) +\
              kronvec_sync_diag(theta, p.copy(), i, n)
    return tmp + kronvec_seed_diag(theta, p, n)