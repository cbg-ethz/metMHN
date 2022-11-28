import numpy as np
import pandas as pd
from itertools import compress

# Some helper functions


def dia2(tij):
    ret = np.diag([-tij, 0])
    ret[1, 0] = tij
    return ret

# List all possible states in lexicographic order


def state_space(n):
    return [f'{i:05b}'[::-1] for i in range(2 ** n)]

# List all possible states, that datapoint could have visited


def trunk_states(dpoint):
    n = len(dpoint)
    inds = np.ones(1)
    for i in range(n):
        if dpoint[i] == 1:
            inds = np.kron(np.array([1, 1]), inds)
        else:
            inds = np.kron(np.array([1, 0]), inds)

    return [state for state in compress(state_space(n), inds)]


# Methods for Building the full Q explicitly

def multi_kron_prods(diag_fac, off_diag_fac, last, theta, i, n):
    Q = np.eye(1)
    for j in range(i):
        Q = np.kron(diag_fac(theta[i, j]), Q)
    Q = np.kron(off_diag_fac, Q)
    for j in range(i+1, n):
        Q = np.kron(diag_fac(theta[i, j]), Q)
    return np.kron(last, Q)


def diag_sync(tij):
    return np.diag([1, 0, 0, tij])


def diag_met(tij):
    return np.diag([1, 1, tij, tij])


def diag_prim(tij):
    return np.diag([1, tij, 1, tij])


def build_q(theta):
    n = theta.shape[0]-1
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

# Methods for explicit calculation of the SSR version of Q

# Builds Q_i for the synchronized part of Q


def sync_ssr_q(mut, theta, i, n):
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
    Q = np.kron(diag10, Q)
    return Q

# Builds Q_i for the metastatic part of Q after seeding


def met_ssr_q(mut, theta, i, n):
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


def prim_ssr_q(mut, theta, i, n):
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


def seeding_ssr_q(dpoint, theta, n):
    mut = [dpoint[j] + 2 * dpoint[j + 1] for j in range(0, 2 * n, 2)]
    diag10 = np.diag([1, 0])
    Q = np.diag([1])
    for j in range(n):
        if mut[j] == 3:
            Q = np.kron(np.diag([1, 0, 0, theta[n, j]]), Q)
        elif mut[j] > 0:
            Q = np.kron(diag10, Q)
    if dpoint[2 * n] == 1:
        Q = np.kron(dia2(theta[n, n]), Q)
    else:
        Q = Q * (-theta[n, n])
    return Q


def ssr_build_q(dpoint, theta):
    n = theta.shape[0] - 1
    mut = [dpoint[j] + 2 * dpoint[j + 1] for j in range(0, 2 * n, 2)]
    Q = np.zeros(2 ** (sum(dpoint)))
    for i in range(n):
        Q = Q + sync_ssr_q(mut, theta, i, n) + met_ssr_q(mut,
                                                         theta, i, n) + prim_ssr_q(mut, theta, i, n)
    return Q + seeding_ssr_q(dpoint, theta, n)
