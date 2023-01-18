from itertools import compress, chain, combinations
import numpy as np

import Utilityfunctions


def state_space(n: int) -> np.array:
    """
    Enumerates all possible states in lexicographic order
    Args:
        n (int): total number of events
    Returns:
         np.array: complete statespace
    """
    return np.array([f'{i:0b}'.zfill(n)[::-1] for i in range(2**n)])


def trunk_states(state: np.array) -> np.array:
    """
    Enumerates all possible states that a tumor(pair) with genotype(s) "state" could have visited
    Args:
        state (np.array): Bitstring, genotype of a tumor(pair) of a single patient
    Returns:
        np.array
    """
    n = state.size
    inds = np.ones(2**n)
    for i in range(n):
        if state[i] == 1:
            inds[0:2**(i+1)] = np.kron(np.array([1, 1]), inds[0:2**i])
        else:
            inds[0:2**(i+1)] = np.kron(np.array([1, 0]), inds[0:2**i])

    return np.array([geno for geno in compress(state_space(n), inds)])


def ssr_to_fss(state: np.array) -> np.array:
    """This gives the indices of the rate matrix that are appearing in the
    state space restricted rate matrix.

    Args:
        state (np.array): Binary state vector, representing the current sample's events.
    Returns:
        np.array: Indices of the rate matrix.
    """
    res = np.ones(1)
    for s in state:
        res = np.kron(np.array([1, s]), res)
    return res.astype(bool)


def random_theta(n: int, sparsity: float) -> np.array:
    """
    Generates a logarithmic theta with normal distributed entries
    Args:
        n (int): Number of mutations
        sparsity (float): Percentage of zero entries in theta
    returns:
        np.array: theta
    """
    npone = n + 1
    log_theta = np.zeros((npone, npone))
    log_theta += np.diag(np.random.normal(size=npone))
    index = np.argwhere(log_theta == 0)[
        np.random.choice(npone**2-npone, size=int((npone**2-npone)
                         * (1-sparsity)), replace=True)
    ]
    log_theta[index[:, 0], index[:, 1]] = np.random.normal(
        size=int((npone**2-npone)*(1-sparsity)))
    return log_theta


def reachable_states(n: int):

    reachable = np.zeros(2**(2*n + 1))

    for i in chain.from_iterable(combinations(list(range(n)), r) for r in range((n+1))):
        reachable[sum((2**(2*j))*3 for j in i)] = 1

    reachable[2**(2*n):] = 1
    return reachable.astype(bool)


def finite_sample(pTh: np.array, k: int) -> np.array:
    """
    Generates k random samples drawn from a probability distribution pTh
    Code taken from https://github.com/spang-lab/LearnMHN/blob/main/mhn/original/UtilityFunctions.py
    Args:
        pTh (np.array): Probability distribution to draw samples from
        k (int): Number of samples to draw
    Returns:
         np.array: Subsampled probability distribution
    """
    n = pTh.size
    return np.bincount(np.random.choice(n, k, replace=True, p=pTh), minlength=n) / k


def diagnosis_theta(log_theta: np.array, log_diag_rate: float = 0) -> np.array:
    """
    transforms a logarithmic theta in order to be used for the diagnosis MHN
    Args:
        log_theta (np.array): theta matrix with logarithmic entries
        log_diag_rate (float): logarithmic base rate of the diagnosis event
    Returns:
         (n+1)x(n+1) dimensional theta matrix
    """
    n = log_theta.shape[0] - 1
    theta_diag = np.zeros(shape=(n + 2, n + 2), dtype=float)
    theta_diag[0:n, 0:n] = log_theta[0:n, 0:n]
    theta_diag[-1, 0:n] = log_theta[n, 0:n]
    theta_diag[0:n, -1] = log_theta[0:n, n]
    theta_diag[0:n + 2, -2] = np.array([-np.inf] * (n + 2))
    theta_diag[n, n] = log_diag_rate
    theta_diag[n + 1, n + 1] = log_theta[n, n]
    return theta_diag

def marginalize(p_in: np.array, n: int, marg_met: bool=True) -> np.array:
    """
    Returns the marginal distribution of a joint distribution of primary tumors and metastases wrt. to the tumor type
    Args:
        p_in (np.array): Joint distribution to marginalise
        n (int): number of genomic events
        marg_met (bool): If true: marginalize over metastases, else: marginalize over primaries
    Returns:
         np.array: 2^n dimensional marginal distribution
    """
    p = p_in.copy()
    for _ in range(n):
        p = p.reshape((-1, 4), order="C")
        if not marg_met:
            y = np.column_stack((p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]))
        else:
            y = np.column_stack((p[:, 0] + p[:, 1], p[:, 2] + p[:, 3]))
        p = y.flatten(order="F")

    p = p.reshape((-1, 2), order="C")
    return p.flatten(order="F")

def ssr_marginalize(p_in: np.array, n: int, state: np.array, marg_met: bool=True, marg_seeding: bool=False) -> np.array:
    """
    Returns the marginal distribution of a joint distribution of primary tumors and metastases wrt. to the tumor type
    Args:
        p_in (np.array): Joint distribution to marginalise
        n (int): number of genomic events
        state (np.array): Binary state vector, representing the current sample's events
        marg_met (bool): If true: marginalize over metastases, else: marginalize over primaries
        marg_seeding (bool): If true marginalize over the status of the seeding event
    Returns:
         np.array: 2^m dimensional marginal distribution
    """
    p = p_in.copy()
    for i in range(n):
        mut = state[2 * i: 2 * i + 2]
        muts = mut.sum()
        print(mut)

        if muts == 0:               
            pass
        # Prim[i] = 1 and Met[i] = 0 and marg. over mets
        # Or Prim[i] = 0 and Met[i] = 1 and marg. over prims
        elif (mut[0] == 0 and not marg_met) or (mut[1] == 0 and marg_met):
            p = p.reshape((-1,2), order="C").ravel(order="F")
            print(i)
        # Prim[i] = Met[i] = 1
        elif muts == 2:
            p = p.reshape((-1, 4), order = "C")
            if marg_met:
                 # Marg. over mets
                y = np.column_stack((p[:, 0] + p[:, 2], p[:, 1] + p[:, 3]))
            else:
                # Marg. over prims
                y = np.column_stack((p[:, 0] + p[:, 1], p[:, 2] + p[:, 3]))
            p = y.ravel(order="F")
        # Prim[i] = 0 and Met[i] = 1 and marg. over mets
        # Or Prim[i] =1 and Met[i] = 0 and marg. over prims
        else: 
            p = p.reshape((-1, 2), order = "C")
            p = p[:, 0] + p[:, 1]
            p = p.ravel(order="F")

    if state[-1] == 1 and marg_seeding:
        p = p.reshape((-1, 2), order="C")
        p = p[:, 0] + p[:, 1]
    else:
        p =  p.reshape((-1,2)).ravel(order="F")
    return p
