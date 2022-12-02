from itertools import compress
import numpy as np


def state_space(n: int) -> list:
    """
    Args:
        n (int): total number of events
    Returns:
         list: complete statespace
    """
    return [f'{i:0b}'.zfill(n)[::-1] for i in range(2**n)]


def trunk_states(dpoint: list) -> list:
    """
    Args:
        dpoint (list): Bitstring, genotype of a single patient
    Returns:
        list: states that the tumor could have visited during progression
    """
    n = len(dpoint)
    inds = np.ones(1)
    for i in range(n):
        if dpoint[i] == 1:
            inds = np.kron(np.array([1, 1]), inds)
        else:
            inds = np.kron(np.array([1, 0]), inds)

    return [state for state in compress(state_space(n), inds)]


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
    theta = np.zeros((npone, npone))
    theta += np.diag(np.random.normal(size=npone))
    index = np.argwhere(theta == 0)[
        np.random.choice(npone**2-npone, size=int((npone**2-npone)
                         * (1-sparsity)), replace=True)
    ]
    theta[index[:, 0], index[:, 1]] = np.random.normal(
        size=int((npone**2-npone)*(1-sparsity)))
    return theta
