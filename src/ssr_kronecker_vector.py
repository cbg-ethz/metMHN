import numpy as np


def kronvec_sync(log_theta: np.array, p: np.array, i: int, n: int, state: np.array, diag: bool = True) -> np.array:
    """This computes the restricted version of the product of the synchronized part of the ith Q summand
    Q_i with a vector Q_i p.

    Args:
        log_theta (np.array): Log values of the theta matrix
        p (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (np.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.

    Returns:
        np.array: Q_i p
    """
    # there are no non-diagonal entries if event i is not mutated in both prim and met
    if not diag and sum(state[2 * i: 2 * i + 2]) != 2:
        return np.zeros(p.size)

    y = p.copy()

    for j in range(n):

        mut = state[2 * j: 2 * j + 2]
        if mut.sum() == 0:  # neither prim nor met
            if i == j:
                y *= -np.exp(log_theta[i, i])
        elif mut.sum() == 1:  # prim xor met
            y = y.reshape((-1, 2), order="C")
            y[:, 1] = 0
            if i == j:
                y[:, 0] *= -np.exp(log_theta[i, i])
            y = y.flatten(order="F")
        else:  # both prim and met
            y = y.reshape((-1, 4), order="C")
            y[:, [1, 2]] = 0
            if i == j:
                if diag:
                    y[:, 0] *= -np.exp(log_theta[i, i])
                y[:, 3] = -1 * y[:, 0]
            else:
                y[:, 3] *= np.exp(log_theta[i, j])
            y = y.flatten(order="F")
    if state[-1] == 1:
        y = y.reshape((-1, 2), order="C")
        y[:, 1] = 0
        y = y.flatten(order="F")

    return y


def kronvec_prim(log_theta: np.array, p: np.array, i: int, n: int, state: np.array, diag: bool = True) -> np.array:
    """This computes the restricted version of the product of the asynchronous primary tumour
    part of the ith Q summand Q_i with a vector Q_i p.

    Args:
        log_theta (np.array): Log values of the theta matrix
        p (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (np.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.

    Returns:
        np.array: Q_i p
    """

    # there are no non-diagonal entries if event i is not mutated in prim
    if not diag and state[2 * i] == 0:
        return np.zeros(p.size)

    if state[-1] == 0:
        return np.zeros(p.size)

    y = p.copy()

    for j in range(n):

        mut = state[2 * j: 2 * j + 2]
        if mut.sum() == 0:
            if i == j:
                y *= -np.exp(log_theta[i, i])
        elif mut.sum() == 2:
            y = y.reshape((-1, 4), order="C")
            if i == j:
                theta = np.exp(log_theta[i, i])
                if diag:
                    y[:, [0, 2]] *= -theta
                    y[:, [1, 3]] = - y[:, [0, 2]]
                else:
                    y[:, [1, 3]] = theta * y[:, [0, 2]]
                    y[:, [0, 2]] = 0
            else:
                theta = np.exp(log_theta[i, j])
                y[:, 1] *= theta
                y[:, 3] *= theta
            y = y.flatten(order="F")
        else:
            y = y.reshape((-1, 2), order="C")
            if i == j:
                theta = np.exp(log_theta[i, i])
                y[:, 0] *= -theta
                if mut[0] == 1:  # prim mutated
                    y[:, 1] = -y[:, 0]
                    if not diag:
                        y[:, 0] = 0
                else:  # met mutated
                    y[:, 1] *= -theta
            else:
                if mut[0] == 1:
                    y[:, 1] *= np.exp(log_theta[i, j])
            y = y.flatten(order="F")
    y = y.reshape(-1, 2)
    y[:, 0] = 0
    y = y.flatten(order="F")

    return y


def kronvec_met(log_theta: np.array, p: np.array, i: int, n: int, state: np.array, diag: bool = True) -> np.array:
    """This computes the restricted version of the product of the asynchronous metastasis
    part of the ith Q summand Q_i with a vector Q_i p.

    Args:
        log_theta (np.array): Log values of the theta matrix
        p (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        i (int): Index of the summand.
        n (int): Total number of events in the MHN.
        state (np.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.

    Returns:
        np.array: Q_i p
    """

    # there are no non-diagonal entries if event i is not mutated in met
    if not diag and state[2 * i + 1] == 0:
        return np.zeros(p.size)

    if state[-1] == 0:
        return np.zeros(p.size)

    y = p.copy()

    for j in range(n):

        mut = state[2 * j: 2 * j + 2]
        if mut.sum() == 0:
            if i == j:
                y *= -np.exp(log_theta[i, i])
        elif mut.sum() == 2:
            y = y.reshape((-1, 4), order="C")
            if i == j:
                theta = np.exp(log_theta[i, i])
                if diag:
                    y[:, [0, 1]] *= -theta
                    y[:, [2, 3]] = -1 * y[:, [0, 1]]
                else:
                    y[:, [2, 3]] = theta * y[:, [0, 1]]
                    y[:, [0, 1]] = 0
            else:
                theta = np.exp(log_theta[i, j])
                y[:, [2, 3]] *= theta
            y = y.flatten(order="F")
        else:
            y = y.reshape((-1, 2), order="C")
            if i == j:
                theta = np.exp(log_theta[i, i])
                y[:, 0] *= -theta
                if mut[1] == 1:  # met mutated
                    y[:, 1] = -y[:, 0]
                    if not diag:
                        y[:, 0] = 0
                else:  # prim mutated
                    y[:, 1] *= -theta
            else:
                if mut[1] == 1:
                    y[:, 1] *= np.exp(log_theta[i, j])
            y = y.flatten(order="F")
    y = y.reshape(-1, 2)
    y[:, 0] = 0
    y[:, 1] *= np.exp(log_theta[i, -1])
    y = y.flatten(order="F")

    return y


def kronvec_seed(log_theta: np.array, p: np.array, n: int, state: np.array, diag: bool = True) -> np.array:
    """This computes the restricted version of the product of the seeding summand of Q with a vector Q_M p.

    Args:
        log_theta (np.array): Log values of the theta matrix
        p (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (np.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q_M (and not set it to 0). Defaults to True.

    Returns:
        np.array: Q_i p
    """

    # there are no non-diagonal entries if met has not seeded
    if not diag and state[-1] == 0:
        return np.zeros(p.size)

    y = p.copy()

    for j in range(n):

        mut = state[2 * j: 2 * j + 2]
        if mut.sum() == 2:
            y = y.reshape((-1, 4), order="C")
            y[:, [1, 2]] = 0
            y[:, 3] *= np.exp(log_theta[-1, j])
            y = y.flatten(order="F")
        elif mut.sum() == 1:
            y = y.reshape((-1, 2), order="C")
            y[:, 1] = 0
            y = y.flatten(order="F")
    if state[-1] == 1:
        y = y.reshape((-1, 2), order="C")
        y[:, 1] = np.exp(log_theta[-1, -1]) * y[:, 0]
        if diag:
            y[:, 0] = -y[:, 1]
        else:
            y[:, 0] = 0
        y = y.flatten(order="F")
    else:
        y *= -np.exp(log_theta[-1, -1])

    return y


def kronvec(log_theta: np.array, p: np.array, n: int, state: np.array, diag: bool = True) -> np.array:
    """This computes the restricted version of the product of the rate matrix Q with a vector Q p.

    Args:
        log_theta (np.array): Log values of the theta matrix
        p (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        n (int): Total number of events in the MHN.
        state (np.array): Binary state vector, representing the current sample's events.
        diag (bool, optional): Whether to use the diagonal of Q (and not set it to 0). Defaults to True.

    Returns:
        np.array: _description_
    """
    y = np.zeros(shape=2**sum(state))
    for i in range(n):
        y += kronvec_sync(log_theta=log_theta, p=p, i=i, n=n, state=state, diag=diag)
        y += kronvec_prim(log_theta=log_theta, p=p, i=i, n=n, state=state, diag=diag)
        y += kronvec_met(log_theta=log_theta, p=p, i=i, n=n, state=state, diag=diag)
    y += kronvec_seed(log_theta=log_theta, p=p, n=n, state=state, diag=diag)

    return y
