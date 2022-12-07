import numpy as np

from ssr_kronecker_vector import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed, kronvec, kron_diag
import Utilityfunctions as utils


def R_inv_vec(log_theta: np.array, x: np.array, lam: float,  state: np.array, transpose: bool = False) -> np.array:
    """This computes R^{-1} x = (\lambda I - Q)^{-1} x

    Args:
        log_theta (np.array): Log values of the theta matrix
        x (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        lam (float): Value of \lambda
        state (np.array): Binary state vector, representing the current sample's events.


    Returns:
        np.array: R^{-1} x
    """
    n = log_theta.shape[0] - 1

    lidg = -1 / (kron_diag(log_theta=log_theta, n=n, state=state) - lam)
    y = lidg * x

    for _ in range(sum(state) + 1):
        y = lidg * (kronvec(log_theta=log_theta, p=y, n=n,
                            state=state, diag=False, transpose=transpose) + x)

    return y


def gradient(log_theta: np.array, p_D: np.array, lam1: float, lam2: float, state: np.array) -> np.array:
    """This computes the gradient of the score function, which is the log-likelihood of a data vector p_D
    with respect to the log_theta matrix

    Args:
        log_theta (np.array): Log values of the theta matrix.
        p_D (np.array): Data vector.
        lam1 (float): Rate of the first sampling.
        lam2 (float): Rate of the second sampling.
        state (np.array): Binary state vector, representing the current sample's events.


    Returns:
        np.array: \partial_theta (p_D^T log p_theta)
    """

    restricted = utils.ssr_to_fss(state)

    reachable = utils.reachable_states(log_theta.shape[0] - 1)
    if np.any(p_D[~reachable[restricted]] != 0):
        raise ValueError("The data vector contains unreachable states.")

    n_ss = sum(state)
    p_0 = np.zeros(2**n_ss)
    p_0[0] = 1
    lam = (lam1 * lam2 / (lam1 - lam2))
    R_1_inv_p_0 = R_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam1,
        state=state)

    R_2_inv_p_0 = R_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam2,
        state=state
    )

    p_theta = lam * (R_1_inv_p_0 - R_2_inv_p_0)

    # some states are not reachable and therefore have zero probability density
    minuend = p_D * np.divide(lam, p_theta, where=reachable[restricted])
    subtrahend = minuend.copy()
    minuend = R_inv_vec(log_theta=log_theta, x=minuend,
                          lam=lam2, state=state, transpose=True)
    subtrahend = R_inv_vec(log_theta=log_theta, x=minuend,
                             lam=lam1, state=state, transpose=True)
    minuend = x_partial_Q_y(log_theta=log_theta,
                            x=minuend, y=R_2_inv_p_0, state=state)
    subtrahend = x_partial_Q_y(
        log_theta=log_theta, x=subtrahend, y=R_1_inv_p_0, state=state)

    return minuend - subtrahend


def log_likelihood(log_theta: np.array, p_D: np.array, lam1: float, lam2: float, state: np.array) -> float:

    reachable = utils.reachable_states(log_theta.shape[0] - 1)
    if np.any(p_D[~reachable] != 0):
        raise ValueError("The data vector contains unreachable states.")

    p_0 = np.zeros(2 ** state.sum())
    p_0[0] = 1
    p_th = R_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam2,
        state=state
    ) - R_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam1,
        state=state
    )
    p_th *= lam1 * lam2 / (lam1 - lam2)

    return np.dot(
        p_D[utils.ssr_to_fss(state) & reachable],
        np.log(p_th[reachable[utils.ssr_to_fss(state)]])
    )


if __name__ == "__main__":

    pass