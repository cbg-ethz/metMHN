import numpy as np

from ssr_kronecker_vector import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed, kronvec, kron_diag
import Utilityfunctions as utils


def x_partial_Q_y(log_theta: np.array, x: np.array, y: np.array, state: np.array) -> np.array:
    """This function computes x \partial Q y with \partial Q the Jacobian of Q w.r.t. all thetas
    efficiently using the shuffle trick (sic!).

    Args:
        log_theta (np.array): Logarithmic theta values of the MHN 
        x (np.array): x vector to multiply with from the left. Length must equal the number of
        nonzero entries in the state vector.
        y (np.array): y vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        state (np.array): Binary state vector, representing the current sample's events.

    Returns:
        np.array: x \partial_(\Theta_{ij}) Q y for i, j = 1, ..., n+1
    """
    n = log_theta.shape[0] - 1

    z = np.zeros(shape=(n + 1, n + 1))

    for i in range(n):
        z_sync = x * kronvec_sync(log_theta=log_theta,
                                  p=y, i=i, n=n, state=state)
        z_prim = x * kronvec_prim(log_theta=log_theta,
                                  p=y, i=i, n=n, state=state)
        z_met = x * kronvec_met(log_theta=log_theta,
                                p=y, i=i, n=n, state=state)

        z[i, -1] = z_met.sum()

        for j in range(n):
            current = state[2*j: 2*j + 2]

            if current.sum() == 0:
                if i == j:
                    z[i, j] = sum([
                        z_sync.sum(),
                        z_prim.sum(),
                        z_met.sum()]
                    )

            elif current.sum() == 2:
                z_sync = z_sync.reshape((-1, 4), order="C")
                z_prim = z_prim.reshape((-1, 4), order="C")
                z_met = z_met.reshape((-1, 4), order="C")

                z[i, j] = sum([
                    z_sync[:, 3].sum(),
                    z_prim[:, [1, 3]].sum(),
                    z_met[:, [2, 3]].sum()]
                )

                if i == j:
                    z[i, j] += sum([
                        z_sync[:, 0].sum(),
                        z_prim[:, [0, 2]].sum(),
                        z_met[:, [0, 1]].sum()]
                    )

                z_sync = z_sync.flatten(order="F")
                z_prim = z_prim.flatten(order="F")
                z_met = z_met.flatten(order="F")

            else:
                z_sync = z_sync.reshape((-1, 2), order="C")
                z_prim = z_prim.reshape((-1, 2), order="C")
                z_met = z_met.reshape((-1, 2), order="C")

                if i != j:
                    if current[1] == 1:  # met mutated
                        z[i, j] = z_met[:, 1].sum()
                    else:  # prim mutated
                        z[i, j] = z_prim[:, 1].sum()
                else:
                    z[i, j] = sum([
                        z_sync[:, 0].sum(),
                        z_prim.sum(),
                        z_met.sum()]
                    )
                z_sync = z_sync.flatten(order="F")
                z_prim = z_prim.flatten(order="F")
                z_met = z_met.flatten(order="F")

    z_seed = x * kronvec_seed(log_theta=log_theta, p=y, n=n, state=state)

    z[-1, -1] = z_seed.sum()

    for j in range(n):
        current = state[2*j: 2*j + 2]

        if current.sum() == 2:
            z_seed = z_seed.reshape((-1, 4), order="C")

            z[-1, j] = z_seed[:, 3].sum()

            z_seed = z_seed.flatten(order="F")

        elif current.sum() == 1:
            z_seed = z_seed.reshape((-1, 2), order="C").flatten(order="F")

    return z


def R_i_inv_vec(log_theta: np.array, x: np.array, lam: float,  state: np.array, transpose: bool = False) -> np.array:
    """This computes R_i^{-1} x = (\lambda_i I - Q)^{-1} x

    Args:
        log_theta (np.array): Log values of the theta matrix
        x (np.array): Vector to multiply with from the right. Length must equal the number of
        nonzero entries in the state vector.
        lam (float): Value of \lambda_i
        state (np.array): Binary state vector, representing the current sample's events.


    Returns:
        np.array: R_i^{-1} x
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
    R_1_inv_p_0 = R_i_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam1,
        state=state)

    R_2_inv_p_0 = R_i_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam2,
        state=state
    )

    p_theta = lam * (R_1_inv_p_0 - R_2_inv_p_0)

    # some states are not reachable and therefore have zero probability density
    minuend = p_D * np.divide(lam, p_theta, where=reachable[restricted])
    subtrahend = minuend.copy()
    minuend = R_i_inv_vec(log_theta=log_theta, x=minuend,
                          lam=lam2, state=state, transpose=True)
    subtrahend = R_i_inv_vec(log_theta=log_theta, x=minuend,
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
    p_th = R_i_inv_vec(
        log_theta=log_theta,
        x=p_0,
        lam=lam2,
        state=state
    ) - R_i_inv_vec(
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

    n = 2
    sparsity = 0.5
    log_theta = utils.random_theta(n=n, sparsity=sparsity)

    state = np.random.randint(2, size=2*n+1)
    length = 2**sum(state)
    lam1, lam2 = np.random.random(2)

    p = np.zeros(2**(2*n+1))
    p[0] = 1
    gradient(log_theta=log_theta, p_D=p[utils.ssr_to_fss(state)],
        lam1=lam1,lam2=lam2, state=state)
