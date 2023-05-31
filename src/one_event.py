import numpy as np

# This contains the important functions for the case that state_size = 1
# This needs to be done separately, as otherwise reshape(-1, 4) throws an error
# in the tracing process of jax.


def small_Q(log_theta: np.array) -> np.array():

    return np.array(
        [
            sum(-np.exp(log_theta[i, i])
                for i in range(n + 1)),
            0.
        ],
        [
            np.exp(log_theta[-1, -1]),
            sum(-np.exp(log_theta[i, i])
                * (1 + np.exp(log_theta[i, -1])) for i in range(n))
        ])


def kronvec(log_theta: np.array, p: np.array,
            diag: bool = True, transpose: bool = False) -> np.array:

    if diag:
        Q = small_Q(log_theta=log_theta)
        if not transpose:
            return Q @ p
        else:
            return Q.T @ p
    else:
        if not transpose:
            return np.array([0., p[0] * np.exp(log_theta[-1, -1])])
        else:
            return np.array([p[1] * np.exp(log_theta[-1, -1]), 0.])


def R_i_inv_vec(log_theta: np.array, x: np.array, lam: float, transpose: bool = False) -> np.array:

    R = lam * np.eye(2) - small_Q(log_theta=log_theta)

    b = x.copy()

    if not transpose:
        b[0] /= R[0, 0]
        b[1] -= b[0] * R[1, 0]
        b[1] /= R[1, 1]
    else:
        b[1] /= R[1, 1]
        b[0] -= b[1] * R[1, 0]
        b[0] /= R[0, 0]


def x_partial_Q_y(log_theta: np.array, x: np.array, y: np.array) -> np.array:

    z = np.zeros_like(log_theta)
    n = log_theta.shape[0]

    for i in range(n):
        theta_ii, theta_iM = np.exp(log_theta[i, [i, -1]])
        z[i, i] = -theta_ii * x @ np.diag([1., 1 + theta_iM]) @ y
        z[i, -1] = x @ np.diag([0., -theta_ii * theta_iM]) @ y
    theta_MM = np.exp(log_theta[-1, -1])
    z[-1, -1] = x @ np.array([[-theta_MM, 0.], [theta_MM, 0.]]) @ y
