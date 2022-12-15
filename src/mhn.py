import numpy as np


def kron_vec(theta: np.ndarray, i: int, x_vec: np.ndarray, diag: bool = False,
             transp: bool = False) -> np.ndarray:
    """
    This function multiplies the kronecker-product you get from the ith row of theta with a vector
    :param theta: matrix containing the theta values
    :param i: row of theta used for the kronecker-product
    :param x_vec: vector that is multiplied with the kronecker-product matrix
    :param diag: if False, the diagonal of the kronecker-product matrix is set to zero
    :param transp: if True, the kronecker-product matrix is transposed
    :return:
    """
    theta_i = np.exp(theta[i])
    n = theta_i.size

    x_vec = x_vec.astype(np.float64)

    for j in range(n - 1, -1, -1):
        x = x_vec.reshape((2, 2 ** (n - 1)))

        if j == i:
            if not transp:
                x[1] = x[0] * theta_i[j]
                if diag:
                    x[0] = -x[1]
                else:
                    x[0] = 0
            else:
                if diag:
                    x[0] = (x[1] - x[0]) * theta_i[j]
                    x[1] = 0
                else:
                    x[0] = x[1] * theta_i[j]
                    x[1] = 0

        else:
            x[1] = x[1] * theta_i[j]

        x_vec = x.T.flatten()

    return x_vec


def q_vec(theta: np.ndarray, x: np.ndarray, diag: bool = False, transp: bool = False) -> np.ndarray:
    """
    Multiplies the vector x with the matrix Q
    :param theta: thetas used to construct Q
    :param x: vector that is multiplied with Q
    :param diag: if False, the diagonal of Q is set to zero
    :param transp: if True, x is multiplied with Q^T
    :return: product of Q and x
    """
    n = theta.shape[1]
    y = np.zeros(2**n)

    for i in range(n):
        y += kron_vec(theta, i, x, diag, transp)

    return y

def q_subdiag(theta: np.ndarray, i: int) -> np.ndarray:
    """
    Creates a single subdiagonal of Q from the ith row in Theta
    :return: subdiagonal of Q corresponding to the ith row of Theta
    """
    row = theta[i]
    n = row.size

    # s is the subdiagonal of Q, the entries are calculated as described in eq. 2
    s = np.empty(2**n)
    s[0] = np.exp(row[i])

    for j in range(n):
        s[2**j: 2**(j+1)] = s[:2**j] * np.exp(row[j]) * (i != j)

    return s

def q_diag(theta: np.ndarray) -> np.ndarray:
    """
    get the diagonal of Q
    :param theta: theta representing the MHN
    """
    n = theta.shape[1]
    dg = np.zeros((2**n))

    for i in range(n):
        # the diagonal elements are the negative sums of their columns
        dg = dg - q_subdiag(theta, i)

    return dg


def jacobi(theta: np.ndarray, b: np.ndarray, lam, transp: bool = False) -> np.ndarray:
    """
    Returns the solution for [I - Q]^-1 x = b
    :param theta: thetas used to construct Q
    :param b:
    :param transp: if True, returns solution for ([I - Q]^-1)^T x = b
    :return:
    """
    n = theta.shape[1]

    x = np.full(2**n, 1 / 2**n)

    dg = lam - q_diag(theta)

    for _ in range(n+1):
        x = b + q_vec(theta, x, diag=False, transp=transp)
        x = x / dg

    return x


def generate_pTh(theta: np.ndarray, lam, p0: np.ndarray = None) -> np.ndarray:
    """
    Returns the probability distribution given by theta
    :param theta:
    :param p0:
    :return:
    """
    n = theta.shape[1]

    if p0 is None:
        p0 = np.zeros(2**n)
        p0[0] = 1

    return jacobi(theta, p0, lam)

