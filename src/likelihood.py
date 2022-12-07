import kronecker_vector as fss
import numpy as np
import Utilityfunctions as utils

def jacobi(log_theta: np.array, b: np.array, lam: float, transpose: bool = False, x_inp: np.array = None) -> np.array:
    """
    This function computes (I-Q)x=b using jacobi iteration
    Args:
        log_theta (np.array): Logarithmic theta matrix
        b (np.array): State vector, resolvent is multiplied from the left to this vector
        x (np.aray): Helper vector, result of (lam*I-Q)^-1 b
        lam (float): rate of sampling
        transpose (bool): Flag, indicates whether Q should be transposed
    Returns:
        np.array: (lam*I-Q)^-1 b
    """
    n = log_theta.shape[0]-1
    if x_inp is None:
        x = np.ones(2**(2*n+1))/(2**(2*n+1))
    else:
        x = x_inp.copy()
    dg = fss.diag_q(log_theta) + lam
    
    for _ in range(2*n+2):
        x = b + fss.qvec(log_theta, x, diag=False, transpose=transpose)
        x = x/dg
    return x


def generate_pths(log_theta: np.array, p_0: np.array, lam1: float, lam2: float) -> np.array:
    """
    calculates the two components of the pth vector
    Args:
        log_theta (np.array): logarithmic theta matrix
        p_0 (np.array): starting distribution
        lam1 (float): rate of first diagnosis
        lam2 (float): rate of second diagnosis

    Returns:
         np.array: pth1, pth2
    """
    return jacobi(log_theta, p_0, lam1), jacobi(log_theta, p_0, lam2)


def diag_forward(log_theta: np.array, p: np.array) -> np.array:
    """
    calculates stationary distribution using diagnosis formalism
    Arguments:
        theta (np.array): logarithmic theta matrix
        p (np.array): Starting distribution
    Returns:
        np.array: stationary distribution
    """
    n = log_theta.shape[0]-1
    diagnosed = fss.diag_diagnosis(n)
    dg = fss.diag_q(log_theta)+diagnosed
    for i in range(2*n+1):
        np.divide(p, dg, out=p, where=dg != 0)
        p = fss.qvec(log_theta, p.copy(), False) + diagnosed*p
    return p



def likelihood(log_theta: np.array, p_D: np.array, lam1: float, lam2: float,
               p_th_1_space: np.array, p_th_2_space: np.array) -> float:
    """
    This function computes the log likelihood score
    Args:
        log_theta (np.array): Logarithmic theta matrix
        p_D (np.array): Vector holding empirical frequencies of genotypes in a dataset
        lam1 (float): Rate of first sampling
        lam2 (float): Rate of second sampling
        p_th_1_space (np.array): Second term needed to calculate pth
        p_th_2_space (np.array): First term need to calculate pth
    Returns:
         float: pd^T log(p_th)
    """
    n = log_theta.shape[0] - 1
    p_0 = np.zeros(2 ** (2 * n + 1))
    p_0[0] = 1
    p_th_1_space, p_th_2_space = generate_pths(log_theta, p_0, lam1, lam2)
    p_th = lam1 * lam2 / (lam1 - lam2) * (p_th_2_space - p_th_1_space)
    return p_D.dot(np.log(p_th, out=np.zeros_like(p_th), where=utils.reachable_states(n=n)))



def gradient(log_theta: np.array, p_D: np.array, lam1: float, lam2: float, n: int, p_0: np.array,
             p_th_1_space: np.array = None, p_th_2_space: np.array = None) -> np.array:
    """
    Calculates the gradient of the likelihood
    Args:
        log_theta (np.array): logarithmic theta matrix
        p_D (np.array): Data vector holding frequencies of genotypes in D
        lam1 (float): Rate of first diagnosis
        lam2 (float): Rate of delta t
        n (int): Number of mutations
        p_0 (np.array): Starting distribution
        p_th_1_space (np.array): First component of p_th, updated in the previous call to likelihood
        p_th_2_space (np.array): Second component of p_th, updated in the previous call to likelihood
    Returns:
         np.array: g, gradient of likelihood
    """
    # Build p_theta
    if p_th_1_space is None and p_th_2_space is None:
        p_th_1, p_th_2 = generate_pths(log_theta, p_0, lam1, lam2)
    else:
        p_th_1 = p_th_1_space
        p_th_2 = p_th_2_space
    lam_ratio = lam1 * lam2 / (lam1 - lam2)
    p_th = lam_ratio * (p_th_2 - p_th_1)

    # Build the vector to multiply from the left to dQ/d th
    q = np.divide(p_D, p_th, out=np.zeros_like(p_D), where=p_th != 0)
    q1 = jacobi(log_theta, q, lam1, True)
    q2 = jacobi(log_theta, q, lam2, True)
    d_theta = fss.x_partial_Q_y(log_theta, q2, p_th_2, n) - fss.x_partial_Q_y(log_theta, q1, p_th_1, n)

    # Derivatives wrt. lam1 and lam2
    d_lam1 = q.dot(-1 * lam2/(lam1*(lam1 - lam2)) * p_th + lam_ratio * jacobi(log_theta, p_th_1, lam1))
    d_lam2 = q.dot(lam1 / (lam2 * (lam1 - lam2)) * p_th - lam_ratio * jacobi(log_theta, p_th_2, lam2))

    # Full gradient
    return np.append(lam_ratio*d_theta.flatten(), [d_lam1, d_lam2])
