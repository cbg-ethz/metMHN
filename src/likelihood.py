import kronecker_vector as fss
import numpy as np
import Utilityfunctions as utils


def jacobi(log_theta: np.array, b: np.array, lam: float, transp: bool = False, x: np.array = None) -> np.array:
    """
    This function computes (I-Q)x=b using jacobi iteration
    Args:
        log_theta (np.array): Logarithmic theta matrix
        b (np.array): State vector, resolvent is multiplied from the left to this vector
        x (np.aray): Helper vector, result of (lam*I-Q)^-1 b
        lam (float): rate of sampling
        transp (bool): Flag, indicates whether Q should be transposed
    Returns:
        np.array: (lam*I-Q)^-1 b
    """
    n = log_theta.shape[0]-1
    if x is None:
        x = np.ones(2**(2*n+1))/(2**(2*n+1))
    dg = fss.diag_q(log_theta) + lam
    
    for _ in range(2*n+2):
        x = b + fss.qvec(log_theta, x, diag=False, transp=transp)
        x = x/dg
    return x


def generate_pths(log_theta: np.array, p0: np.array, lam1: float, lam2: float) -> np.array:
    """
    calculates the two component sof the pth vector
    Args:
        log_theta (np.array): logarithmic theta matrix
        p0 (np.array): starting distribution
        lam1 (float): rate of first diagnosis
        lam2 (float): rate of second diagnosis

    Returns:
         np.array: pth1, pth2
    """
    return jacobi(log_theta, p0, lam1), jacobi(log_theta, p0, lam2)


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



def likelihood(log_theta: np.array, pd: np.array, lam1: float, lam2: float,
               pTh1_space: np.array, pTh2_space: np.array) -> float:
    """
    This function computes the log likelihood score
    Args:
        log_theta (np.array): Logarithmic theta matrix
        pd (np.array): Vector holding empirical frequencies of genotypes in a dataset
        lam1 (float): Rate of first sampling
        lam2 (float): Rate of second sampling
        pTh1_space (np.array): Second term needed to calculate pth
        pTh2_space (np.array): First term need to calculate pth
    Returns:
         float: pd^T log(pTh)
    """
    n = log_theta.shape[0] - 1
    p0 = np.zeros(2 ** (2 * n + 1))
    p0[0] = 1
    pTh1_space, pTh2_space = generate_pths(log_theta, p0, lam1, lam2)
    pTh = lam1*lam2/(lam1-lam2)*(pTh2_space-pTh1_space)
    return pd.dot(np.log(pTh, out=np.zeros_like(pTh), where=pTh != 0))



def gradient(log_theta: np.array, pD: np.array, lam1: float, lam2: float, n: int, p0: np.array,
             pTh1_space: np.array = None, pTh2_space: np.array = None) -> np.array:
    """
    Calculates the gradient of the likelihood
    Args:
        log_theta (np.array): logarithmic theta matrix
        pD (np.array): Data vector holding frequencies of genotypes in D
        lam1 (float): Rate of first diagnosis
        lam2 (float): Rate of delta t
        n (int): Number of mutations
        p0 (np.array): Starting distribution
        pTh1_space (np.array): First component of pTh, updated in the previous call to likelihood
        pTh2_space (np.array): Second component of pTh, updated in the previous call to likelihood
    Returns:
         np.array: g, gradient of likelihood
    """
    # Build p_theta
    if pTh1_space is None and pTh2_space is None:
        pTh1, pTh2 = generate_pths(log_theta, p0, lam1, lam2)
    else:
        pTh1 = pTh1_space
        pTh2 = pTh2_space
    lam_ratio = lam1*lam2/(lam1-lam2)
    pTh = lam_ratio*(pTh2 - pTh1)

    # Build the vector to multiply from the left to dQ/d th
    q = np.divide(pD, pTh, out=np.zeros_like(pD), where=pTh != 0)
    q1 = jacobi(log_theta, q, lam1, True)
    q2 = jacobi(log_theta, q, lam2, True)
    d_theta = fss.q_partialQ_pth(log_theta, q2, pTh2, n) - fss.q_partialQ_pth(log_theta, q1, pTh1, n)

    # Derivatives wrt. lam1 and lam2
    d_lam1 = np.dot(q, (-(lam1-lam2) / lam2 * pTh + lam_ratio * jacobi(log_theta, pTh1, lam1)))
    d_lam2 = np.dot(q, lam1 / (lam2*(lam1-lam2)) * pTh - lam_ratio * jacobi(log_theta, pTh2, lam2))
    return lam_ratio*d_theta
