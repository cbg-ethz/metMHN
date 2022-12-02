import kronecker_vector as fss
import numpy as np


def jacobi(theta: np.array, b: np.array, lam: float, transp: bool = False, x: np.array = None) -> np.array:
    """
    This function computes (I-Q)x=b using jacobi iteration
    Args:
        theta (np.array): Logarithmic theta matrix
        b (np.array): State vector, resolvent is multiplied from the left to this vector
        x (np.aray): Helper vector, result of (lam*I-Q)^-1 b
        lam (float): rate of sampling
        transp (bool): Flag, indicates whether Q should be transposed
    Returns:
        np.array: (lam*I-Q)^-1 b
    """
    n = theta.shape[0]-1
    if x is None:
        x = np.ones(2**(2*n+1))/(2**(2*n+1))
    dg = fss.diag_q(theta) + lam
    
    for i in range(2*n+2):
        x = b + fss.qvec(theta, x, diag=False, transp=transp)
        x = x/dg
    return x


def diag_forward(theta: np.array, p: np.array) -> np.array:
    """
    calculates stationary distribution using diagnosis formalism
    Arguments:
        theta (np.array): logarithmic theta matrix
        p (np.array): Starting distribution
    Returns:
        np.array: stationary distribution
    """
    n = theta.shape[0]-1
    diagnosed = fss.diag_diagnosis(n)
    dg = fss.diag_q(theta)+diagnosed
    for i in range(2*n+1):
        np.divide(p, dg, out=p, where=dg != 0)
        p = fss.qvec(theta, p.copy(), False) + diagnosed*p
    return p


def likelihood(theta: np.array, pd: np.array, lam1: float, lam2: float) -> float:
    """
    This function computes the log likelihood score
    Args:
        theta (np.array): Logarithmic theta matrix
        pd (np.array): Vector holding empirical frequencies of genotypes in a dataset
        lam1 (float): Rate of first sampling
        lam2 (float): Rate of second sampling
    Returns:
         float: pd^T log(pTh)
    """
    n = np.shape[0] - 1
    p0 = np.zeros(2**(2*n+1))
    pTh = jacobi(theta, p0, lam2)-jacobi(theta, p0, lam1)
    pTh = lam1*lam2/(lam1-lam2)*pTh
    return pd.dot(np.log(pTh, out=np.zeros_like(pTh), where=pTh != 0))


def gradient(theta: np.array, pD: np.array, lam1: float, lam2: float, n: int, p0: np.array) -> np.array:
    """
    Calculates the gradient of the likelihood
    Args:
        theta (np.array): logarithmic theta matrix
        pD (np.array): Data vector holding frequencies of genotypes in D
        lam1 (float): Rate of first diagnosis
        lam2 (float): Rate of delta t
        n (int): Number of mutations
        p0 (np.array): Starting distribution
    Returns:
         np.array: g, gradient of likelihood
    """
    pTh1 = jacobi(theta, p0, lam1)
    pTh2 = jacobi(theta, p0, lam2)
    lam_ratio = lam1*lam2/(lam1-lam2)
    pTh = lam_ratio*(pTh2 - pTh1)
    q1 = jacobi(theta, np.divide(pD, pTh, out=np.zeros_like(pD), where=pTh != 0), lam1, True)
    q2 = jacobi(theta, np.divide(pD, pTh, out=np.zeros_like(pD), where=pTh != 0), lam2, True)
    return fss.q_partialQ_pth(theta, q1, pTh1, n) + fss.q_partialQ_pth(theta, q2, pTh2, n)
