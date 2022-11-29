import kronecker_vector as fss
import numpy as np


def jacobi(theta: np.array, b: np.array, lam: float, x: np.array = None) -> np.array:
    """
    This function computes (I-Q)x=b using jacobi iteration
    Args:
        theta (np.array): Logarithmic theta matrix
        b (np.array): State vector, resolvent is multiplied from the left to this vector
        x (np.aray): Helper vector, result of (lam*I-Q)^-1 b
        lam (float): rate of sampling
    Returns:
        np.array: (lam*I-Q)^-1 b
    """
    n = theta.shape[0]-1
    if x is None:
        x = np.ones(2**(2*n+1))/(2**(2*n+1))
    dg = fss.diag_q(theta) + lam
    
    for i in range(2*n+2):
        x = b + fss.qvec(theta, x, diag=False)
        x = x/dg
    return x


def likelihood(theta: np.array, pd: np.array, lam1: float, lam2: float) -> float:
    """
    This function computes the log likelihood score
    Args:
        theta (np.array): Logarithmic theta matrix
        pd (np.array): Vector holding empirical frequencies of genotypes in a dataset
        lam1 (float): Rate of first sampling
        lam2 (float): Rate of second sampling
    Returns:
         float: pd^T pTh
    """
    n = np.shape[0] - 1
    p0 = np.zeros(2**(2*n+1))
    pTh = jacobi(theta, p0, lam2)-jacobi(theta, p0, lam1)
    pTh = lam1*lam2/(lam1-lam2)*pTh
    return pd.dot(np.log(pTh))
