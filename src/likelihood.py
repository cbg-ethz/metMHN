import kronecker_vector as fss
import numpy as np

def jacobi(theta, b, x=None):
    n = theta.shape[0]-1
    if x is None:
        x = np.ones(2**(2*n+1))/(2**(2*n+1))
    dg = fss.diag_q(theta) + 1
    for i in range(2*n+2):
        inc = fss.qvec(theta, x, diag=False)
        x = b + inc
        x = x/dg
    return x

def likelihood(theta, pd):
    n = np.shape[0] - 1
    p0 = np.zeros(2**(2*n+1))
    pTh = jacobi(theta, p0)
    return pd.dot(np.log(pTh))
