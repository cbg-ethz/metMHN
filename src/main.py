from likelihood import likelihood, jacobi
import numpy as np
import kronecker_vector as kv
if __name__=="__main__":
    theta = np.array([i for i in range(1,10)]).reshape((3,3))
    p0 = np.zeros(2**5)
    p0[0] = 1.
    print(kv.qvec(theta, p0, False))
    pth = jacobi(theta, p0)
    print(pth)
