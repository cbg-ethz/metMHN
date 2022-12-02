import likelihood as fss
import numpy as np
import kronecker_vector as kv
import Utilityfunctions as ut
import explicit_statetespace as essp


if __name__=="__main__":
    n = 3
    theta = ut.random_theta(n, 0.4)
    Q = essp.build_q(theta)
    p0 = np.zeros(2**(2*n+1))
    p0[0] = 1

    # Test Q p
    assert(np.allclose(
        Q @ p0,
        kv.qvec(theta, p0, True, False)
        )
    )

    # Test Q^T p
    assert(np.allclose(
        Q.T @ p0,
        kv.qvec(theta, p0, True, True)
        )
    )


    # Test (Q - diag(Q)) p
    q_diag = np.diag(np.diag(Q))
    assert(np.allclose(
        (Q - q_diag)  @ p0,
         kv.qvec(theta, p0, False, False)
        )
    )


    # Test diag(Q) p
    assert(np.allclose(
        q_diag @ p0,
        (-1)*kv.diag_q(theta) * p0
        )
    )


    # Test (lambda*I-Q)^(-1) p
    lam = np.random.exponential(10, 1)
    resolvent = lam*np.eye(2**(2*n+1))-Q
    assert(np.allclose(
        np.linalg.solve(resolvent, p0),
        fss.jacobi(theta, p0, lam)
        )
    )


    # Test (lambda*I-Q)^T^(-1) p
    assert(np.allclose(
        np.linalg.solve(resolvent.T, p0),
        fss.jacobi(theta, p0, lam, transp=True)
        )
    )


    # Test q \partial Q \partial theta_ij p
    p = fss.jacobi(theta, p0, lam)
    q = fss.jacobi(theta, p, lam, transp=True)
    theta_test = np.zeros_like(theta)
    assert(np.allclose(
        essp.build_q_grad_p(theta_test, q, p),
        kv.q_partialQ_pth(theta_test, q, p, n)
        )
    )

