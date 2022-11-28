import numpy as np

from ssr_kronecker_vector import kronvec_sync, kronvec_met, kronvec_prim, kronvec_seed


def res_x_partial_Q_y(log_theta: np.array, x: np.array, y: np.array, state: np.array) -> np.array:
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
    n = log_theta.shape[0]

    z = np.zeros(shape=(2*n + 1, 2*n + 1))

    for i in range(n):
        z_sync = x * kronvec_sync(log_theta=log_theta,
                                  p=y, i=i, n=n, state=state)
        z_prim = x * kronvec_prim(log_theta=log_theta,
                                  p=y, i=i, n=n, state=state)
        z_met = x * kronvec_met(log_theta=log_theta,
                                p=y, i=i, n=n, state=state)

        z[i, -1] = sum(z_met)

        for j in range(n):
            current = state[2*j: 2*j + 2]

            if sum(current) == 0:
                if i == j:
                    z[i, j] = sum(
                        sum(z_sync),
                        sum(z_prim),
                        sum(z_met)
                    )

            elif sum(current) == 2:
                z_sync = z_sync.reshape((-1, 4), order="C")
                z_prim = z_prim.reshape((-1, 4), order="C")
                z_met = z_met.reshape((-1, 4), order="C")

                z[i, j] = sum(
                    sum(z_sync[:, 3]),
                    sum(z_prim[:, [1, 3]]),
                    sum(z_met[:, [2, 3]])
                )

                if i == j:
                    z[i, j] += sum(
                        sum(z_sync[:, 0]),
                        sum(z_prim[:, [0, 2]]),
                        sum(z_met[:, [0, 1]])
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
                        z[i, j] = sum(z_met[:, 1])
                    else:  # prim mutated
                        z[i, j] = sum(z_prim[:, 1])
                else:
                    z[i, j] = sum(
                        sum(z_sync[:, 0]),
                        sum(z_prim),
                        sum(z_met)
                    )
                z_sync = z_sync.flatten(order="F")
                z_prim = z_prim.flatten(order="F")
                z_met = z_met.flatten(order="F")

    z_seed = x * kronvec_seed(log_theta=log_theta, p=y, n=n, state=state)

    z[-1, -1] = sum(z_seed)

    for j in range(n):
        current = state[2*j: 2*j + 2]

        if sum(current) == 2:
            z_seed = z_seed.reshape((-1, 4), order="C")

            z[-1, j] = sum(z_seed[:, 3])

            z_seed = z_seed.flatten(order="F")

        elif sum(current) == 1:
            z_seed = z_seed.reshape((-1, 2), order="C").flatten(order="F")

    return z
