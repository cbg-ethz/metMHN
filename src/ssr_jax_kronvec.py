import numpy as np
from jax import jit
import jax.numpy as jnp
from functools import partial

# @partial(jit, static_argnums=4)
# def kronvec_sync(log_theta: jnp.array, p: jnp.array, i: int, n: int, state: jnp.array, diag: bool = True, transpose: bool = False) -> jnp.array:
#     """This computes the restricted version of the product of the synchronized part of the ith Q summand
#     Q_i with a vector Q_i p.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         p (np.array): Vector to multiply with from the right. Length must equal the number of
#         nonzero entries in the state vector.
#         i (int): Index of the summand.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.
#         diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
#         transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

#     Returns:
#         np.array: Q_i p
#     """
#     # there are no non-diagonal entries if event i is not mutated in both prim and met
#     if not diag and sum(state[2 * i: 2 * i + 2]) != 2:
#         return np.zeros(p.size)

#     y = p.copy()

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]
#         if mut.sum() == 0:  # neither prim nor met
#             if i == j:
#                 y *= -np.exp(log_theta[i, i])
#         elif mut.sum() == 1:  # prim xor met
#             y = y.reshape((-1, 2), order="C")
#             y[:, 1] = 0
#             if i == j:
#                 y[:, 0] *= -np.exp(log_theta[i, i])
#             y = y.flatten(order="F")
#         else:  # both prim and met
#             y = y.reshape((-1, 4), order="C")
#             y[:, [1, 2]] = 0
#             if i == j:
#                 if not transpose:
#                     if diag:
#                         y[:, 0] *= -np.exp(log_theta[i, i])
#                         y[:, 3] = -1 * y[:, 0]
#                     else:  # no diag
#                         y[:, 3] = np.exp(log_theta[i, i]) * y[:, 0]
#                         y[:, 0] = 0
#                 else:  # transpose
#                     if diag:
#                         y[:, 0] = np.exp(log_theta[i, i]) * \
#                             (-y[:, 0] + y[:, 3])
#                     else:  # no diag
#                         y[:, 0] = np.exp(log_theta[i, i]) * y[:, 3]
#                     y[:, 3] = 0
#             else:
#                 y[:, 3] *= np.exp(log_theta[i, j])
#             y = y.flatten(order="F")
#     if state[-1] == 1:
#         y = y.reshape((-1, 2), order="C")
#         y[:, 1] = 0
#         y = y.flatten(order="F")

#     return y


# def kronvec_prim(log_theta: np.array, p: np.array, i: int, n: int, state: np.array, diag: bool = True, transpose: bool = False) -> np.array:
#     """This computes the restricted version of the product of the asynchronous primary tumour
#     part of the ith Q summand Q_i with a vector Q_i p.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         p (np.array): Vector to multiply with from the right. Length must equal the number of
#         nonzero entries in the state vector.
#         i (int): Index of the summand.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.
#         diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
#         transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

#     Returns:
#         np.array: Q_i p
#     """

#     # there are no non-diagonal entries if event i is not mutated in prim
#     if not diag and state[2 * i] == 0:
#         return np.zeros(p.size)

#     if state[-1] == 0:
#         return np.zeros(p.size)

#     y = p.copy()

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]
#         if mut.sum() == 0:
#             if i == j:
#                 y *= -np.exp(log_theta[i, i])
#         elif mut.sum() == 2:
#             y = y.reshape((-1, 4), order="C")
#             if i == j:
#                 theta = np.exp(log_theta[i, i])
#                 if not transpose:
#                     if diag:
#                         y[:, [0, 2]] *= -theta
#                         y[:, [1, 3]] = - y[:, [0, 2]]
#                     else:
#                         y[:, [1, 3]] = theta * y[:, [0, 2]]
#                         y[:, [0, 2]] = 0
#                 else:  # transpose
#                     if diag:
#                         y[:, 0] = theta * (-y[:, 0] + y[:, 1])
#                         y[:, 2] = theta * (-y[:, 2] + y[:, 3])
#                     else:
#                         y[:, [0, 2]] = theta * y[:, [1, 3]]
#                     y[:, [1, 3]] = 0

#             else:
#                 theta = np.exp(log_theta[i, j])
#                 y[:, 1] *= theta
#                 y[:, 3] *= theta
#             y = y.flatten(order="F")
#         else:  # mut.sum = 1
#             y = y.reshape((-1, 2), order="C")
#             if i == j:
#                 theta = np.exp(log_theta[i, i])
#                 if mut[0] == 1:  # prim mutated
#                     if not transpose:
#                         if diag:
#                             y[:, 0] *= -theta
#                             y[:, 1] = -y[:, 0]
#                         else:  # not diag:
#                             y[:, 1] = theta * y[:, 0]
#                             y[:, 0] = 0
#                     else:  # transpose
#                         if diag:
#                             y[:, 0] = theta * (-y[:, 0] + y[:, 1])
#                         else:
#                             y[:, 0] = theta * y[:, 1]
#                         y[:, 1] = 0
#                 else:  # met mutated
#                     y[:, [0, 1]] *= -theta
#             else:
#                 if mut[0] == 1:
#                     y[:, 1] *= np.exp(log_theta[i, j])
#             y = y.flatten(order="F")
#     y = y.reshape(-1, 2)
#     y[:, 0] = 0
#     y = y.flatten(order="F")

#     return y


# def kronvec_met(log_theta: np.array, p: np.array, i: int, n: int, state: np.array, diag: bool = True, transpose: bool = False) -> np.array:
#     """This computes the restricted version of the product of the asynchronous metastasis
#     part of the ith Q summand Q_i with a vector Q_i p.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         p (np.array): Vector to multiply with from the right. Length must equal the number of
#         nonzero entries in the state vector.
#         i (int): Index of the summand.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.
#         diag (bool, optional): Whether to use the diagonal of Q_i (and not set it to 0). Defaults to True.
#         transpose (bool, optional): Whether to transpose Q_i before multiplying. Defaults to False.

#     Returns:
#         np.array: Q_i p
#     """

#     # there are no non-diagonal entries if event i is not mutated in met
#     if not diag and state[2 * i + 1] == 0:
#         return np.zeros(p.size)

#     if state[-1] == 0:
#         return np.zeros(p.size)

#     y = p.copy()

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]
#         if mut.sum() == 0:
#             if i == j:
#                 y *= -np.exp(log_theta[i, i])
#         elif mut.sum() == 2:
#             y = y.reshape((-1, 4), order="C")
#             if i == j:
#                 theta = np.exp(log_theta[i, i])
#                 if not transpose:
#                     if diag:
#                         y[:, [0, 1]] *= -theta
#                         y[:, [2, 3]] = -1 * y[:, [0, 1]]
#                     else:
#                         y[:, [2, 3]] = theta * y[:, [0, 1]]
#                         y[:, [0, 1]] = 0
#                 else:  # transpose
#                     if diag:
#                         y[:, 0] = theta * (-y[:, 0] + y[:, 2])
#                         y[:, 1] = theta * (-y[:, 1] + y[:, 3])
#                     else:  # no diag
#                         y[:, [0, 1]] = theta * y[:, [2, 3]]
#                     y[:, [2, 3]] = 0
#             else:
#                 theta = np.exp(log_theta[i, j])
#                 y[:, [2, 3]] *= theta
#             y = y.flatten(order="F")
#         else:
#             y = y.reshape((-1, 2), order="C")
#             if i == j:
#                 theta = np.exp(log_theta[i, i])
#                 if mut[1] == 1:  # met mutated
#                     if not transpose:
#                         y[:, 0] *= -theta
#                         y[:, 1] = -y[:, 0]
#                         if not diag:
#                             y[:, 0] = 0
#                     else:  # transpose
#                         if diag:
#                             y[:, 0] = theta * (-y[:, 0] + y[:, 1])
#                         else:  # no diag
#                             y[:, 0] = theta * y[:, 1]
#                         y[:, 1] = 0
#                 else:  # prim mutated
#                     y[:, [0, 1]] *= -theta
#             else:
#                 if mut[1] == 1:
#                     y[:, 1] *= np.exp(log_theta[i, j])
#             y = y.flatten(order="F")
#     y = y.reshape(-1, 2)
#     y[:, 0] = 0
#     y[:, 1] *= np.exp(log_theta[i, -1])
#     y = y.flatten(order="F")

#     return y


# def kronvec_seed(log_theta: np.array, p: np.array, n: int, state: np.array, diag: bool = True, transpose: bool = False) -> np.array:
#     """This computes the restricted version of the product of the seeding summand of Q with a vector Q_M p.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         p (np.array): Vector to multiply with from the right. Length must equal the number of
#         nonzero entries in the state vector.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.
#         diag (bool, optional): Whether to use the diagonal of Q_M (and not set it to 0). Defaults to True.
#         transpose (bool, optional): Whether to transpose Q_M before multiplying. Defaults to False.

#     Returns:
#         np.array: Q_i p
#     """

#     # there are no non-diagonal entries if met has not seeded
#     if not diag and state[-1] == 0:
#         return np.zeros(p.size)

#     y = p.copy()

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]
#         if mut.sum() == 2:
#             y = y.reshape((-1, 4), order="C")
#             y[:, [1, 2]] = 0
#             y[:, 3] *= np.exp(log_theta[-1, j])
#             y = y.flatten(order="F")
#         elif mut.sum() == 1:
#             y = y.reshape((-1, 2), order="C")
#             y[:, 1] = 0
#             y = y.flatten(order="F")
#     if state[-1] == 1:
#         y = y.reshape((-1, 2), order="C")
#         theta = np.exp(log_theta[-1, -1])
#         if not transpose:
#             y[:, 1] = theta * y[:, 0]
#             if diag:
#                 y[:, 0] = -y[:, 1]
#             else:
#                 y[:, 0] = 0
#         else:  # transpose
#             if diag:
#                 y[:, 0] = theta * (-y[:, 0] + y[:, 1])
#             else:
#                 y[:, 0] = theta * y[:, 1]
#             y[:, 1] = 0
#         y = y.flatten(order="F")
#     else:
#         y *= -np.exp(log_theta[-1, -1])

#     return y


# def kronvec(log_theta: np.array, p: np.array, n: int, state: np.array, diag: bool = True, transpose: bool = False) -> np.array:
#     """This computes the restricted version of the product of the rate matrix Q with a vector Q p.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         p (np.array): Vector to multiply with from the right. Length must equal the number of
#         nonzero entries in the state vector.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.
#         diag (bool, optional): Whether to use the diagonal of Q (and not set it to 0). Defaults to True.
#         transpose (bool, optional): Whether to transpose Q before multiplying. Defaults to False.

#     Returns:
#         np.array: Q p
#     """
#     y = np.zeros(shape=2**sum(state))
#     for i in range(n):
#         y += kronvec_sync(log_theta=log_theta, p=p, i=i,
#                           n=n, state=state, diag=diag, transpose=transpose)
#         y += kronvec_prim(log_theta=log_theta, p=p, i=i,
#                           n=n, state=state, diag=diag, transpose=transpose)
#         y += kronvec_met(log_theta=log_theta, p=p, i=i,
#                          n=n, state=state, diag=diag, transpose=transpose)
#     y += kronvec_seed(log_theta=log_theta, p=p, n=n,
#                       state=state, diag=diag, transpose=transpose)

#     return y


# def kron_sync_diag(log_theta: np.array, i: int, n: int, state: np.array) -> np.array:
#     """This computes the diagonal of the synchronized part of the ith Q summand Q_i.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         nonzero entries in the state vector.
#         i (int): Index of the summand.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.

#     Returns:
#         np.array: diag(Q_i_sync)
#     """
#     diag = np.ones(2 ** sum(state))

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]

#         if i == j:
#             if sum(mut) == 0:
#                 diag *= -np.exp(log_theta[i, i])
#             elif sum(mut) == 1:
#                 diag = diag.reshape((-1, 2), order="C")
#                 diag[:, 0] *= -np.exp(log_theta[i, i])
#                 diag[:, 1] = 0
#                 diag = diag.flatten(order="F")
#             else:
#                 diag = diag.reshape((-1, 4), order="C")
#                 diag[:, 0] *= -np.exp(log_theta[i, i])
#                 diag[:, [1, 2, 3]] = 0
#                 diag = diag.flatten(order="F")
#         else:
#             if sum(mut) == 1:
#                 diag = diag.reshape((-1, 2), order="C")
#                 diag[:, 1] = 0
#                 diag = diag.flatten(order="F")
#             elif sum(mut) == 2:
#                 diag = diag.reshape((-1, 4), order="C")
#                 diag[:, [1, 2]] = 0
#                 diag[:, 3] *= np.exp(log_theta[i, j])
#                 diag = diag.flatten(order="F")
#     if state[-1] == 1:
#         diag = diag.reshape((-1, 2), order="C")
#         diag[:, 1] = 0
#         diag = diag.flatten(order="F")

#     return diag


# def kron_prim_diag(log_theta: np.array, i: int, n: int, state: np.array) -> np.array:
#     """This computes the diagonal of the asynchronous primary tumour part of the ith
#     Q summand Q_i.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         nonzero entries in the state vector.
#         i (int): Index of the summand.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.

#     Returns:
#         np.array: diag(Q_i_prim)
#     """

#     if state[-1] == 0:
#         return np.zeros(2 ** sum(state))

#     diag = np.ones(2 ** sum(state))

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]

#         if i == j:
#             if sum(mut) == 0:
#                 diag *= -np.exp(log_theta[i, i])
#             elif sum(mut) == 2:
#                 diag = diag.reshape((-1, 4), order="C")
#                 diag[:, [0, 2]] *= -np.exp(log_theta[i, i])
#                 diag[:, [1, 3]] = 0
#                 diag = diag.flatten(order="F")
#             elif mut[0] == 1:  # prim mutated
#                 diag = diag.reshape((-1, 2), order="C")
#                 diag[:, 0] *= -np.exp(log_theta[i, i])
#                 diag[:, 1] = 0
#                 diag = diag.flatten(order="F")
#             else:  # met mutated
#                 diag = diag.reshape((-1, 2), order="C")
#                 diag *= -np.exp(log_theta[i, i])
#                 diag = diag.flatten(order="F")
#         else:
#             if sum(mut) == 1:
#                 if mut[0] == 1:  # prim mutated
#                     diag = diag.reshape((-1, 2), order="C")
#                     diag[:, 1] *= np.exp(log_theta[i, j])
#                     diag = diag.flatten(order="F")
#                 else:  # met mutated
#                     diag = diag.reshape((-1, 2), order="C").flatten(order="F")
#             elif sum(mut) == 2:
#                 diag = diag.reshape((-1, 4), order="C")
#                 diag[:, [1, 3]] *= np.exp(log_theta[i, j])
#                 diag = diag.flatten(order="F")
#     diag = diag.reshape((-1, 2), order="C")
#     diag[:, 0] = 0
#     diag = diag.flatten(order="F")

#     return diag


# def kron_met_diag(log_theta: np.array, i: int, n: int, state: np.array) -> np.array:
#     """This computes the diagonal of the asynchronous metastasis part of the ith
#     Q summand Q_i.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         nonzero entries in the state vector.
#         i (int): Index of the summand.
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.

#     Returns:
#         np.array: diag(Q_i_met)
#     """

#     if state[-1] == 0:
#         return np.zeros(2 ** sum(state))

#     diag = np.ones(2 ** sum(state))

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]

#         if i == j:
#             if sum(mut) == 0:
#                 diag *= -np.exp(log_theta[i, i])
#             elif sum(mut) == 2:
#                 diag = diag.reshape((-1, 4), order="C")
#                 diag[:, [0, 1]] *= -np.exp(log_theta[i, i])
#                 diag[:, [2, 3]] = 0
#                 diag = diag.flatten(order="F")
#             elif mut[0] == 1:  # prim mutated
#                 diag = diag.reshape((-1, 2), order="C")
#                 diag *= -np.exp(log_theta[i, i])
#                 diag = diag.flatten(order="F")
#             else:  # met mutated
#                 diag = diag.reshape((-1, 2), order="C")
#                 diag[:, 0] *= -np.exp(log_theta[i, i])
#                 diag[:, 1] = 0
#                 diag = diag.flatten(order="F")
#         else:
#             if sum(mut) == 1:
#                 diag = diag.reshape((-1, 2), order="C")
#                 if mut[1] == 1:  # met mutated
#                     diag[:, 1] *= np.exp(log_theta[i, j])
#                 diag = diag.flatten(order="F")
#             elif sum(mut) == 2:
#                 diag = diag.reshape((-1, 4), order="C")
#                 diag[:, [2, 3]] *= np.exp(log_theta[i, j])
#                 diag = diag.flatten(order="F")
#     diag = diag.reshape((-1, 2), order="C")
#     diag[:, 0] = 0
#     diag[:, 1] *= np.exp(log_theta[i, -1])
#     diag = diag.flatten(order="F")

#     return diag


# def kron_seed_diag(log_theta: np.array, n: int, state: np.array) -> np.array:
#     """This computes the diagonal of the seeding summand of Q.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.

#     Returns:
#         np.array: diag(Q_seed)
#     """

#     diag = np.ones(2 ** sum(state))

#     for j in range(n):

#         mut = state[2 * j: 2 * j + 2]

#         if sum(mut) == 1:
#             diag = diag.reshape((-1, 2), order="C")
#             diag[:, 1] = 0
#             diag = diag.flatten(order="F")
#         elif sum(mut) == 2:
#             diag = diag.reshape((-1, 4), order="C")
#             diag[:, [1, 2]] = 0
#             diag[:, 3] *= np.exp(log_theta[-1, j])
#             diag = diag.flatten(order="F")
#     if state[-1] == 1:
#         diag = diag.reshape((-1, 2), order="C")
#         diag[:, 0] *= -np.exp(log_theta[-1, -1])
#         diag[:, 1] = 0
#         diag = diag.flatten(order="F")
#     else:
#         diag *= -np.exp(log_theta[-1, -1])

#     return diag


# def kron_diag(log_theta: np.array, n: int, state: np.array) -> np.array:
#     """This computes diagonal of the rate matrix Q.

#     Args:
#         log_theta (np.array): Log values of the theta matrix
#         n (int): Total number of events in the MHN.
#         state (np.array): Binary state vector, representing the current sample's events.

#     Returns:
#         np.array: diag(Q)
#     """
#     y = np.zeros(shape=2**sum(state))
#     for i in range(n):
#         y += kron_sync_diag(log_theta=log_theta, i=i,
#                             n=n, state=state)
#         y += kron_prim_diag(log_theta=log_theta, i=i,
#                             n=n, state=state)
#         y += kron_met_diag(log_theta=log_theta, i=i,
#                            n=n, state=state)
#     y += kron_seed_diag(log_theta=log_theta, n=n, state=state)

#     return y
