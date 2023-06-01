import numpy as np
from scipy.linalg.blas import dcopy, dscal, daxpy
from math import factorial
import networkx as nx
import itertools


class bits_fixed_n:
    """
    Iterator over integers whose binary representation has a fixed number of 1s, in lexicographical order

    :param n: How many 1s there should be
    :param k: How many bits the integer should have
    """

    def __init__(self, n, k):
        self.v = int("1"*n, 2)
        self.stop_no = int("1"*n + "0"*(k-n), 2)
        self.stop = False

    def __iter__(self):
        return self

    def __next__(self):
        if self.stop:
            raise StopIteration
        if self.v == self.stop_no:
            self.stop = True
        t = (self.v | (self.v - 1)) + 1
        w = t | ((((t & -t)) // (self.v & (-self.v)) >> 1) - 1)
        self.v, w = w, self.v
        return w


class MetMHN:
    """
    This class represents the Mutual Hazard Network
    """

    def __init__(self, log_theta: np.array, tau1: float, tau2: float, events: list[str] = None, meta: dict = None):
        """
        :param log_theta: logarithmic values of the theta matrix representing the MHN
        :param events: (optional) list of strings containing the names of the events considered by the MHN
        :param meta: (optional) dictionary containing metadata for the MHN, e.g. parameters used to train the model
        """

        self.log_theta = log_theta
        self.events = events
        self.meta = meta
        self.tau1 = tau1
        self.tau2 = tau2

    def get_restr_diag_unpaired(self, state: np.array) -> np.array:
        """This returns the diagonal of the restricted rate matrix of the metMHN's Markov chain.

        Args:
            state (np.array): Binary unpaired state vector, dtype must be int32. This is the vector according
            to which state space restriction will be performed. Shape (n,) with n the number of events including
            seeding.

        Returns:
            np.array: Diagonal of the restricted rate matrix. Shape (2^k,) with k the number of 1s in state.
        """
        k = state.sum()
        nx = 1 << k
        n = self.log_theta.shape[0]
        diag = np.zeros(nx)
        subdiag = np.zeros(nx)

        for i in range(n):

            current_length = 1
            subdiag[0] = 1
            # compute the ith subdiagonal of Q
            for j in range(n):
                if state[j]:
                    exp_theta = np.exp(self.log_theta[i, j])
                    if i == j:
                        exp_theta *= -1
                        dscal(n=current_length, a=exp_theta, x=subdiag, incx=1)
                        dscal(n=current_length, a=0,
                              x=subdiag[current_length:], incx=1)
                    else:
                        dcopy(n=current_length, x=subdiag, incx=1,
                              y=subdiag[current_length:], incy=1)
                        dscal(n=current_length, a=exp_theta,
                              x=subdiag[current_length:], incx=1)

                    current_length *= 2

                elif i == j:
                    exp_theta = - np.exp(self.log_theta[i, j])
                    dscal(n=current_length, a=exp_theta, x=subdiag, incx=1)

            # add the subdiagonal to dg
            daxpy(n=nx, a=1, x=subdiag, incx=1, y=diag, incy=1)
        return diag

    def likeliest_order(self, state: np.array, tau: int) -> tuple[float, np.array]:
        """For a given state, this returns the order in which the events were most likely to accumulate.
        So far, this only works for an unpaired state which was observed at a timepoint that is an
        exponentially distributed random variable with rate tau1 or tau2. 

        Args:
            state (np.array): Binary unpaired state vector. Shape (n,) with n the number of events including
            seeding.
            tau (int): Which rate parameter to use, tau1 or tau2. Must be either 1 or 2.

        Returns:
            float: likelihood of the order.
            np.array likeliest order.
        """
        restr_diag = self.get_restr_diag_unpaired(state=state)
        log_theta = self.log_theta[state.astype(bool)][:, state.astype(bool)]
        if tau not in [1, 2]:
            raise ValueError("tau must be either 1 or 2.")
        tau = self.tau1 if tau == 1 else self.tau2

        k = state.sum()
        # {state: highest path probability to this state}
        A = {0: tau / (tau - restr_diag[0])}
        # {state: path with highest probability to this state}
        B = {0: []}
        for i in range(1, k+1):         # i is the number of events
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):  # all states with i events
                A_new[st] = -1
                state_events = np.array(
                    [i for i in range(k) if (1 << i) | st == st])  # events in state
                for e in state_events:
                    pre_st = st - (1 << e)  # pre state
                    # numerator of additional factor
                    num = np.exp(log_theta[e, state_events].sum())
                    if A[pre_st] * num > A_new[st]:
                        A_new[st] = A[pre_st] * num
                        B_new[st] = B[pre_st].copy()
                        B_new[st].append(e)
                A_new[st] /= (tau - restr_diag[st])
            A = A_new
            B = B_new
        i = (1 << k) - 1
        return (A[i], np.arange(self.log_theta.shape[0])[state.astype(bool)][B[i]])

    def m_likeliest_orders(self, state: np.array, tau: int, m: int) -> tuple[np.array, np.array]:
        """For a given state, this returns the m orders in which the events were most likely to accumulate.
        So far, this only works for an unpaired state which was observed at a timepoint that is an
        exponentially distributed random variable with rate tau1 or tau2. 

        Args:
            state (np.array): Binary unpaired state vector. Shape (n,) with n the number of events including
            seeding.
            tau (int): Which rate parameter to use, tau1 or tau2. Must be either 1 or 2.
            m (int): Number of orders to compute the likelihoods for.

        Returns:
            np.array(float): likelihoods of the m likeliest orders.
            np.array: m likeliest orders.
        """
        restr_diag = self.get_restr_diag_unpaired(state=state)
        log_theta = self.log_theta[state.astype(bool)][:, state.astype(bool)]
        if tau not in [1, 2]:
            raise ValueError("tau must be either 1 or 2.")
        tau = self.tau1 if tau == 1 else self.tau2

        k = state.sum()

        if k <= 1:
            return self.likeliest_order(state=state, tau=tau)

        # {state: highest path probability to this state}
        A = {0: np.array(tau / (tau - restr_diag[0]))}
        # {state: path with highest probability to this state}
        B = {0: np.empty(0, dtype=int)}
        for i in range(1, k+1):                     # i is the number of events
            _m = min(factorial(i - 1), m)
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):
                A_new[st] = np.zeros(i * _m)
                B_new[st] = np.zeros((i * _m, i), dtype=int)
                state_events = np.array(
                    [i for i in range(k) if 1 << i | st == st])  # events in state
                for j, e in enumerate(state_events):
                    pre_st = st - (1 << e)
                    # numerator of additional factor
                    num = np.exp(log_theta[e, state_events].sum())
                    A_new[st][j * _m: (j + 1) * _m] = num * A[pre_st]
                    B_new[st][j * _m: (j + 1) * _m, :-1] = B[pre_st]
                    B_new[st][j * _m: (j + 1) * _m, -1] = e
                sorting = A_new[st].argsort()[::-1][:m]
                A_new[st] = A_new[st][sorting]
                B_new[st] = B_new[st][sorting]
                A_new[st] /= (tau - restr_diag[st])
            A = A_new
            B = B_new
        i = (1 << k) - 1
        return (A[i], (np.arange(self.log_theta.shape[0])[state.astype(bool)])[B[i].flatten()].reshape(-1, k))

    def simulate(self, timepoint: int) -> tuple[np.array, float]:
        """This function simulates one sample according to the metMHN.

        Args:
            timepoint (int): At which timepoint to stop the simulation, t_1 or t_2. Must be either 1 or 2.

        Returns:
            np.array: Binary state of the sample.
            float: timepoint of observation. 
        """
        n = self.log_theta.shape[0]
        events = np.arange(n, dtype=int)
        state = np.zeros(n, dtype=int)
        order = list()
        if timepoint not in [1, 2]:
            raise ValueError("tau must be either 1 or 2.")
        t_obs = np.random.exponential(1 / self.tau1)
        if timepoint == 2:
            t_obs += np.random.exponential(1 / self.tau2)
        t = np.random.exponential(-1 /
                                  self.get_restr_diag_unpaired(state=state)[-1])
        while t < t_obs:
            probs = [np.exp(self.log_theta[e, events[state.astype(bool)]].sum(
            ) + self.log_theta[e, e]) for e in events[~state.astype(bool)]]
            ps = sum(probs)
            probs = [p/ps for p in probs]
            e = np.random.choice(events[~state.astype(bool)], p=probs)
            state[e] = 1
            order.append(e)
            if state.sum() == n:
                break
            t += np.random.exponential(-1 /
                                       self.get_restr_diag_unpaired(state=state)[-1])
        return order, t_obs

    def history_tree(self, orders) -> nx.Graph:
        """For a list of given orders of observations visualize them
        using a tree

        Args:
            orders (List[Tuple]): List of orders of observations in the 
            form of tuples

        Returns:
            nx.Graph: Graph object with optional nodekey "terminal".

        """
        g = nx.Graph()

        g.graph["observations"] = set(itertools.chain(*orders))

        for order in orders:
            for i in range(len(order)+1):
                g.add_node(order[:i])
            g.nodes[order]["terminal"] = True
            g.nodes[order]["event"] = order[-1]
            for i in range(len(order)):
                if (order[:i], order[:i+1]) in list(g.edges):
                    g.edges[(order[:i], order[:i+1])]["weight"] += 1
                else:
                    g.add_edge(order[:i], order[:i+1], weight=1)

        return g
