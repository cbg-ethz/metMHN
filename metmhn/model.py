from MetaMHN.perf import order_to_int, int_to_order, append_to_int_order
from collections import deque
from np.kronvec import kron_diag as get_diag_paired
from scipy.linalg.blas import dcopy, dscal, daxpy
import numpy as np

append_to_int_order = np.vectorize(
    append_to_int_order, excluded=["numbers", "new_event"])


def tuple_max(x: np.array, y: np.array) -> tuple[np.array]:
    """If given two values x_i and y_i for each i, we want to find i
    s.t. for all non-negative linear factors a and b we have 
    ax_i + by_i >= ax_j + by_j
    There will in general not be a unique i that satisfies this,
    therefore we just return all possible candidates i that could
    fulfill this for the right values a and b. 

    Args:
        x (np.array): x
        y (np.array): y

    Returns:
        tuple[np.array, np.array]: Vectors that only contain the
        maximizing candidates. 
    """
    x.sort(order="order")
    y.sort(order="order")
    indices = (x["prob"] >= x["prob"][np.argmax(y["prob"])]) \
        & (y["prob"] >= y["prob"][np.argmax(x["prob"])])
    return x[indices], y[indices]


def reachable(bin_state: int, n: int, state: np.array) -> bool:
    """This function checks for a binary state in state space restricted
    form whether it can be actually reached by an MHN

    Args:
        bin_state (int): Binary state in state space restriction w.r.t.
        state.
        n (int): Number of events (excluding metastasis).
        state (np.array): Binary state vector w.r.t. which there is
        restricted to.

    Returns:
        bool: Whether bin_state is a reachable state.
    """
    # transform the restricted bin_state to a full state
    full_bin = 0

    # iterate over the entries of state and pad bin_state with 0s
    for bit in state:
        full_bin <<= 1
        if bit:
            full_bin |= (1 & bin_state)
            bin_state >>= 1
    # reverse bitstring
    full_bin = int(f"{full_bin:0{2*n + 1}b}"[::-1], 2)
    # if seeding has happened
    if full_bin & (1 << (2 * n)):
        return True
    else:
        # check whether pt and met state agree
        return not (full_bin ^ (full_bin >> 1)) & int("01"*n, base=2)


def met(bin_state: int, pt) -> int:
    """Get the metastasis part of a state, both binary and state-space
    restricted to some state vector x.

    Args:
        bin_state (int): state, binary
        pt (numpy.array, dtype=bool): Boolean array of length k, where
        k = x.sum(). Its entries encode whether the nonzero entries of x
        belongs to the pt or the met part.  

    Returns:
        int: Inter
    """
    return int(
        "".join(
            i for i, pt_ev in zip(bin(bin_state)[2:], pt[::-1]) if not pt_ev),
        base=2)


def get_combos(order: np.array, n: int) -> list[tuple[np.array]]:
    """For a order of events in PT and Met, there are usually multiple
    timepoints at which the first observation could have happened.
    This function returns all possible combinations of pre- and past-
    first-observation events.

    Args:
        order (np.array): Sequence of PT and Met events as integers
        n (int): number of events in total

    Returns:
        list[tuple[np.array]]: List of combinations of pre- and past-
    first-observation events.
    """
    seeding = np.where(order == 2*n)[0]
    combos = list()
    for i in range(len(order)-seeding[0]):
        combos.append(np.split(order, [len(order)-i]))
        if not order[-i - 1] % 2:
            break
    return combos


class bits_fixed_n:
    """
    Iterator over integers whose binary representation has a fixed
    number of 1s, in lexicographical order.
    From https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

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

    def __init__(self, log_theta: np.array, obs1: np.array, obs2: np.array,
                 events: list[str] = None, meta: dict = None):
        """
        :param log_theta: logarithmic values of the theta matrix
        representing the MHN
        :param events: (optional) list of strings containing the names
        of the events considered by the MHN
        :param meta: (optional) dictionary containing metadata for the
        MHN, e.g. parameters used to train the model
        """

        self.log_theta = log_theta
        self.events = events
        self.meta = meta
        self.obs1 = obs1
        self.obs2 = obs2
        self.n = log_theta.shape[1] - 1

    def get_diag_unpaired(self, state: np.array) -> np.array:
        """This returns the diagonal of the restricted rate matrix of
        the metMHN's Markov chain.

        Args:
            state (np.array): Binary unpaired state vector, dtype must
            be int32. This is the vector according to which state space
            restriction will be performed. Shape (n,) with n the number
            of events including seeding.

        Returns:
            np.array: Diagonal of the restricted rate matrix. Shape
            (2^k,) with k the number of 1s in state.
        """
        k = state.sum()
        nx = 1 << k
        diag = np.zeros(nx)
        subdiag = np.zeros(nx)

        for i in range(self.n):

            current_length = 1
            subdiag[0] = 1
            # compute the ith subdiagonal of Q
            for j in range(self.n):
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

    def likeliest_order_paired(
        self, state: np.array, verbose: bool = False
    ) -> tuple[tuple[int, ...], float]:

        k = state.sum()
        if not reachable(
                bin_state=int("1" * k, base=2), n=self.n, state=state):
            raise ValueError("This state is not reachable by mhn.")

        # whether active events belong to pt
        pt = np.nonzero(state)[0] % 2 == 0
        pt[-1] = False

        # get the numbers of events
        events = np.nonzero(state)[0] // 2

        # get the positions of the pt 1s
        pt_events = np.nonzero(pt.astype(int))[0]
        diag_paired = get_diag_paired(
            log_theta=self.log_theta, n=self.n, state=state)
        diag_unpaired = self.get_diag_unpaired(
            state=np.concatenate([state[1::2], state[-1::]]))

        # In A1[i][state][order], the probabilities to reach a state
        # with a given order are stored. Here, i can be 0, 1 or 2, where
        # A1[2] holds the states that have n_events events and A1[1] and
        # A1[1] hold the ones with 1 and 2 events less, respectively.

        order_type = [("order", int), ("prob", float)]
        # get there with tau1
        A1 = deque([
            dict(),
            {0: np.array(
                [(0, 1 / (1 - diag_paired[0]))],
                dtype=order_type)}])
        # get there with tau2
        A2 = deque([dict()])

        for n_events in range(1, k + 1):
            # create dicts to hold the probs and orders to reach states
            # with n_events events
            A1.append(dict())
            A2.append(dict())

            # iterate over all states with n_events events
            for current_state in bits_fixed_n(n=n_events, k=k):

                if verbose:
                    print(
                        f"{n_events:3}/{k:3}, {len(A1[2]):10}, {sum(len(x) for x in A1[2].values()):10}", end="\r")

                # check whether state is reachable
                if not reachable(bin_state=current_state, state=state, n=self.n):
                    continue

                # whether seeding has happened
                if current_state & (1 << (k - 1)):

                    # get the positions of the 1s
                    state_events = [
                        i for i in range(k)
                        if (1 << i) | current_state == current_state]

                    # Does the pt part fit the observation?
                    pt_terminal = np.isin(pt_events, state_events).all()

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A1[2][current_state] = np.empty([0], dtype=order_type)

                    if pt_terminal:

                        obs1 = np.exp(self.obs1[
                            events[state_events][pt[state_events]]].sum())
                        obs2 = np.exp(self.obs2[
                            events[state_events][~pt[state_events]]].sum())

                        denom2 = 1 / \
                            (obs2 - diag_unpaired[met(current_state, pt)])
                        start_factor = obs1 * denom2

                    # iterate over all previous states
                    for pre_state, pre_orders1 in A1[1].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not (current_state | pre_state == current_state):
                            continue

                        # get the position of the new 1
                        new_event = bin(current_state ^ pre_state)[
                            :1:-1].index("1")

                        # whether new event is pt
                        if pt[new_event]:  # new event is pt
                            denom1 = 1 / (np.exp(self.obs1[
                                events[state_events][pt[state_events]]].sum())
                                - diag_paired[current_state])
                            num = np.exp(self.log_theta[
                                events[new_event],
                                events[state_events][pt[state_events]]].sum())
                        else:  # new event is met
                            denom1 = 1 / (np.exp(self.obs1[
                                events[state_events][pt[state_events]]].sum())
                                - diag_paired[current_state])
                            num = np.exp(self.log_theta[
                                events[new_event],
                                events[state_events][~pt[state_events]]].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders1.copy()
                        new_orders["prob"] *= (num * denom1)
                        new_orders["order"] = append_to_int_order(
                            new_orders["order"],
                            numbers=[
                                e for e in state_events if e != new_event],
                            new_event=new_event)
                        A1[2][current_state] = np.append(
                            A1[2][current_state],
                            new_orders
                        )

                    if pt_terminal:

                        # all probabilities to reach with tau2 here are
                        # at least the ones to reach with tau1 times the
                        # start factor
                        A2[1][current_state] = A1[2][current_state].copy()
                        A2[1][current_state]["prob"] *= start_factor

                        for pre_state, pre_orders1 in A1[1].items():
                            # if current_state is pt terminal, it is
                            # possible that the same holds for a
                            # prestate.
                            # Then we need to calculate how we can get
                            # from pre_state to current_state

                            # Skip pre_state if it is not a subset of
                            # current_state
                            if not (current_state | pre_state ==
                                    current_state):
                                continue

                            # if pre_state was pt_terminal
                            if pre_state in A2[0]:

                                # get the position of the new 1
                                new_event = bin(current_state ^ pre_state)[
                                    :1:-1].index("1")

                                # Get the numerator
                                num = np.exp(self.log_theta[
                                    events[new_event],
                                    events[state_events][
                                        ~pt[state_events]]].sum())

                                # get the orders that are coming from
                                # pre_state
                                new_orders = append_to_int_order(
                                    my_int=A2[0][pre_state]["order"],
                                    numbers=[e for e in state_events
                                             if e != new_event],
                                    new_event=new_event)

                                # add to the probs of these orders the
                                # prob to reach current_state with tau2
                                A2[1][current_state]["prob"][
                                    np.isin(A2[1][current_state]["order"],
                                            new_orders)] \
                                    += A2[0][pre_state]["prob"] \
                                    * num * denom2

                    # Now I have the two dicts A1[2][current_state] and
                    # A2[1][current_state] with possible paths to get to
                    # current_state. I can kick out some of them,
                    # because I am only interested in those orders that
                    # stand a chance to be maximal
                    if pt_terminal:
                        A1[2][current_state], A2[1][current_state] = \
                            tuple_max(
                                A1[2][current_state],
                                A2[1][current_state])

                else:  # seeding has not happened yet
                    # get positions of 1s
                    state_events = [i for i in range(k) if (
                        1 << i) | current_state == current_state]

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A1[2][current_state] = np.empty([0], dtype=order_type)

                    denom = 1 / (np.exp(self.obs1[
                        events[state_events][pt[state_events]]].sum()) -
                        diag_paired[current_state])

                    for pre_state, pre_orders1 in A1[0].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not (current_state | pre_state == current_state):
                            continue

                        # get the position of the new 1
                        new_event = bin(current_state ^ pre_state)[
                            :1:-1].index("1")

                        # get the numerator
                        num = np.exp(self.log_theta[
                            events[new_event],
                            events[state_events][pt[state_events]]].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders1.copy()
                        new_orders["prob"] *= num * denom
                        new_orders["order"] = append_to_int_order(
                            my_int=append_to_int_order(
                                my_int=new_orders["order"],
                                numbers=[e for e in state_events if e not in [
                                    new_event, new_event + 1]],
                                new_event=new_event
                            ),
                            numbers=[
                                e for e in state_events if e != new_event],
                            new_event=new_event + 1
                        )
                        A1[2][current_state] = np.append(
                            A1[2][current_state],
                            new_orders)

                    # just keep the most like order to reach
                    # current_state
                    A1[2][current_state] = A1[2][current_state][
                        [np.argmax(A1[2][current_state]["prob"])]]

            # remove the orders and probs that we do not need anymore
            A1.popleft()
            A2.popleft()

        bin_state = int("1" * k, base=2)
        arg_max = np.argmax(A2[0][bin_state]["prob"])
        o, p = A2[0][bin_state][arg_max]
        p *= np.exp(self.obs2[events[state_events][~pt[state_events]]].sum())
        return int_to_order(o, np.nonzero(state)[0].tolist()), p

    def _likelihood_two_orders(self, order_1: np.array, order_2: np.array) -> float:
        """ Compute the likelihood of two orders of events happening before the first
        and the second observation 

        Args:
            order_1 (np.array): Order of events (2i and 2i+1 encode the ith events happening in PT and Met respectively)
            that have happened when the first observation has been made. Note that these do not correspond to the actual PT
            observation, as it is possible that events have happened in the metastasis that are not visible in the PT 
            observation.
            order_2 (np.array): Order of events (2i and 2i+1 encode the ith events happening in PT and Met respectively)
            that have happened when the second observation has been made. Note that these do not correspond to the actual Met
            observation, as it is possible that events have happened in the primary tumor that are not visible in the Met 
            observation.

        Returns:
            float: likelihood of these two orders happening
        """
        # translate first observation to state
        state = np.zeros(2 * self.n + 1, dtype=int)
        if len(order_1) > 0:
            state[order_1] = 1
        diag = get_diag_paired(log_theta=self.log_theta, n=self.n, state=state)

        event_to_bin = {e: 1 << i for i, e in enumerate(np.sort(order_1))}

        p = 1 / (1 - diag[0])

        current_state = np.zeros(2 * self.n + 1)
        current_state_bin = 0  # binary state
        seeded = False
        for i, e in enumerate(order_1):
            if not seeded:
                if i % 2:  # if the seeding has not happened yet, every second event is just the second part of the joint development
                    continue
                if e == 2 * self.n:  # seeding
                    seeded = True
                    current_state[-1] = 1
                    current_state_bin += event_to_bin[2 * self.n]
                    p *= (np.exp(self.log_theta[
                        self.n, current_state[::2].astype(bool)].sum())
                        / (np.exp(self.obs1[
                            np.append(current_state[:-1:2].astype(bool), False)].sum())
                           - diag[current_state_bin]))
                else:
                    current_state[[e, e + 1]] = 1
                    current_state_bin += (event_to_bin[e] +
                                          event_to_bin[e + 1])
                    p *= (np.exp(self.log_theta[
                        e // 2, current_state[::2].astype(bool)].sum())
                        / (np.exp(self.obs1[
                            np.append(current_state[:-1:2].astype(bool), False)].sum())
                           - diag[current_state_bin]))
            else:
                current_state[e] = 1
                current_state_bin += event_to_bin[e]
                if not e % 2:  # PT event
                    p *= (np.exp(self.log_theta[
                        e//2, np.append(
                            current_state[:-1:2].astype(bool), False)].sum())
                          / (np.exp(self.obs1[
                              np.append(current_state[:-1:2].astype(bool), False)].sum())
                             - diag[current_state_bin]))
                else:  # Met event
                    p *= (np.exp(self.log_theta[
                        e//2, np.append(
                            current_state[1::2].astype(bool), True)].sum())
                          / (np.exp(self.obs1[
                              np.append(current_state[:-1:2].astype(bool), False)].sum())
                             - diag[current_state_bin]))
            pass

        p *= np.exp(self.obs1[
            np.append(current_state[:-1:2].astype(bool), False)].sum())
        current_state = np.append(state[1::2], [1])  # reduce to met events
        k = len(order_2) + current_state.sum()
        state = current_state.copy()
        if len(order_2) > 0:
            state[order_2 // 2] = 1
        event_to_bin = {e: 1 << i for i, e in enumerate(np.nonzero(state)[0])}
        current_state_bin = (
            current_state[state.astype(bool)] << np.arange(k)).sum()
        diag = self.get_diag_unpaired(state=state)
        p /= (np.exp(self.obs2[current_state.astype(bool)].sum())
              - diag[current_state_bin])

        for i, e in enumerate(order_2):
            e = e//2
            current_state[e] = 1
            current_state_bin += event_to_bin[e]
            p *= (np.exp(self.log_theta[e, current_state.astype(bool)].sum())
                  / (np.exp(self.obs2[current_state.astype(bool)].sum())
                     - diag[current_state_bin]))
            pass

        p *= np.exp(self.obs2[current_state.astype(bool)].sum())

        return p

    def likelihood(self, order: tuple[int]) -> float:
        """This function returns the probability of observing a specific
        order of events after two observations (timepoint of first
        observation does not matter)

        Args:
            order (tuple[int]): Sequence of events

        Returns:
            float: Probability of observing this order.
        """
        return sum(
            self._likelihood_two_orders(o1, o2) for o1, o2 in get_combos(
                order=np.array(order), n=self.n))

    def simulate(self, timepoint: int) -> tuple[np.array, float]:
        """This function simulates one sample according to the metMHN.

        Args:
            timepoint (int): At which timepoint to stop the simulation,
            t_1 or t_2. Must be either 1 or 2.

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
                                  self.get_diag_unpaired(state=state)[-1])
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
                                       self.get_diag_unpaired(state=state)[-1])
        return order, t_obs


if __name__ == "__main__":
    import pandas as pd

    log_theta = pd.read_csv(
        R"results\luad\luad_16_muts_5_cnvs_0028.csv", index_col=0)
    obs1 = log_theta.iloc[0].to_numpy()
    obs2 = log_theta.iloc[1].to_numpy()

    log_theta.drop(index=[0, 1], inplace=True)
    mmhn = MetMHN(log_theta=log_theta.to_numpy(), obs1=obs1, obs2=obs2)
    state = np.zeros(2 * mmhn.n + 1, dtype=int)
    state[[0, 1, -1, 3]] = 1

    mmhn.likelihood(order_1=np.array([0, 1, 42, 3]), order_2=np.array([]))
    print(mmhn.likeliest_order_paired(state))
