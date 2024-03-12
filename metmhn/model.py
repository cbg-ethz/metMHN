from metmhn.perf import order_to_int, int_to_order, append_to_int_order
from collections import deque
from metmhn.np.kronvec import kron_diag as get_diag_paired
from scipy.linalg.blas import dcopy, dscal, daxpy
import numpy as np
from omhn.model import oMHN
from metmhn.state import State, RestrState, RestrMetState, MetState
from itertools import chain
from typing import Union

append_to_int_order = np.vectorize(
    append_to_int_order, excluded=["numbers", "new_event"])


def tuple_max(x: np.array, y: np.array) -> tuple[np.array]:
    """If given two values x_i and y_i for each i, we want to find i s.t. for all non-negative linear factors a and b we have ax_i + by_i >= ax_j + by_j.
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


def triple_max(x: np.array, y: np.array, z: np.array) -> tuple[np.array]:
    """If given three values x_i, y_i and z_i for each i, we want to find i s.t. for all non-negative linear factors a, b and c we have ax_i + by_i + cz_i >= ax_j + by_j + cz_j.
    There will in general not be a unique i that satisfies this,
    therefore we just return all possible candidates i that could
    fulfill this for the right values a, b and c.

    Args:
        x (np.array): x
        y (np.array): y
        z (np.array): z

    Returns:
        tuple[np.array]: Vectors that only contain the maximizing
        candidates.
    """

    x.sort(order="order")
    y.sort(order="order")
    z.sort(order="order")

    indices = list()

    for i in range(len(x)):
        non_dominated = True
        for j in range(len(x)):
            if x[i]["prob"] < x[j]["prob"] and y[i]["prob"] < y[j]["prob"] and z[i]["prob"] < z[j]["prob"]:
                non_dominated = False
                break
        if non_dominated:
            indices.append(i)

    return x[indices], y[indices], z[indices]


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


def get_combos(order: np.array, n: int, first_obs: str) -> list[tuple[np.array]]:
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

    if first_obs == "PT":
        for i in range(len(order)-seeding[0]):
            combos.append(np.split(order, [len(order)-i]))
            if not order[-i - 1] % 2:
                break
        return combos
    elif first_obs == "Met":
        for i in range(len(order)-seeding[0]):
            combos.append(np.split(order, [len(order)-i]))
            if order[-i - 1] % 2:
                break
        return combos
    else:
        raise ValueError("first_obs must be 'PT' or 'Met'.")


def bits_fixed_n(n, k):
    """
    Generator over integers whose binary representation has a fixed
    number of 1s, in lexicographical order.
    From https://graphics.stanford.edu/~seander/bithacks.html#NextBitPermutation

    :param n: How many 1s there should be
    :param k: How many bits the integer should have
    """
    v = int("1"*n, 2)
    stop_no = v << (k-n)
    w = -1
    while w != stop_no:
        t = (v | (v - 1)) + 1
        w = t | ((((t & -t)) // (v & (-v)) >> 1) - 1)
        v, w = w, v
        yield w


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

        _pt_log_theta = log_theta.copy()
        _pt_log_theta[:-1, -1] = 0
        self._pt_omhn = oMHN(
            log_theta=np.vstack([_pt_log_theta, self.obs1])
        )

    def likeliest_order(
        self,
        state: Union[np.array, MetState],
        met_status: str,
        first_obs: str = None
    ) -> tuple[tuple[int, ...], float]:
        """Returns the most probable order for a coupled observation 
        consisting of PT and Met

        Args:
            state (np.array or MetState): state describing the coupled observation, 
            numpy shape (2*n + 1).
            met_status (str): Must be one of
                - "isMetastasis" for an unpaired metastasis
                - "present" for an unpaired primary tumor that at some
                    point develops a metastasis
                - "absent" for an unpaired primary tumor that does not
                    develop a metastasis
                - "isPaired" for a paired sample
            first_obs (str): Which was the first observation. Must be 
            one of "Met", "PT" or "sync"

        Returns:
            tuple[tuple[int, ...], float]: most probable order and its 
            probability
        """

        if isinstance(state, np.ndarray):
            state = MetState.from_seq(state)

        match met_status:
            case "isMetastasis":
                return self._likeliest_order_unpaired_mt(state=state.MT)
            case "absent":
                p, o = self._pt_omhn.likeliest_order(
                    state=state.PT_S.to_seq()
                )
                return p, 2 * o
            case "present":
                p, o = self._pt_omhn.likeliest_order(
                    state=state.PT_S.to_seq()
                )
                return p, 2 * o
            case "isPaired":
                match first_obs:
                    case"PT":
                        return self._likeliest_order_pt_mt(state)
                    case "Met":
                        return self._likeliest_order_mt_pt(state)
                    case "sync":
                        return self._likeliest_order_sync(state)
                    case _:
                        raise ValueError(
                            "first_obs must be one of 'PT', 'Met', 'sync'")
            case _:
                raise ValueError(
                    "met_status must be one of 'isMetastasis', 'absent', 'present', 'isPaired")

    def _get_diag_unpaired(self, state: State, seeding: bool = True) -> np.array:
        """This returns the diagonal of the restricted rate matrix of
        the metMHN's Markov chain.

        Args:
            state (np.array): Binary unpaired state vector, dtype must
            be int32. This is the vector according to which state space
            restriction will be performed. Shape (n,) with n the number
            of events including seeding.
            seeding (bool, optional): Whether the seeding can be
            acquired

        Returns:
            np.array: Diagonal of the restricted rate matrix. Shape
            (2^k,) with k the number of 1s in state.
        """
        k = len(state)
        nx = 1 << k
        diag = np.zeros(nx)
        subdiag = np.zeros(nx)

        n = self.n + 1 if seeding else self.n

        for i in range(n):

            current_length = 1
            subdiag[0] = 1
            # compute the ith subdiagonal of Q
            for j in range(n):
                if j in state:
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

    def _likeliest_order_unpaired_mt(self, state: State) -> tuple[tuple[int, ...], float]:
        """
        Calculates the likeliest order of events for unpaired metastasis
        observation.

        Args:
            state (State): The state representing the metastasis.

        Returns:
            tuple[tuple[int, ...], float]: A tuple containing the
            likeliest order of events and the corresponding probability.

        """
        diag = self._get_diag_unpaired(state=state, seeding=True)
        log_theta = self.log_theta[state.to_seq()][:, state.to_seq()]
        obs1 = self.obs1[state.to_seq()]
        obs2 = self.obs2[state.to_seq()]

        k = len(state)
        # {state: highest path probability to this state}
        A = {0: 1/(1-diag[0])}
        # {state: path with highest probability to this state}
        B = {0: []}
        for i in range(1, k+1):         # i is the number of events
            A_new = dict()
            B_new = dict()
            for st in bits_fixed_n(n=i, k=k):
                A_new[st] = -1
                state_events = np.array(
                    # events in state
                    [i for i in range(k) if (1 << i) | st == st])
                for e in state_events:
                    # numerator in Gotovos formula
                    num = np.exp(log_theta[e, state_events].sum())
                    pre_st = st - (1 << e)
                    if A[pre_st] * num > A_new[st]:
                        A_new[st] = A[pre_st] * num
                        B_new[st] = B[pre_st].copy()
                        B_new[st].append(e)
                # obs1 if seeding has not happened yet, else obs2
                obs = np.exp(obs1[state_events].sum()) \
                    if not (1 << (k - 1)) & st \
                    else np.exp(obs2[state_events].sum())
                A_new[st] /= (obs - diag[st])
            A = A_new
            B = B_new
        i = (1 << k) - 1
        A[i] *= np.exp(obs2.sum())
        order = np.arange(self.log_theta.shape[1])[state.to_seq()][B[i]]
        order = 2 * order + 1
        order[np.where(order == self.n * 2 + 1)] -= 1
        return (A[i], order)

    def _likeliest_order_pt_mt(self, state: MetState, verbose: bool = False) -> tuple[tuple[int, ...], float]:
        """
        Calculates the likeliest order of events to reach a given state
        with first a PT observation followed by an MT observation.

        Args:
            state (MetState): The target state to reach.
            verbose (bool, optional): Whether to print verbose output.
            Defaults to False.

        Returns:
            tuple[tuple[int, ...], float]: A tuple containing the
            likeliest order of events as a tuple of integers and the
            corresponding probability.
        """
        k = len(state)
        if not state.reachable:
            raise ValueError("This state is not reachable by mhn.")

        seq_state = state.to_seq()

        diag_paired = get_diag_paired(
            log_theta=self.log_theta, n=self.n, state=state.to_seq())
        diag_unpaired = self._get_diag_unpaired(
            state=state.MT)

        # In A1[i][state][order], the probabilities to reach a state
        # with a given order are stored. Here, i can be 0, 1 or 2, where
        # A1[2] holds the states that have n_events events and A1[1] and
        # A1[1] hold the ones with 1 and 2 events less, respectively.

        order_type = [("order", int), ("prob", float)]
        # get there with tau1
        A1 = deque([
            dict(),
            {RestrMetState(0, restrict=state): np.array(
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
                if not reachable(
                        bin_state=current_state, state=state.to_seq(), n=self.n):
                    continue

                current_state = RestrMetState(current_state, restrict=state)

                # whether seeding has happened
                if k - 1 in current_state:

                    # Does the pt part fit the observation?
                    pt_terminal = current_state.PT_events == state.PT_events

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A1[2][current_state] = np.empty([0], dtype=order_type)

                    if pt_terminal:

                        obs1 = np.exp(self.obs1[
                            current_state.PT_events + current_state.Seeding,].sum())
                        obs2 = np.exp(self.obs2[
                            current_state.MT_events + current_state.Seeding,].sum())

                        denom2 = 1 / \
                            (obs2 - diag_unpaired[current_state.MT.data])
                        start_factor = obs1 * denom2

                    # iterate over all previous states
                    for pre_state, pre_orders1 in A1[1].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        denom1 = 1 / \
                            (np.exp(self.obs1[
                                current_state.PT_events +
                                    current_state.Seeding,].sum())
                                + np.exp(self.obs2[
                                    current_state.MT_events +
                                    current_state.Seeding,].sum())
                                - diag_paired[current_state.data])

                        # whether new event is pt
                        if len(diff.PT) > 0:  # new event is pt
                            num = np.exp(self.log_theta[
                                diff.PT_events[0],
                                current_state.PT_events].sum())
                        else:  # new event is met or seeding
                            num = np.exp(self.log_theta[
                                diff.MT_events or diff.Seeding,
                                current_state.MT_events + current_state.Seeding].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders1.copy()
                        new_orders["prob"] *= (num * denom1)
                        new_orders["order"] = append_to_int_order(
                            new_orders["order"],
                            numbers=list(pre_state),
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
                            if not pre_state <= current_state:
                                continue

                            # if pre_state was pt_terminal
                            if pre_state in A2[0]:

                                # get the position of the new 1
                                diff = current_state ^ pre_state
                                new_event = tuple(diff)[0]

                                # Get the numerator
                                num = np.exp(self.log_theta[
                                    diff.events[0],
                                    current_state.MT_events + current_state.Seeding].sum())

                                # get the orders that are coming from
                                # pre_state
                                new_orders = append_to_int_order(
                                    my_int=A2[0][pre_state]["order"],
                                    numbers=list(pre_state),
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

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A1[2][current_state] = np.empty([0], dtype=order_type)

                    denom = 1 / (np.exp(self.obs1[
                        current_state.PT_events,].sum()) -
                        diag_paired[current_state.data])

                    for pre_state, pre_orders1 in A1[0].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        # get the numerator
                        num = np.exp(self.log_theta[
                            diff.events[0],
                            current_state.PT_events].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders1.copy()
                        new_orders["prob"] *= num * denom
                        new_orders["order"] = append_to_int_order(
                            my_int=append_to_int_order(
                                my_int=new_orders["order"],
                                numbers=list(pre_state),
                                new_event=new_event
                            ),
                            numbers=list(pre_state) + [new_event],
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

        bin_state = RestrMetState((1 << k) - 1, restrict=state)
        arg_max = np.argmax(A2[0][bin_state]["prob"])
        o, p = A2[0][bin_state][arg_max]
        p *= np.exp(self.obs2[current_state.MT_events +
                    current_state.Seeding,].sum())
        return int_to_order(o, np.nonzero(state.to_seq())[0].tolist()), p

    def _likeliest_order_mt_pt(
        self, state: MetState, verbose: bool = False
    ) -> tuple[tuple[int, ...], float]:

        k = len(state)
        if not reachable(
                bin_state=(1 << k) - 1, n=self.n, state=state):
            raise ValueError("This state is not reachable by mhn.")

        diag_paired = get_diag_paired(
            log_theta=self.log_theta, n=self.n, state=state.to_seq())
        diag_unpaired = self._get_diag_unpaired(
            state=state.PT_S, seeding=False)

        # In A1[i][state][order], the probabilities to reach a state
        # with a given order are stored. Here, i can be 0, 1 or 2, where
        # A1[2] holds the states that have n_events events and A1[1] and
        # A1[0] hold the ones with 1 and 2 events less, respectively.

        order_type = [("order", int), ("prob", float)]
        # get there with tau1
        A1 = deque([
            dict(),
            {RestrMetState(0, restrict=state): np.array(
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
                if not reachable(bin_state=current_state, state=state.to_seq(),
                                 n=self.n):
                    continue

                current_state = RestrMetState(
                    current_state, restrict=state)

                # whether seeding has happened
                if k - 1 in current_state:

                    # Does the met part fit the observation?
                    met_terminal = state.MT_events == current_state.MT_events

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A1[2][current_state] = np.empty([0], dtype=order_type)

                    if met_terminal:

                        obs1 = np.exp(self.obs1[
                            current_state.PT_events + current_state.Seeding,].sum())
                        obs2 = np.exp(self.obs2[
                            current_state.MT_events + current_state.Seeding,].sum())
                        denom2 = 1 / \
                            (obs1 - diag_unpaired[current_state.PT.data])
                        start_factor = obs2 * denom2

                    # iterate over all previous states
                    for pre_state, pre_orders1 in A1[1].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        denom1 = 1 / \
                            (np.exp(self.obs1[
                                current_state.PT_events + current_state.Seeding,].sum())
                             + np.exp(self.obs2[
                                 current_state.MT_events + current_state.Seeding,].sum())
                                - diag_paired[current_state.data])

                        # whether new event is pt
                        if len(diff.PT_events) > 0:  # new event is pt
                            num = np.exp(self.log_theta[
                                diff.PT_events,
                                current_state.PT_events].sum())
                        else:  # new event is met or seeding
                            num = np.exp(self.log_theta[
                                diff.MT_events or diff.Seeding,
                                current_state.MT_events + current_state.Seeding].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders1.copy()
                        new_orders["prob"] *= (num * denom1)
                        new_orders["order"] = append_to_int_order(
                            new_orders["order"],
                            numbers=list(pre_state),
                            new_event=new_event)
                        A1[2][current_state] = np.append(
                            A1[2][current_state],
                            new_orders
                        )

                    if met_terminal:

                        # all probabilities to reach with tau2 here are
                        # at least the ones to reach with tau1 times the
                        # start factor
                        A2[1][current_state] = A1[2][current_state].copy()
                        A2[1][current_state]["prob"] *= start_factor

                        for pre_state, pre_orders1 in A1[1].items():
                            # if current_state is met terminal, it is
                            # possible that the same holds for a
                            # prestate.
                            # Then we need to calculate how we can get
                            # from pre_state to current_state

                            # Skip pre_state if it is not a subset of
                            # current_state
                            if not pre_state <= current_state:
                                continue

                            # if pre_state was met_terminal
                            if pre_state in A2[0]:

                                # get the position of the new 1
                                diff = current_state ^ pre_state
                                new_event = tuple(diff)[0]

                                # Get the numerator
                                num = np.exp(self.log_theta[
                                    diff.events[0],
                                    current_state.PT_events].sum())

                                # get the orders that are coming from
                                # pre_state
                                new_orders = append_to_int_order(
                                    my_int=A2[0][pre_state]["order"],
                                    numbers=list(pre_state),
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
                    if met_terminal:
                        A1[2][current_state], A2[1][current_state] = \
                            tuple_max(
                                A1[2][current_state],
                                A2[1][current_state])

                else:  # seeding has not happened yet

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A1[2][current_state] = np.empty([0], dtype=order_type)

                    denom = 1 / (np.exp(self.obs1[
                        current_state.PT_events,].sum()) -
                        diag_paired[current_state.data])

                    for pre_state, pre_orders1 in A1[0].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        # get the numerator
                        num = np.exp(self.log_theta[
                            diff.events[0],
                            current_state.PT_events].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders1.copy()
                        new_orders["prob"] *= num * denom
                        new_orders["order"] = append_to_int_order(
                            my_int=append_to_int_order(
                                my_int=new_orders["order"],
                                numbers=list(pre_state),
                                new_event=new_event
                            ),
                            numbers=list(pre_state) + [new_event],
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

        bin_state = RestrMetState((1 << k) - 1, restrict=state)
        arg_max = np.argmax(A2[0][bin_state]["prob"])
        o, p = A2[0][bin_state][arg_max]
        p *= np.exp(self.obs1[bin_state.PT_events + bin_state.Seeding,].sum())
        return int_to_order(o, np.nonzero(state.to_seq())[0].tolist()), p

    def _likeliest_order_unknown(
        self, state: MetState, verbose: bool = False
    ) -> tuple[tuple[int, ...], float]:

        k = len(state)
        if not reachable(
                bin_state=(1 << k) - 1, n=self.n, state=state):
            raise ValueError("This state is not reachable by mhn.")

        diag_paired = get_diag_paired(
            log_theta=self.log_theta, n=self.n, state=state.to_seq())
        diag_unpaired_pt = self._get_diag_unpaired(
            state=state.PT_S, seeding=False)
        diag_unpaired_mt = self._get_diag_unpaired(
            state=state.MT, seeding=True)

        # In A1[i][state][order], the probabilities to reach a state
        # with a given order are stored. Here, i can be 0, 1 or 2, where
        # A1[2] holds the states that have n_events events and A1[1] and
        # A1[0] hold the ones with 1 and 2 events less, respectively.

        order_type = [("order", int), ("prob", float)]
        # get there with before any observation
        A = deque([
            dict(),
            {RestrMetState(0, restrict=state): np.array(
                [(0, 1 / (1 - diag_paired[0]))],
                dtype=order_type)}])
        # get there after PT observation
        AP = deque([dict()])
        # get there after MT observation
        AM = deque([dict()])

        for n_events in range(1, k + 1):
            # create dicts to hold the probs and orders to reach states
            # with n_events events
            A.append(dict())
            AP.append(dict())
            AM.append(dict())

            # iterate over all states with n_events events
            for current_state in bits_fixed_n(n=n_events, k=k):

                if verbose:
                    print(
                        f"{n_events:3}/{k:3}, {len(A[2]):10}, {sum(len(x) for x in A[2].values()):10}", end="\r")

                # check whether state is reachable
                if not reachable(bin_state=current_state, state=state.to_seq(),
                                 n=self.n):
                    continue

                current_state = RestrMetState(
                    current_state, restrict=state)

                # whether seeding has happened
                if k - 1 in current_state:

                    # Does the pt/mt part fit the observation?
                    pt_terminal = state.PT_events == current_state.PT_events
                    mt_terminal = state.MT_events == current_state.MT_events

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A[2][current_state] = np.empty([0], dtype=order_type)

                    obs1 = np.exp(self.obs1[
                        current_state.PT_events + current_state.Seeding,].sum())
                    obs2 = np.exp(self.obs2[
                        current_state.MT_events + current_state.Seeding,].sum())

                    denom1 = 1 / \
                        (obs1 + obs2 - diag_paired[current_state.data])

                    # iterate over all previous states
                    for pre_state, pre_orders in A[1].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        # whether new event is pt
                        if len(diff.PT_events) > 0:  # new event is pt
                            num = np.exp(self.log_theta[
                                diff.PT_events,
                                current_state.PT_events].sum())
                        else:  # new event is met or seeding
                            num = np.exp(self.log_theta[
                                diff.MT_events or diff.Seeding,
                                current_state.MT_events + current_state.Seeding].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders.copy()
                        new_orders["prob"] *= (num * denom1)
                        new_orders["order"] = append_to_int_order(
                            new_orders["order"],
                            numbers=list(pre_state),
                            new_event=new_event)
                        A[2][current_state] = np.append(
                            A[2][current_state],
                            new_orders
                        )

                    if mt_terminal:

                        denom2 = 1 / \
                            (obs1 - diag_unpaired_pt[current_state.PT.data])
                        start_factor = obs2 * denom2

                        # all probabilities to reach after MT obs are at
                        # least the ones to reach before MT obs times
                        # the start factor
                        AM[1][current_state] = A[2][current_state].copy()
                        AM[1][current_state]["prob"] *= start_factor

                        for pre_state, pre_orders in AM[0].items():
                            # if current_state is mt terminal, it is
                            # possible that the same holds for a
                            # prestate.
                            # Then we need to calculate how we can get
                            # from pre_state to current_state

                            # Skip pre_state if it is not a subset of
                            # current_state
                            if not pre_state <= current_state:
                                continue

                            # get the position of the new 1
                            diff = current_state ^ pre_state
                            new_event = tuple(diff)[0]

                            # Get the numerator
                            num = np.exp(self.log_theta[
                                diff.events[0],
                                current_state.PT_events].sum())

                            # get the orders that are coming from
                            # pre_state
                            new_orders = append_to_int_order(
                                my_int=pre_orders["order"],
                                numbers=list(pre_state),
                                new_event=new_event)

                            # add to the probs of these orders the
                            # prob to reach current_state
                            AM[1][current_state]["prob"][
                                np.isin(AM[1][current_state]["order"],
                                        new_orders)] \
                                += AM[0][pre_state]["prob"] \
                                * num * denom2

                    if pt_terminal:

                        denom2 = 1 / \
                            (obs2 - diag_unpaired_mt[current_state.MT.data])
                        start_factor = obs1 * denom2

                        # all probabilities to reach after PT obs are at
                        # least the ones to reach before PT obs times
                        # the start factor
                        AP[1][current_state] = A[2][current_state].copy()
                        AP[1][current_state]["prob"] *= start_factor

                        for pre_state, pre_orders in AP[0].items():
                            # if current_state is pt_terminal, it is
                            # possible that the same holds for a
                            # prestate.
                            # Then we need to calculate how we can get
                            # from pre_state to current_state

                            # Skip pre_state if it is not a subset of
                            # current_state
                            if not pre_state <= current_state:
                                continue

                            # get the position of the new 1
                            diff = current_state ^ pre_state
                            new_event = tuple(diff)[0]

                            # Get the numerator
                            num = np.exp(self.log_theta[
                                diff.events[0],
                                current_state.MT_events].sum())

                            # get the orders that are coming from
                            # pre_state
                            new_orders = append_to_int_order(
                                my_int=pre_orders["order"],
                                numbers=list(pre_state),
                                new_event=new_event)

                            # add to the probs of these orders the
                            # prob to reach current_state
                            AP[1][current_state]["prob"][
                                np.isin(AP[1][current_state]["order"],
                                        new_orders)] \
                                += AP[0][pre_state]["prob"] \
                                * num * denom2

                    # Now I have possibly three dicts
                    # A[2][current_state], AP[2][current_state] and
                    # AM[1][current_state] with possible paths to get to
                    # current_state. I can kick out some of them,
                    # because I am only interested in those orders that
                    # stand a chance to be maximal
                    if pt_terminal and mt_terminal:
                        A[2][current_state], AP[1][current_state], \
                            AM[1][current_state] = \
                            triple_max(
                                A[2][current_state],
                                AP[1][current_state],
                                AM[1][current_state])
                    elif pt_terminal:
                        A[2][current_state], AP[1][current_state] = \
                            tuple_max(
                                A[2][current_state],
                                AP[1][current_state])
                    elif mt_terminal:
                        A[2][current_state], AM[1][current_state] = \
                            tuple_max(
                                A[2][current_state],
                                AM[1][current_state])

                else:  # seeding has not happened yet

                    # initialize empty numpy struct array for probs and
                    # orders to reach current_state
                    A[2][current_state] = np.empty([0], dtype=order_type)

                    denom = 1 / (np.exp(self.obs1[
                        current_state.PT_events,].sum()) -
                        diag_paired[current_state.data])

                    for pre_state, pre_orders in A[0].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        # get the numerator
                        num = np.exp(self.log_theta[
                            diff.events[0],
                            current_state.PT_events].sum())

                        # Assign the probabilities for A1
                        new_orders = pre_orders.copy()
                        new_orders["prob"] *= num * denom
                        new_orders["order"] = append_to_int_order(
                            my_int=append_to_int_order(
                                my_int=new_orders["order"],
                                numbers=list(pre_state),
                                new_event=new_event
                            ),
                            numbers=list(pre_state) + [new_event],
                            new_event=new_event + 1
                        )
                        A[2][current_state] = np.append(
                            A[2][current_state],
                            new_orders)

                    # just keep the most like order to reach
                    # current_state
                    A[2][current_state] = A[2][current_state][
                        [np.argmax(A[2][current_state]["prob"])]]

            # remove the orders and probs that we do not need anymore
            A.popleft()
            AP.popleft()
            AM.popleft()

        bin_state = RestrMetState((1 << k) - 1, restrict=state)
        AP[0][bin_state]["prob"] *= np.exp(self.obs2[
            bin_state.MT_events + bin_state.Seeding,].sum())
        AM[0][bin_state]["prob"] *= np.exp(self.obs1[
            bin_state.PT_events + bin_state.Seeding,].sum())

        arg_max = np.argmax(AP[0][bin_state]["prob"] +
                            AM[0][bin_state]["prob"])
        o, p = AP[0][bin_state][arg_max]["order"], AP[0][bin_state][arg_max]["prob"] + \
            AM[0][bin_state][arg_max]["prob"]
        return int_to_order(o, np.nonzero(state.to_seq())[0].tolist()), p

    def _likeliest_order_sync(self, state: MetState
                              ) -> tuple[tuple[int, ...], float]:
        """Returns the most probable order for a coupled observation 
        consisting of PT and Met if they were observed at the same time

        Args:
            state (np.array): state describing the coupled observation, 
            shape (2*n + 1).

        Returns:
            tuple[tuple[int, ...], float]: most probable order and its 
            probability
        """

        k = len(state)

        if not state.reachable:
            raise ValueError("This state is not reachable by mhn.")

        diag_paired = get_diag_paired(
            log_theta=self.log_theta, n=self.n, state=state.to_seq())

        # In A[i][state], the max. probability to reach a state and its
        # order are stored. Here, i can be 0, 1 or 2, where A[2]
        # holds the states that have n_events events and A[1] and A[0]
        # hold the ones with 1 and 2 events less, respectively.
        A = deque([
            dict(),
            {RestrMetState(0, restrict=state): [0, 1 / (1 - diag_paired[0])]}
        ])

        for n_events in range(1, k + 1):
            # create dicts to hold the probs and orders to reach states
            # with n_events events
            A.append(dict())

            # iterate over all states with n_events events
            for current_state in bits_fixed_n(n=n_events, k=k):

                # check whether state is reachable
                if not reachable(
                        bin_state=current_state, state=state.to_seq(), n=self.n):
                    continue

                current_state = RestrMetState(
                    current_state, restrict=state)
                # initialize empty numpy struct array for probs and
                # orders to reach current_state
                A[2][current_state] = [-1, -1.]

                # whether seeding has happened
                if k - 1 in current_state:

                    # iterate over all previous states
                    for pre_state, pre_order in A[1].items():

                        # Skip pre_state if it is not a subset of
                        # current_state

                        if not pre_state <= current_state:
                            continue

                        # get difference of pre_state and current_state
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        # whether new event is pt
                        if len(diff.PT) > 0:  # new event is pt
                            num = np.exp(self.log_theta[
                                diff.events[0],
                                current_state.PT_events].sum())
                        else:  # new event is met
                            num = np.exp(self.log_theta[
                                diff.MT_events or diff.Seeding,
                                current_state.MT_events + current_state.Seeding].sum())

                        if pre_order[1] * num > A[2][current_state][1]:
                            A[2][current_state][1] = pre_order[1] * num
                            A[2][current_state][0] = append_to_int_order(
                                pre_order[0],
                                numbers=list(pre_state),
                                new_event=new_event)

                    obs1 = np.exp(self.obs1[
                        current_state.PT_events + current_state.Seeding,].sum())
                    obs2 = np.exp(self.obs2[
                        current_state.MT_events + current_state.Seeding,].sum())
                    A[2][current_state][1] /= \
                        (obs1 + obs2 - diag_paired[current_state.data])

                else:
                    # seeding has not happened yet

                    for pre_state, pre_order in A[0].items():

                        # Skip pre_state if it is not a subset of
                        # current_state
                        if not pre_state <= current_state:
                            continue

                        # get the position of the new 1
                        diff = current_state ^ pre_state
                        new_event = tuple(diff)[0]

                        # get the numerator
                        num = np.exp(self.log_theta[
                            diff.events[0],
                            current_state.PT_events].sum())

                        if pre_order[1] * num > A[2][current_state][1]:
                            A[2][current_state][1] = pre_order[1] * num
                            A[2][current_state][0] = append_to_int_order(
                                append_to_int_order(
                                    my_int=pre_order[0],
                                    numbers=list(pre_state),
                                    new_event=new_event
                                ),
                                numbers=list(pre_state) + [new_event],
                                new_event=new_event + 1)
                    A[2][current_state][1] /= \
                        (np.exp(self.obs1[current_state.PT_events,].sum())
                         - diag_paired[current_state.data])

            # remove the orders and probs that we do not need anymore
            A.popleft()

        final = RestrMetState((1 << k) - 1, restrict=state)
        o, p = A[1][final]

        obs1 = np.exp(self.obs1[final.PT_events + final.Seeding,].sum())
        obs2 = np.exp(self.obs2[final.MT_events + final.Seeding,].sum())

        p *= (obs1 + obs2)
        return int_to_order(o, np.nonzero(state.to_seq())[0].tolist()), p

    def _likelihood_pt_mt_timed(self, order_1: np.array, order_2: np.array
                                ) -> float:
        """ Compute the likelihood of two orders of events happening
        before the first and the second observation 

        Args:
            order_1 (np.array): Order of events (2i and 2i+1 encode the
            ith events happening in PT and Met respectively) that have
            happened when the first observation has been made. Note that
            these do not correspond to the actual PT observation, as it
            is possible that events have happened in the metastasis that
            are not visible in the PT observation.
            order_2 (np.array): Order of events (2i and 2i+1 encode the
            ith events happening in PT and Met respectively) that have
            happened when the second observation has been made. Note
            that these do not correspond to the actual Metobservation,
            as it is possible that events have happened in the primary
            tumor that are not visible in the Met observation.

        Returns:
            float: likelihood of these two orders happening
        """
        # translate first observation to state
        state = MetState(order_1, size=2 * self.n + 1)
        diag = get_diag_paired(log_theta=self.log_theta,
                               n=self.n, state=state.to_seq())

        p = 1 / (1 - diag[0])

        current_state = RestrMetState(0, restrict=state)

        for i, e in enumerate(order_1):
            if not current_state.Seeding:
                # if the seeding has not happened yet, every second
                # event is just the second part of the joint development
                if i % 2:
                    continue
                if e == 2 * self.n:  # seeding
                    current_state.add(len(state) - 1)
                    p *= (np.exp(self.log_theta[
                        self.n, current_state.PT_events + current_state.Seeding].sum())
                        / (np.exp(self.obs1[current_state.PT_events + current_state.Seeding,].sum())
                           + np.exp(self.obs2[current_state.MT_events + current_state.Seeding,].sum())
                           - diag[current_state.data]))
                else:
                    _event = list(state).index(e)
                    current_state.add(_event)
                    current_state.add(_event + 1)
                    p *= (np.exp(self.log_theta[
                        e // 2, current_state.PT_events + current_state.Seeding].sum())
                        / (np.exp(self.obs1[current_state.PT_events + current_state.Seeding,].sum())
                           - diag[current_state.data]))
            else:  # seeded
                current_state.add(list(state).index(e))
                if not e % 2:  # PT event
                    p *= (np.exp(self.log_theta[
                        e//2, current_state.PT_events].sum())
                        / (np.exp(self.obs1[current_state.PT_events + current_state.Seeding,].sum())
                           + np.exp(self.obs2[current_state.MT_events + current_state.Seeding,].sum())
                           - diag[current_state.data]))
                else:  # Met event
                    p *= (np.exp(self.log_theta[
                        e//2, current_state.MT_events + current_state.Seeding].sum())
                        / (np.exp(self.obs1[current_state.PT_events + current_state.Seeding,].sum())
                           + (np.exp(self.obs2[current_state.MT_events + current_state.Seeding,].sum())
                              - diag[current_state.data])))

        p *= np.exp(self.obs1[current_state.PT_events +
                    current_state.Seeding,].sum())

        # reduce to met events
        state = state.MT
        for e in order_2:
            state.add(int(e//2))
        current_state = RestrState(
            (i for i, e in enumerate(state)
             if e in current_state.MT_events + current_state.Seeding),
            restrict=state)

        diag = self._get_diag_unpaired(state=state)
        p /= (np.exp(self.obs2[current_state.events,].sum())
              - diag[current_state.data])

        for i, e in enumerate(order_2):
            e = e//2
            current_state.add(list(state).index(e))
            p *= (np.exp(self.log_theta[e, current_state.events].sum())
                  / (np.exp(self.obs2[current_state.events,].sum())
                     - diag[current_state.data]))
            pass

        p *= np.exp(self.obs2[current_state.events,].sum())

        return p

    def _likelihood_pt_mt(self, order: tuple[int]) -> float:
        """This function returns the probability of observing a specific
        order of events after two observations (timepoint of first
        observation does not matter)

        Args:
            order (tuple[int]): Sequence of events

        Returns:
            float: Probability of observing this order.
        """
        return sum(
            self._likelihood_pt_mt_timed(o1, o2) for o1, o2 in get_combos(
                order=np.array(order), n=self.n, first_obs="PT"))

    def _likelihood_mt_pt_timed(self, order_1: np.array, order_2: np.array
                                ) -> float:
        """ Compute the likelihood of two orders of events happening
        before the first and the second observation 

        Args:
            order_1 (np.array): Order of events (2i and 2i+1 encode the
            ith events happening in PT and Met respectively) that have
            happened when the first observation has been made. Note that
            these do not correspond to the actual PT observation, as it
            is possible that events have happened in the metastasis that
            are not visible in the PT observation.
            order_2 (np.array): Order of events (2i and 2i+1 encode the
            ith events happening in PT and Met respectively) that have
            happened when the second observation has been made. Note
            that these do not correspond to the actual Metobservation,
            as it is possible that events have happened in the primary
            tumor that are not visible in the Met observation.

        Returns:
            float: likelihood of these two orders happening
        """
        # translate first observation to state
        state = MetState(order_1, size=2 * self.n + 1)
        diag = get_diag_paired(log_theta=self.log_theta,
                               n=self.n, state=state.to_seq())

        p = 1 / (1 - diag[0])

        current_state = RestrMetState(0, restrict=state)

        for i, e in enumerate(order_1):
            if not current_state.Seeding:
                # if the seeding has not happened yet, every second
                # event is just the second part of the joint development
                if i % 2:
                    continue
                if e == 2 * self.n:  # seeding
                    current_state.add(len(state) - 1)
                    p *= (np.exp(self.log_theta[
                        self.n,
                        current_state.PT_events + current_state.Seeding
                    ].sum())
                        / (np.exp(self.obs1[
                            current_state.PT_events + current_state.Seeding,
                        ].sum())
                        + np.exp(self.obs2[
                            current_state.MT_events + current_state.Seeding,
                        ].sum())
                        - diag[current_state.data]))
                else:
                    _event = list(state).index(e)
                    current_state.add(_event)
                    current_state.add(_event + 1)
                    p *= (np.exp(self.log_theta[
                        e // 2, current_state.PT_events + current_state.Seeding
                    ].sum())
                        / (np.exp(self.obs1[
                            current_state.PT_events + current_state.Seeding,
                        ].sum())
                        - diag[current_state.data]))
            else:  # seeded
                current_state.add(list(state).index(e))
                if not e % 2:  # PT event
                    p *= (np.exp(self.log_theta[
                        e//2, current_state.PT_events].sum())
                        / (np.exp(self.obs1[
                            current_state.PT_events + current_state.Seeding,
                        ].sum())
                        + np.exp(self.obs2[
                            current_state.MT_events + current_state.Seeding,
                        ].sum())
                        - diag[current_state.data]))
                else:  # Met event
                    p *= (np.exp(self.log_theta[
                        e//2, current_state.MT_events + current_state.Seeding
                    ].sum())
                        / (np.exp(self.obs1[
                            current_state.PT_events + current_state.Seeding,
                        ].sum())
                        + (np.exp(self.obs2[
                            current_state.MT_events + current_state.Seeding,
                        ].sum())
                            - diag[current_state.data])))

        p *= np.exp(self.obs2[current_state.MT_events + current_state.Seeding,
                              ].sum())

        # reduce to pt events
        state = state.PT
        for e in order_2:
            state.add(int(e//2))
        current_state = RestrState(
            (i for i, e in enumerate(state) if e in current_state.PT_events),
            restrict=state)

        diag = self._get_diag_unpaired(state=state, seeding=False)
        p /= (np.exp(self.obs1[current_state.events + (self.n,),].sum())
              - diag[current_state.data])

        for i, e in enumerate(order_2):
            e = e//2
            current_state.add(list(state).index(e))
            p *= (np.exp(self.log_theta[e, current_state.events].sum())
                  / (np.exp(self.obs1[current_state.events + (self.n,),].sum())
                     - diag[current_state.data]))
            pass

        p *= np.exp(self.obs1[current_state.events + (self.n,),].sum())

        return p

    def _likelihood_mt_pt(self, order: tuple[int]) -> float:
        """This function returns the probability of observing a specific
        order of events after two observations (timepoint of first
        observation does not matter)

        Args:
            order (tuple[int]): Sequence of events

        Returns:
            float: Probability of observing this order.
        """
        return sum(
            self._likelihood_mt_pt_timed(o1, o2) for o1, o2 in get_combos(
                order=np.array(order), n=self.n, first_obs="Met"))

    def _likelihood_sync(self, order: np.array) -> float:
        state = MetState(order, size=2 * self.n + 1)
        diag = get_diag_paired(
            log_theta=self.log_theta, n=self.n, state=state.to_seq())

        p = 1 / (1 - diag[0])
        current_state = RestrMetState(0, restrict=state)

        for i, event in enumerate(order):
            if not current_state.Seeding:
                # if the seeding has not happened yet, every second
                # event is just the second part of the joint development
                if i % 2:
                    continue
                if event == 2 * self.n:  # seeding
                    current_state.add(len(state) - 1)
                    obs1 = np.exp(
                        self.obs1[current_state.PT_events + current_state.Seeding,].sum())
                    obs2 = np.exp(
                        self.obs2[current_state.MT_events + current_state.Seeding,].sum())

                    p *= (np.exp(self.log_theta[
                        self.n, current_state.PT_events + current_state.Seeding].sum())
                        / (obs1 + obs2 - diag[current_state.data]))
                else:
                    _event = list(state).index(event)
                    current_state.add(_event)
                    current_state.add(_event + 1)
                    obs1 = np.exp(self.obs1[
                        current_state.PT_events,].sum())
                    p *= (np.exp(self.log_theta[
                        event // 2, current_state.PT_events].sum())
                        / (obs1 - diag[current_state.data]))
            else:
                current_state.add(list(state).index(event))
                obs1 = np.exp(self.obs1[
                    current_state.PT_events + current_state.Seeding,].sum())
                obs2 = np.exp(self.obs2[
                    current_state.MT_events + current_state.Seeding,].sum())
                p *= (np.exp(self.log_theta[
                    event // 2,
                    current_state.MT_events + current_state.Seeding
                    if event % 2 else
                    current_state.PT_events
                ].sum())
                    / (obs1 + obs2 - diag[current_state.data]))
        obs1 = np.exp(self.obs1[
            current_state.PT_events + current_state.Seeding,].sum())
        obs2 = np.exp(self.obs2[
            current_state.MT_events + current_state.Seeding,].sum())
        p *= (obs1 + obs2)
        return p

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
                                  self._get_diag_unpaired(state=state)[-1])
        while t < t_obs:
            probs = [
                np.exp(self.log_theta[
                    e, events[state.astype(bool)]].sum()
                    + self.log_theta[e, e])
                for e in events[~state.astype(bool)]]
            ps = sum(probs)
            probs = [p/ps for p in probs]
            e = np.random.choice(events[~state.astype(bool)], p=probs)
            state[e] = 1
            order.append(e)
            if state.sum() == n:
                break
            t += np.random.exponential(-1 /
                                       self._get_diag_unpaired(state=state)[-1])
        return order, t_obs


if __name__ == "__main__":
    import pandas as pd

    log_theta = pd.read_csv(
        "../metmhn-analyses/2024-01-metmhn-paper/data/luad/luad_g14_0005.csv", index_col=0)
    obs1 = log_theta.iloc[0].to_numpy()
    obs2 = log_theta.iloc[1].to_numpy()

    log_theta = log_theta.drop(index=[0, 1]).to_numpy()
    mmhn = MetMHN(log_theta=log_theta, obs1=obs1, obs2=obs2)

    state = MetState([56, 32, 6, 0],
                     size=log_theta.shape[1] * 2 - 1)

    # print(mmhn.likeliest_order(state, met_status="isPaired", first_obs="Met"))
    print(mmhn._likelihood_mt_pt((56, 6, 0, 32)))
    # print(get_combos(np.array([42, 1, 12, 13, 30]), n=mmhn.n, first_obs="Met"))
    # print(mmhn._likelihood_mt_pt_timed(np.array([ 0,  1, 42,  2]), np.array([])))
