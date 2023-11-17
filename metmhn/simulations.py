import numpy as np

def single_traject(
        log_theta: np.ndarray,
        d_effects: np.ndarray,
        rng: np.random.Generator,
        state: np.ndarray=None,
        second_s: bool=False,
        ) -> np.ndarray:
    """Sample a trajectory to first diagnosis or from first diagnosis to second diagnosis

    Args:
        log_theta (np.ndarray): Theta matrix with logarithmic entries
        d_effects (np.ndarray): Effects of mutations in diagnosis rate
        rng (np.random.Generator): Random number generator
        state (np.ndarray, optional): Starting state of a tumor (PT-MT) sequence
        second_s (bool, optional): If true MT-state influences diag-rate, 
            else PT-state influences diagnosis rate. Defaults to False.

    Returns:
        np.ndarray: PT-MT-state, Pre Seeding Muts
    """
    b_rates = np.diag(log_theta.copy())
    log_theta_prim = log_theta.copy()
    log_theta_prim[:-1,-1] = 0.
    n = log_theta.shape[0]
    r_rates = np.zeros(2*n+1, dtype=np.float64)
    inds = np.arange(0, 2*n+1, dtype=np.int8)
    pre_seeding = np.zeros(n, dtype=np.int8)
    if state is None:
        state = np.zeros(2*n+1, dtype=np.int8)
    else:
        state[-1] = 0
    while True:
        # Seeding didn't happen yet
        if state[n-1] == 0:
            r_rates[-1] = np.exp(np.dot(d_effects, state[:n]))
            r_rates[:n] = np.exp(log_theta @ state[:n] + b_rates)
            r_rates[:n] *= (1-state[:n])
            out_rate = np.sum(r_rates)
            next_event = rng.choice(inds, size=1, p=r_rates/out_rate)
            # Tumor is diagnosed, trajectory stops
            if next_event == 2*n:
                break
            else:
                state[[next_event, next_event+n]] = 1
                pre_seeding[next_event] = 1
        # Seeding already happened
        else:
            if second_s:
                # Scale diagnosis rate by the effects of the MT
                r_rates[-1] = np.exp(np.dot(d_effects, state[n:-1]))
            else:
                # Scale diagnosis rate by the effects of the PT
                r_rates[-1] = np.exp(np.dot(d_effects, state[:n]))
            
            r_rates[:n] = np.exp(log_theta_prim @ state[:n] + b_rates)
            r_rates[:n] *= (1-state[:n])

            r_rates[n:-1] = np.exp(log_theta @ state[n:-1] + b_rates)
            r_rates[n:-1] *= (1-state[n:-1])

            out_rate = np.sum(r_rates)
            next_event = rng.choice(inds, size=1, p=r_rates/out_rate)
            state[next_event] = 1
        # Tumor is diagnosed, trajectory stops
        if state[-1] == 1:
            break
    return state, pre_seeding


def sample_metmhn(
        log_theta: np.ndarray,
        fd_effects: np.ndarray,
        sd_effects: np.ndarray,
        rng: np.random.Generator
        ) -> np.ndarray:
    """Simulate a full trajectory from metMHN

    Args:
        log_theta (np.ndarray): Theta matrix with logarithmic entries
        fd_effects (np.ndarray): Effects of mutations in the PT on the rate of diagnosis
        sd_effects (np.ndarray): Effects of mutations in the MT on the rate of diagnosis
        rng (np.random.Generator): Random number generator

    Returns:
        np.ndarray: Full PT+MT state, preseeding muts
    """
    # Simulate until 1st diagnosis
    n = log_theta.shape[0]
    state, ps = single_traject(log_theta, fd_effects, rng,
                               state=None, second_s=False)
    # Record the state of the primary at 1st diagnosis
    prim_obs = state[:n].copy()
    # Simulate until 2nd diagnosis, if the seeding happened
    if state[n-1] == 1:
        state, _ = single_traject(log_theta, sd_effects, rng,
                                  state=state, second_s=True)
    # Concatenate the prim and met in the shape metMHN expects
    ret = np.vstack((prim_obs, state[n:-1])).flatten(order='F')
    return ret[:-1], ps


def simulate_dat(
    theta_in: np.ndarray,
    fd_effects: np.ndarray,
    sd_effects: np.ndarray,
    n_dat: int,
    rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a dataset of n_dat observations from metMHN

    Args:
        theta_in (np.ndarray): Theta matrix with logarithmic entries
        fd_effects (np.ndarray): Effects of mutations in the PT on the rate of first diagnosis
        sd_effcets (np.ndarray): Effects of mutations int the MT on the rate of second diagnosis
        n_dat (int): Number of samples to generate
        rng (np.random.Generator]): Numpy Random number generator object

    Returns:
        tuple[np.ndarray, np.ndarray]: Dataset of observations, 
            Counts of mutations that happened prior to the seeding
    """
    n = theta_in.shape[0]
    dat = np.zeros((n_dat, 2*n-1), dtype=np.int8)
    pre_seeding_counts = np.zeros(n, dtype=np.int32)
    i = 0
    for i in range(n_dat):
        datum, ps = sample_metmhn(theta_in, fd_effects, sd_effects, rng)
        dat[i,:] = datum
        pre_seeding_counts += ps

    return dat, pre_seeding_counts[:-1]


def p_shared_mut_pre_seed(theta: np.ndarray, 
                          fd_effects: np.ndarray, 
                          sd_effects: np.ndarray, 
                          n_dat: int, 
                          rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    n = theta.shape[0]
    pre_seeded_muts = np.zeros(n-1, dtype=np.int32)
    total_muts = np.zeros(n-1, dtype=np.int32)
    i = 0
    while i < n_dat:
        datum, psp = sample_metmhn(theta, fd_effects, sd_effects, rng)
        if datum[-1] == 1:
            both = (datum[:-1:2]+datum[1::2]==2)
            pre_seeded_muts += psp[:-1] * both
            total_muts += both
            i += 1
        else:
            pass
    return pre_seeded_muts, total_muts 


def p_any_mut_pre_seed(theta: np.ndarray, 
                       fd_effects: np.ndarray, 
                       sd_effects: np.ndarray, 
                       n_dat: int, 
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = theta.shape[0]
    pre_seeded_muts = np.zeros(n-1, dtype=np.int32)
    total_muts_prim = np.zeros(n-1, dtype=np.int32)
    total_muts_met = np.zeros(n-1, dtype=np.int32)
    i = 0
    while i < n_dat:
        datum, psp = sample_metmhn(theta, fd_effects, sd_effects, rng)
        if datum[-1] == 1:
            pre_seeded_muts += psp[:-1]
            total_muts_prim += datum[:-1:2]
            total_muts_met +=  datum[1::2]
            i += 1     
    return pre_seeded_muts, total_muts_prim, total_muts_met