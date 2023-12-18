import numpy as np

def single_traject(
        log_theta: np.ndarray,
        pt_d_ef: np.ndarray,
        mt_d_ef: np.ndarray,
        rng: np.random.Generator,
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
    n_d = n+1
    # 0:(n-1) PT mut. rates, (n-1) Seeding rate, (n) PT diag. rate
    # (n+1):2n+1 MT mut. rates, 2n+1 MT diag. rate
    r_rates = np.zeros(2*n+2, dtype=np.float64)
    inds = np.arange(0, 2*n+2, dtype=np.int8)
    pre_seeding = np.zeros(n, dtype=np.int8)
    state = np.zeros(2*n+2, dtype=np.int8)
    obs_order = 0.
    while True:
        # Seeding didn't happen yet
        if state[n-1] == 0:
            r_rates[n] = np.exp(np.dot(pt_d_ef, state[:n]))
            r_rates[:n] = np.exp(log_theta @ state[:n] + b_rates)
            r_rates[:n] *= (1-state[:n])
            out_rate = np.sum(r_rates)
            next_event = rng.choice(inds, size=1, p=r_rates/out_rate)
            # Tumor is diagnosed, trajectory stops
            if next_event == n:
                break
            else:
                state[[next_event, next_event+n_d]] = 1
                pre_seeding[next_event] = 1
        # Seeding already happened
        else:
            # Calculate MT-diag. rate as function of MT-state
            r_rates[-1] = np.exp(np.dot(mt_d_ef, state[n_d:-1]))
            # Calculate PT-diag. rate as function of PT-state
            r_rates[n] = np.exp(np.dot(pt_d_ef, state[:n]))
            
            # If only the PT is diagnosed only its progression is halted
            r_rates[:n] = np.exp(log_theta_prim @ state[:n] + b_rates)*(1-state[n])
            r_rates[:n] *= (1-state[:n])

            # If only the MT is diagnosed only its progression is halted
            r_rates[n_d:-1] = np.exp(log_theta @ state[n_d:-1] + b_rates)*(1-state[-1])
            r_rates[n_d:-1] *= (1-state[n_d:-1])

            out_rate = np.sum(r_rates)
            next_event = rng.choice(inds, size=1, p=r_rates/out_rate)
            state[next_event] = 1
            if state[n] == 1 and state[-1]==0:
                obs_order = 1
            elif state[n] == 0 and state[-1] == 1:
                obs_order = 2
            else:
                pass
        # Both tumors are diagnosed, trajectory stops
        if state[-1] + state[n] == 2:
            break
    state = state.reshape((-1,2), order="F").flatten()
    return state[:-3], pre_seeding, obs_order


def simulate_dat(
    theta_in: np.ndarray,
    pt_d_ef: np.ndarray,
    mt_d_ef: np.ndarray,
    n_dat: int,
    rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
    """Simulate a dataset consisting of observations and PT/MT observation times

    Args:
        theta_in (np.ndarray): Theta matrix with logarithmic entries
        pt_d_ef (np.ndarray): Effects of PT-Mutations on its diagnosis
        mt_d_ef (np.ndarray): Effects of MT-Mutations on its diagnosis
        n_dat (int): number of datapoints to simulate
        rng (np.random.Generator): Numpy Random Number Generator

    Returns:
        tuple[np.ndarray, np.ndarray]: dataset and pre-seeded muts
    """
    n = theta_in.shape[0]
    dat = np.zeros((n_dat, 2*n), dtype=np.int8)
    pre_seeding_counts = np.zeros(n, dtype=np.int32)
    for i in range(n_dat):
        datum, ps, obs_order = single_traject(theta_in, pt_d_ef, mt_d_ef, rng)
        dat[i,:-1] = datum
        dat[i, -1] = obs_order
        pre_seeding_counts += ps

    return dat, pre_seeding_counts[:-1]


def p_shared_mut_pre_seed(theta: np.ndarray, 
                          pt_d_ef: np.ndarray, 
                          mt_d_ef: np.ndarray, 
                          n_dat: int, 
                          rng: np.random.Generator
                          ) -> tuple[np.ndarray, np.ndarray]:
    n = theta.shape[0]
    pre_seeded_muts = np.zeros(n-1, dtype=np.int32)
    total_muts = np.zeros(n-1, dtype=np.int32)
    i = 0
    while i < n_dat:
        datum, psp, _ = single_traject(theta, pt_d_ef, mt_d_ef, rng)
        if datum[-1] == 1:
            both = (datum[:-2:2]+datum[1:-1:2]==2)
            pre_seeded_muts += psp[:-1] * both
            total_muts += both
            i += 1
        else:
            pass
    return pre_seeded_muts, total_muts 


def p_any_mut_pre_seed(theta: np.ndarray, 
                       pt_d_ef: np.ndarray, 
                       mt_d_ef: np.ndarray, 
                       n_dat: int, 
                       rng: np.random.Generator
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = theta.shape[0]
    pre_seeded_muts = np.zeros(n-1, dtype=np.int32)
    total_muts_prim = np.zeros(n-1, dtype=np.int32)
    total_muts_met = np.zeros(n-1, dtype=np.int32)
    i = 0
    while i < n_dat:
        datum, psp, _ = single_traject(theta, pt_d_ef, mt_d_ef, rng)
        if datum[-1] == 1:
            pre_seeded_muts += psp[:-1]
            total_muts_prim += datum[:-1:2]
            total_muts_met +=  datum[1::2]
            i += 1     
    return pre_seeded_muts, total_muts_prim, total_muts_met