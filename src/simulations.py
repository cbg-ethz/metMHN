import numpy as np

def single_traject(theta, th_prim, b_rates, t_obs, prim, met, n, rng):
    pre_seeding_events = np.zeros_like(prim)
    j_prim = int(prim.sum())
    j_met = int(met.sum()) 
    t = 0.
    inds = np.arange(0, 2*n, dtype=int)
    full_prim = np.zeros((n,n))
    full_met = np.zeros((n,n))
    while True:
        # Seeding didn't happen yet
        if prim[-1] == 0:
            rates = np.exp(theta @ prim + b_rates)
            rates[prim == 1] = 0.
            out_rate = np.sum(rates)
            t += rng.exponential(scale = 1/out_rate, size = 1)
            if (t >= t_obs): break
            next_event = rng.choice(inds[:n], size = 1, p = rates/out_rate)
            prim[next_event] = 1
            met[next_event] = 1
            pre_seeding_events[next_event] = 1
            full_prim[j_prim, next_event] = 1
            full_met[j_met, next_event] = 1
            j_prim += 1
            j_met += 1
        # Seeding already happened
        else:
            prim_rates = np.exp(th_prim @ prim + b_rates)
            prim_rates[prim == 1] = 0.
            met_rates = np.exp(theta @ met + b_rates)
            met_rates[met == 1] = 0.
            out_rate = np.sum(prim_rates) + np.sum(met_rates)
            t += rng.exponential(scale = 1/out_rate, size = 1)
            if(t >= t_obs): break
            next_event =  rng.choice(inds,
                                    size = 1,
                                    p = np.concatenate((prim_rates, met_rates))/out_rate)
            if next_event >= n:
                met[next_event - n] = 1
                full_met[j_met, next_event-n] = 1
                j_met += 1
            else:
                prim[next_event] = 1
                full_prim[j_prim, next_event] = 1
                j_prim += 1
    return prim, met, pre_seeding_events, full_prim, full_met


def sample_metmhn(theta, th_prim, b_rates, lam1, lam2, n, rng):
    t_1 = rng.exponential(scale = 1/lam1, size = 1) 
    t_2 = rng.exponential(scale = 1/lam2, size = 1)
    prim = np.zeros(n)
    met = np.zeros(n)
    # Simulate until 1st diagnosis
    prim, met, pre_seeding_events, full_prim, full_met = single_traject(theta, th_prim, b_rates, t_1, prim, met, n, rng)
    # Record the state of the primary at 1st diagnosis
    prim_obs = prim.copy()
    met_obs = met.copy()
    # Simulate until 2nd diagnosis, if the seeding happened
    if prim[-1] == 1:
        # Record only the state of the met at 2nd diagnosis 
        prim, met_obs, tmp, full_prim2, full_met2 = single_traject(theta, th_prim, b_rates, t_2, prim, met, n, rng)
        full_met +=  full_met2
    # Concatenate the prim and met in the shape metMHN expects
    return np.vstack((prim_obs, met_obs)).flatten(order = "F")[:-1], t_2, pre_seeding_events, full_prim, full_met


def simulate_dat(theta_in, n_dat, lam1, lam2, rng):
    n = theta_in.shape[0]
    dat = np.zeros((n_dat, 2*n-1))
    ages = np.zeros(n_dat)
    pre_seeding_probs = np.zeros(n)

    theta = theta_in.copy()
    b_rates = np.diag(theta_in)
    theta[np.diag_indices(n)] = 0.0
    th_prim = theta.copy()
    th_prim[0:-1, -1] = 0.0

    i = 0
    while i < n_dat:
        datum, age, psp, full_prim, full_met = sample_metmhn(theta, th_prim, b_rates, lam1, lam2, n, rng)
        if datum.sum() > 0:
            dat[i,:] = datum
            ages[i] = age
            pre_seeding_probs += psp
            i += 1
    dat = dat.astype(int)
    return dat, ages, pre_seeding_probs/n_dat 


def p_mut_pre_seed(theta_in, n_dat, lam1, lam2, rng):
    n = theta_in.shape[0]
    pre_seeded_muts = np.zeros(n-1)
    total_muts = np.zeros(n-1)

    theta = theta_in.copy()
    b_rates = np.diag(theta_in)
    theta[np.diag_indices(n)] = 0.0
    th_prim = theta.copy()
    th_prim[0:-1, -1] = 0.0
    i = 0
    while i < n_dat:
        datum, age, psp, full_prim, full_met = sample_metmhn(theta, th_prim, b_rates, lam1, lam2, n, rng)
        if datum[-1] == 1:
            both =(datum[:-1:2]+datum[1::2]==2)
            pre_seeded_muts += psp[:-1] * both
            total_muts += both
            i += 1
    return pre_seeded_muts, total_muts 


def p_mut_pre_seed_2(theta_in, n_dat, lam1, lam2, rng):
    n = theta_in.shape[0]
    pre_seeded_muts = np.zeros(n-1)
    total_muts_prim = np.zeros(n-1)
    total_muts_met = np.zeros(n-1)

    theta = theta_in.copy()
    b_rates = np.diag(theta_in)
    theta[np.diag_indices(n)] = 0.0
    th_prim = theta.copy()
    th_prim[0:-1, -1] = 0.0
    i = 0
    while i < n_dat:
        datum, age, psp, full_prim, full_met = sample_metmhn(theta, th_prim, b_rates, lam1, lam2, n, rng)
        if datum[-1] == 1:
            pre_seeded_muts += psp[:-1]
            total_muts_prim += datum[:-1:2]
            total_muts_met +=  datum[1::2]
            i += 1
    return pre_seeded_muts, total_muts_prim, total_muts_met


def p_full_orders(theta_in, n_dat, lam1, lam2, rng):
    n = theta_in.shape[0]
    prim_muts = np.zeros((n,n))
    total_prims = np.zeros(n)
    met_muts = np.zeros((n,n))
    total_mets = np.zeros(n)
    theta = theta_in.copy()
    b_rates = np.diag(theta_in)
    theta[np.diag_indices(n)] = 0.0
    th_prim = theta.copy()
    th_prim[0:-1, -1] = 0.0
    i = 0
    while i < n_dat:
        datum, age, psp, full_prim, full_met = sample_metmhn(theta, th_prim, b_rates, lam1, lam2, n, rng)
        if datum[-1] == 1:
            prim_muts += full_prim
            met_muts +=  full_met
            total_prims += datum[::2]
            total_mets +=  np.concatenate((datum[1::2], np.array([1])))
            i += 1
    return prim_muts/total_prims, met_muts/total_mets 