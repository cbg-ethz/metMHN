from metmhn.regularized_optimization import learn_mhn, log_lik
from itertools import chain, combinations
import numpy as np
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging


my_color_gradient = LinearSegmentedColormap.from_list('my_gradient', (
    # Generated with https://eltos.github.io/gradient/#E69F00-FFFFFF-009E73
    (0.000, (0.902, 0.624, 0.000)),
    (0.500, (0.937, 0.976, 0.965)),
    (1.000, (0.000, 0.620, 0.451)))
    )

def state_space(n: int) -> np.ndarray:
    """
    Generates all possible states of size n in lexicographic order
    Args:
        n (int): total number of events
    Returns:
         np.array: complete statespace
    """
    states = np.arange(2**n, dtype=np.uint8).reshape((2**n, 1))
    ret = np.unpackbits(states, axis=1, count=n, bitorder="little")
    return ret


def trunk_states(state: np.ndarray) -> np.ndarray:
    """
    Enumerates all possible states that a tumor(pair) with genotype(s) "state" could have visited
    Args:
        state (np.array): Bitstring, genotype of a tumor(pair) of a single patient
    Returns:
        np.array
    """
    n = state.size
    inds = np.ones(2**n, dtype=np.uint8)
    for i in range(n):
        if state[i] == 1:
            inds[0:2**(i+1)] = np.kron(np.array([1, 1]), inds[0:2**i])
        else:
            inds[0:2**(i+1)] = np.kron(np.array([1, 0]), inds[0:2**i])
    
    return state_space(n)[inds.astype(bool), :]


def ssr_to_fss(state: np.ndarray) -> np.ndarray:
    """This gives the indices of the rate matrix that are appearing in the
    state space restricted rate matrix.

    Args:
        state (np.array): Binary state vector, representing the current sample's events.
    Returns:
        np.array: Indices of the rate matrix.
    """
    res = np.ones(1)
    for s in state:
        res = np.kron(np.array([1, s]), res)
    return res.astype(bool)


def random_theta(n: int, sparsity: float) -> np.ndarray:
    """
    Generates a logarithmic theta with normal distributed entries
    Args:
        n (int): Number of mutations
        sparsity (float): Percentage of zero entries in theta
    returns:
        np.array: theta
    """
    npone = n + 1
    log_theta = np.zeros((npone, npone))
    log_theta += np.diag(np.random.normal(size=npone))
    index = np.argwhere(log_theta == 0)[
        np.random.choice(npone**2-npone, size=int((npone**2-npone)
                         * (1-sparsity)), replace=True)
    ]
    log_theta[index[:, 0], index[:, 1]] = np.random.normal(
        size=int((npone**2-npone)*(1-sparsity)))
    return log_theta


def reachable_states(n: int) -> np.ndarray:
    """This function returns the indices, w.r.t. to a lexicographical ordering, of the states
    of an MHN with n events, that can actually be reached.

    Args:
        n (int): Number of events

    Returns:
        np.array: dtype bool, True if the state is reachable, False if not.
    """
    reachable = np.zeros(2**(2*n + 1))

    for i in chain.from_iterable(combinations(list(range(n)), r) for r in range((n+1))):
        reachable[sum((2**(2*j))*3 for j in i)] = 1

    reachable[2**(2*n):] = 1
    return reachable.astype(bool)


def finite_sample(p_th: np.ndarray, k: int) -> np.ndarray:
    """
    Generates k random samples drawn from a probability distribution p_th
    Code taken from https://github.com/spang-lab/LearnMHN/blob/main/mhn/original/UtilityFunctions.py
    Args:
        p_th (np.array): Probability distribution to draw samples from
        k (int): Number of samples to draw
    Returns:
        np.array: Subsampled probability distribution
    """
    n = p_th.size
    return np.bincount(np.random.choice(n, k, replace=True, p=p_th), minlength=n) / k


def categorize(x: pd.Series) -> int:
    if x['paired'] == 0:
        if x['metaStatus'] == "absent":
            return 0
        elif x['metaStatus'] == "present":
            return 1
        elif x['metaStatus'] == 'isMetastasis':
            return 2
        elif x['metaStatus'] == 'unknown':
            return 4
        else:
            return -1
    elif x['paired'] == 1:
        return 3
    else:
        return -1


def add_seeding(x: pd.Series) -> int:
    if x['type'] in [1,2,3]:
        return 1
    elif x['type'] == 0:
        return 0
    else:
        return -1
    

def split_data(dat: pd.DataFrame, events: list) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Splits the whole dataset into subsets, based on their type

    Args:
        dat (pd.DataFrame): dataset with tumor genotypes as rows

    Returns:
        tuple: tuple of 4 subsets
    """
    prim_only = jnp.array(dat.loc[dat["type"] == 0, events].to_numpy(dtype=np.int8, na_value = -99))
    prim_met = jnp.array(dat.loc[dat["type"] == 1, events].to_numpy(dtype=np.int8, na_value = -10))
    met_only = jnp.array(dat.loc[dat["type"] == 2, events].to_numpy(dtype=np.int8, na_value = -10))
    coupled = jnp.array(dat.loc[dat["type"] == 3, events].to_numpy(dtype=np.int8, na_value = -10))
    return prim_only, prim_met, met_only, coupled


def marg_frequs(dat_prim_nomet: jnp.ndarray, dat_prim_met: jnp.ndarray, dat_met_only:jnp.ndarray, 
                dat_coupled: jnp.ndarray, events: list) -> pd.DataFrame:
    """This calculates the empirical marginal frequencies of mutations in the stratified datasets

    Args:
        dat_prim_nomet (jnp.ndarray): Dataset of unpaired never met. primary tumors
        dat_prim_met (jnp.ndarray): Dataset of unpaired metastasized primary tumors
        dat_met_only (jnp.ndarray): Dataset of unpaired metastases
        dat_coupled (jnp.ndarray): Dataset of coupled PT-MT pairs
        events (list): List of genomic events of interest

    Returns:
        pd.DataFrame: marginal probabilities
    """
    n_mut = (dat_prim_nomet.shape[1]-1)//2
    n_tot = n_mut + 1
    arr = dat_coupled * np.array([1,2]*n_mut+[1])
    arr = arr @ (np.diag([1,0]*n_mut+[1]) + np.diag([1,0]*n_mut, -1))
    counts = np.zeros((6, n_tot))
    for i in range(0,2*n_tot,2):
        i_h = int(i/2)
        for j in range(1,4):
            counts[j-1, i_h] = np.count_nonzero(arr[:,i]==j)/dat_coupled.shape[0]
        counts[3, i_h] = np.sum(dat_prim_nomet[:, i], axis=0)/dat_prim_nomet.shape[0]
        counts[4, i_h] = np.sum(dat_prim_met[:, i], axis=0)/dat_prim_met.shape[0]
        counts[5, i_h] = np.sum(dat_met_only[:, i+1], axis=0)/dat_met_only.shape[0]

    labels = [["Coupled ("+str(dat_coupled.shape[0])+")"]*3 +\
            ["NM ("+str(dat_prim_nomet.shape[0])+")"] +\
            ["EM-PT ("+str(dat_prim_met.shape[0])+")"] +\
            ["EM-MT ("+str(dat_met_only.shape[0])+")"],
            ["PT-Private", "MT-Private", "Shared"] + ["Present"]*3]
       
    inds =  pd.MultiIndex.from_tuples(list(zip(*labels)))
    counts = pd.DataFrame(np.around(counts, 2), columns=events, index=inds).T

    return counts


def indep(dat: jnp.ndarray, n_coupled: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function generates an initial estimate for theta and d_p and d_m

    Args:
        dat (jnp.ndarray): Dataset
        n_coupled (int): Number of coupled datapoints

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: th_init, d_p_init, d_m_init
    """
    
    n = (dat.shape[1] - 1)//2
    n_single = dat.shape[0] - n_coupled
    theta = np.zeros((n + 1, n + 1))
    for i in range(n):
        mut_count = np.sum(dat.at[:,2*i].get() + dat.at[:, 2*i+1].get())
        if mut_count == 0:
            theta[i, i] = -1e10
        else:
            theta[i,i] = np.log(mut_count/(2 * n_coupled + n_single - mut_count + 1e-10))
    seed_count = np.sum(dat.at[:,-1].get())
    theta[n,n] = np.log(seed_count/(n_coupled + n_single - seed_count + 1e-10))
    return theta, np.zeros(n+1), np.zeros(n+1)


def cross_val(dat: pd.DataFrame, events: list, splits: jnp.ndarray, nfolds: int, m_p_corr: float) -> float:
    """Performs nfolds-crossvalidation across a parameter range in splits 

    Args:
        dat (pd.DataFrame):     Input data
        splits (jnp.ndarray):   Hyperparameter range to test
        nfolds (int):           Number of folds (subgroups) to split dat into
        m_p_corr (float):       Correction factor to account for poverrepresentation of mets

    Returns:
        float: best hyperparameter
    """
    n_dat = dat.shape[0]
    shuffled = dat.sample(frac=1)
    runs_constrained = np.zeros((nfolds, splits.shape[0]))
    batch_size = np.ceil(n_dat/nfolds)
    
    logging.info(f"Crossvalidation started")
    for i in range(nfolds):
        start = batch_size*i
        stop = np.min((batch_size*(i+1), n_dat))

        train_inds = np.concatenate((np.arange(start), 
                                     np.arange(stop, n_dat)))
        train = shuffled.iloc[train_inds,:]
        train_prim_only, train_prim_met, train_met_only, train_coupled = split_data(train, events)
        
        test_inds = np.arange(start, stop)
        test = shuffled.iloc[test_inds, :]
        test_prim_only, test_prim_met, test_met_only, test_coupled = split_data(test, events)
        th_init, fd_init, sd_init = indep(jnp.array(train[events].to_numpy()), test_coupled.shape[1])
        
        for j in range(splits.size):
            th, fd, sd = learn_mhn(th_init, fd_init, sd_init, train_prim_only, 
                                     train_prim_met, train_met_only, train_coupled, 
                                     m_p_corr, splits[j])
            params = np.concatenate((th.flatten(), fd, sd))
            runs_constrained[i,j] = log_lik(params, test_prim_only, test_prim_met, test_met_only,
                                            test_coupled, 0., m_p_corr)
            
            logging.info(f"Split: {splits[j]}, Fold: {i}, Score: {runs_constrained[i,j]}")

    diag_penal_scores = runs_constrained.mean(axis=0)
    best_diag_score = np.min(diag_penal_scores)
    best_diag_penal = splits[np.argmin(diag_penal_scores)]
    
    logging.info(f"Crossvalidation finished")
    logging.info(f"Highest likelihood score: {best_diag_score}")
    logging.info(f"Best Lambda: {best_diag_penal}")
    
    return best_diag_penal


def plot_theta(model: jnp.ndarray, events: jnp.ndarray, 
               alpha: float, verbose=True, font_size=10) -> tuple:
    """Visualize theta, d_m and d_p

    Args:
        model (np.array):   Theta matrix with logarithmic entries
        events (np.array):  Array of event names excluding diagnosis
        alpha (float):      Threshold for effect size

    Returns:
        tuple: tuple of axis objects
    """
    th = model.copy()
    n_total = th.shape[1]
    theta = th[2:, :]
    th_diag = np.diagonal(theta.copy())
    theta[np.diag_indices(n_total)] = 0.
    th = np.row_stack((th[:2,:], theta))
    max_c = np.max(np.abs(th))
    th[np.abs(th)<alpha] = np.nan
    th_diag = np.row_stack((np.array([np.nan, np.nan]).reshape((2,1)), 
                            th_diag.reshape(-1,1)))


    f, (ax, ax2) = plt.subplots(1, 2, figsize=(19,15), sharey="col",
                                gridspec_kw={'width_ratios': [n_total, 1], 
                                             "top":1, "bottom": 0, "right":1, 
                                             "left":0, "hspace":0, "wspace":-0.48})
    events_ext = np.concatenate((np.array(["d_p", "d_m"]), events))
    # Plot off diagonals on one plot
    im1 = ax.matshow(th, cmap=my_color_gradient, vmin=-max_c, vmax=max_c)
    ax.set_xticks(range(n_total), events, fontsize=14, rotation=90)
    ax.set_yticks(range(n_total+2), events_ext, fontsize=14)
    
    # Plot grid lines between cells
    ax.set_xticks(np.arange(-.5, n_total, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_total+2, 1), minor=True)
    ax.grid(which="minor",color='grey', linestyle='-', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False) 

    f.colorbar(im1, ax=ax, orientation="horizontal", shrink=4/n_total, pad=0.03, aspect=8)
    # Plot diagonal entries separately
    im2 = ax2.matshow(th_diag, cmap="Blues")
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    f.colorbar(im2, ax=ax2, orientation="horizontal", shrink=4, pad=0.03, aspect=8)
    # Plot numerical values of theta 
    if verbose:
       for i in range(n_total+2):
            for j in range(n_total):
                if np.isnan(th[i,j]) == False:
                    c = np.round(th[i,j], 2)
                else:
                    c = ""
                ax.text(j, i, str(c), va='center', ha='center', size=font_size)
            if np.isnan(th_diag[i,0]):
                c = ""
            else:
                c = np.round(th_diag[i,0], 2)
            ax2.text(0, i, str(c), va='center', ha='center', size=font_size)
    return f