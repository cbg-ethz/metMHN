from metmhn.regularized_optimization import learn_mhn, score
from joblib import Parallel, delayed
from itertools import chain, combinations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging
from typing import Callable
jax.config.update("jax_enable_x64", True)


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
    

def marg_frequs(dat: jnp.ndarray,  events: list) -> pd.DataFrame:
    """This calculates the empirical marginal frequencies of mutations in the stratified datasets

    Args:
        dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), 
            rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, 
            the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        events (list): List of genomic events of interest

    Returns:
        pd.DataFrame: marginal probabilities
    """
    n_mut = (dat.shape[1]-3)//2
    n_tot = n_mut + 1
    arr = dat[dat[:,-1]==3,:-2] * np.array([1,2]*n_mut+[1])
    arr = arr @ (np.diag([1,0]*n_mut+[1]) + np.diag([1,0]*n_mut, -1))
    counts = np.zeros((6, n_tot))
    _, sizes = np.unique(dat[:,-1], return_counts=True)
    for i in range(0,2*n_tot,2):
        i_h = int(i/2)
        for j in range(1,4):
            counts[j-1, i_h] = np.count_nonzero(arr[:,i]==j)/sizes[3]
        counts[3, i_h] = np.sum(dat[dat[:,-1]==0, i], axis=0)/sizes[0]
        counts[4, i_h] = np.sum(dat[dat[:,-1]==1, i], axis=0)/sizes[1]
        counts[5, i_h] = np.sum(dat[dat[:,-1]==2, i+1], axis=0)/sizes[2]

    labels = [["Coupled ("+str(sizes[3])+")"]*3 +\
            ["NM ("+str(sizes[0])+")"] +\
            ["EM-PT ("+str(sizes[1])+")"] +\
            ["EM-MT ("+str(sizes[2])+")"],
            ["PT-Private", "MT-Private", "Shared"] + ["Present"]*3]
       
    inds =  pd.MultiIndex.from_tuples(list(zip(*labels)))
    counts = pd.DataFrame(np.around(counts, 2), columns=events, index=inds).T

    return counts


def indep(dat: jnp.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """This function generates an initial estimate for theta and d_p and d_m

    Args:
        dat (jnp.ndarray): dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), 
            rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, 
            the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: th_init, d_p_init, d_m_init
    """
    n_coupled = dat[dat[:,-1]==3, -1].shape[0]
    n = (dat.shape[1] - 3)//2
    n_single = dat.shape[0] - n_coupled
    theta = np.zeros((n + 1, n + 1))
    for i in range(n):
        mut_count = np.sum(dat[:,2*i] + dat[:, 2*i+1])
        if mut_count == 0:
            theta[i, i] = -1e10
        else:
            theta[i,i] = np.log(mut_count/(2 * n_coupled + n_single - mut_count + 1e-10))
    seed_count = np.sum(dat[:,-3])
    theta[n,n] = np.log(seed_count/(n_coupled + n_single - seed_count + 1e-10))
    return theta, np.zeros(n+1), np.zeros(n+1)


def cross_val(dat: jnp.ndarray, penal_fun: Callable, splits: jnp.ndarray, n_folds: int, 
              m_p_corr: float, n_jobs: int=1, seed: int = 42) -> float:
    """Perform a n_folds cross validation for hyperparameter search

    Args:
        dat (jnp.ndarray): dat (jnp.ndarray): Matrix of observations dimension (n_dat x (2n+3)), 
            rows correspond to patients and columns to events.
            The first 2n+1 colummns are expected to be binary and inidacte the status of events of the tumors, 
            the next column contains the observation order 
            (0: unknown, 1: First PT then MT, 2: First MT then PT) and the last column indicates the type of the datapoint 
            (0: PT only, no MT observed, 1: PT only, MT recorded but not sequenced, 2: MT, No PT sequenced, 3: PT and MT sequenced)
        penal_fun Callable[[np.ndarray, int], tuple[np.ndarray, np.ndaray]]): Penalization function, 
            should take a parametervector params and total number of events as input and 
            return the value of the penality and the gradient of it wrt. to all model parameters
        splits (jnp.ndarray): Vector of penalization weights to test
        n_folds (int): Number of folds to split the data into
        m_p_corr (float):  Expected percentage of metastasizing tumor in the Dataset
        n_jobs (int, optional): Number of jobs to run. Defaults to 1.
        seed (int, optional): Seed for random number generator. Defaults to 42.

    Returns:
        float: Best hyperparameter
    """
    key = jrp.PRNGKey(seed)
    shuffled =  jrp.permutation(key, dat, axis=0)
    runs_constrained = np.zeros((n_folds, splits.shape[0]))
    batch_size = jnp.ceil(dat.shape[0]/n_folds)
    
    logging.info(f"Crossvalidation started")
    
    def calc_folds(fold_index, batch_size, shuffled, penal_fun, split):
        jax.config.update("jax_enable_x64", True)
        n_dat = shuffled.shape[0]
        start = batch_size * fold_index
        stop = jnp.min(jnp.array([batch_size*(fold_index + 1), n_dat]))
        
        train_inds = jnp.concatenate((jnp.arange(start, dtype=jnp.int8), 
                                      jnp.arange(stop, n_dat, dtype=jnp.int8)))
        train = shuffled[train_inds,:]
        th_init, fd_init, sd_init = indep(train)
        th, dp, dm = learn_mhn(th_init, fd_init, sd_init, train, m_p_corr, penal_fun, split)
        
        test_inds = jnp.arange(start, stop, dtype=jnp.int8)
        test = shuffled[test_inds, :]

        return score(th, dp, dm, test, m_p_corr)
        
    for j in range(splits.size):
        runs_constrained[:, j] = Parallel(n_jobs=n_jobs)(delayed(calc_folds)(i, batch_size, shuffled, penal_fun, splits[j]) for i in range(n_folds))
        logging.info(f"Finished split {j} out of {splits.size}")
    
    penal_scores = runs_constrained.mean(axis=0)
    best_score = np.max(penal_scores)
    best_penal = splits[np.argmax(best_score)]
    logging.info(f"{penal_scores}")
    logging.info(f"Crossvalidation finished")
    logging.info(f"Highest likelihood averaged over all folds: {best_score}")
    logging.info(f"Best Lambda: {best_penal}")
    
    return best_penal


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