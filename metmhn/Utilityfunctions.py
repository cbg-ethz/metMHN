from metmhn.regularized_optimization import learn_mhn, score
from itertools import chain, combinations
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrp
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import logging
from typing import Callable, Any
jax.config.update("jax_enable_x64", True)


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
    counts[5, -1] = np.sum(dat[dat[:,-1]==2, -3], axis=0)/sizes[2]
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


def cross_val(dat: jnp.ndarray, penal_fun, splits: jnp.ndarray, n_folds: int, 
              m_p_corr: float, seed: int = 42) -> tuple:
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
        float, float: Best hyperparameter, standard error of models with the best hyperparameter
    """
    key = jrp.PRNGKey(seed)
    shuffled =  jrp.permutation(key, dat, axis=0)
    runs_constrained = np.zeros((n_folds, splits.shape[0]))
    batch_size = jnp.ceil(dat.shape[0]/n_folds)
    
    logging.info(f"Crossvalidation started")
    for i in range(splits.size):
        for fold_index in range(n_folds):
            n_dat = shuffled.shape[0]
            start = int(batch_size * fold_index)

            stop = int(jnp.min(jnp.array([batch_size*(fold_index + 1), n_dat])))
            train_inds = jnp.concatenate((jnp.arange(start, dtype=jnp.int32), 
                                      jnp.arange(stop, n_dat, dtype=jnp.int32)))
            train = shuffled[train_inds,:]
            th_init, fd_init, sd_init = indep(train)
            th, dp, dm = learn_mhn(th_init, fd_init, sd_init, train, m_p_corr, penal_fun, splits[i], opt_v=False)
        
            test_inds = jnp.arange(start, stop, dtype=jnp.int32)
            
            test = shuffled[test_inds, :]
            runs_constrained[fold_index, i] = score(th, dp, dm, test, m_p_corr)
            logging.info(f"{runs_constrained}")

    mean_scores = runs_constrained.mean(axis=0)
    best_score = np.max(mean_scores)
    blp = np.argmax(mean_scores)
    best_penal = splits[blp]
    se = np.std(splits[:, blp])/np.sqrt(n_folds)
    best_penal_1se = splits[np.max(np.argwhere(mean_scores>(best_score-se)))]
    logging.info(f"{mean_scores}")
    logging.info(f"Crossvalidation finished")
    logging.info(f"Highest likelihood averaged over all folds: {best_score}")
    logging.info(f"Best Lambda: {best_penal}, Best penal + 1 standard error: {best_penal_1se}")
    return best_penal, best_penal_1se


def plot_theta(ax1: plt.Axes, ax2: plt.Axes, model: np.ndarray, events: list,
               alpha: float, cb:Any = None, verbose: bool=True, font_size:int =10) -> tuple:
    """Plot a metMHN-model as a Heatmap

    Args:
        ax1 (plt.Axes): axes object as returned by plt.subfigures
        ax2 (plt.Axes): axes object as returned by plt.subfigures
        model (np.ndarray): (n+2)xn dimenional metMHN model to visualize.
            The first two rows have to be the effects on PT and MT diagnosis, 
            the remaining rows correspond to intergenomic effects.
        events (list): List of size n, containing event names as strings
        alpha (float): Threshold on effect strengths. Effects whose absolute strengths are below alpha, are not plotted.
        cb (Any, optional): Matplotlib compatible colormap
        verbose (bool, optional): If true plot the numeric values of effects strengths in their corresponding cells. 
            Defaults to True.
        font_size (int, optional): Fontsize for captions and annotations. Defaults to 10.

    Returns:
        tuple: Heatmaps on axes 1&2 and their colorbars
    """
    if cb == None:
        # Generated with https://eltos.github.io/gradient/#E69F00-FFFFFF-009E73
        cgr = ((0.000, (0.902, 0.624, 0.000)),
            (0.500, (0.937, 0.976, 0.965)),
            (1.000, (0.000, 0.620, 0.451))
            )
        my_color_gradient = LinearSegmentedColormap.from_list('my_gradient', cgr)

    model_p = model.copy()
    n_total = model_p.shape[1]
    theta = model_p[2:, :]
    th_diag = np.diagonal(theta.copy())
    theta[np.diag_indices(n_total)] = 0.
    model_p = np.row_stack((model_p[:2,:], theta))
    max_c = np.max(np.abs(model_p))
    model_p[np.abs(model_p)<alpha] = np.nan
    th_diag = np.row_stack((np.array([np.nan, np.nan]).reshape((2,1)), 
                            th_diag.reshape(-1,1)))

    events_ext = np.concatenate((np.array(["Obs-PT", "Obs-MT"]), events))
    # Plot off diagonals on one plot
    im1 = ax1.matshow(model_p, cmap=my_color_gradient, vmin=-max_c*1.1, vmax=max_c*1.1)
    ax1.set_xticks(range(n_total), events, fontsize=font_size, rotation=90)
    ax1.set_yticks(range(n_total+2), events_ext, fontsize=font_size)
    
    # Plot grid lines between cells
    ax1.set_xticks(np.arange(-.5, n_total-1, 1), minor=True)
    ax1.set_yticks(np.arange(-.5, n_total+1, 1), minor=True)
    ax1.grid(which="minor",color='grey', linestyle='-', linewidth=1)
    ax1.tick_params(which='minor', bottom=False, left=False, labelbottom=False, labeltop=True)
    ax1.tick_params(which='major', bottom=False, left=True, labelbottom=False, labeltop=True)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")
    cb1 = plt.colorbar(im1, ax=ax1, orientation="horizontal", shrink=2/(n_total), pad=0.03, aspect=8)
    cb1.ax.locator_params(nbins=10)
    # Plot diagonal entries separately
    im2 = ax2.matshow(th_diag, cmap="Blues")
    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.spines[['top', 'right', 'bottom', 'left']].set_visible(False)

    cb2 = plt.colorbar(im2, ax=ax2, orientation="horizontal", shrink=2, pad=0.03, aspect=8)
    cb2.ax.locator_params(nbins=5)
    # Plot numerical values of theta 
    if verbose:
       for i in range(n_total+2):
            for j in range(n_total):
                if np.isnan(model_p[i,j]) == False:
                    c = np.round(model_p[i,j], 2)
                else:
                    c = ""
                ax1.text(j, i, str(c), va='center', ha='center', size=font_size)
            if np.isnan(th_diag[i,0]):
                c = ""
            else:
                c = np.round(th_diag[i,0], 2)
            ax2.text(0, i, str(c), va='center', ha='center', size=font_size)
    return im1, im2, cb1, cb2