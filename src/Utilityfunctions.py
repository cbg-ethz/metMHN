from itertools import chain, combinations
import numpy as np
import jax.numpy as jnp
import pandas as pd
import regularized_optimization as reg_opt
import scipy.optimize as opt
import matplotlib.pyplot as plt
import logging
import jx.vanilla as mhn

def state_space(n: int) -> np.array:
    """
    Generates all possible states of size n in lexicographic order
    Args:
        n (int): total number of events
    Returns:
         np.array: complete statespace
    """
    states = np.arange(2**n, dtype=np.uint8).reshape((2**n, 1))
    ret = np.unpackbits(states, axis=1, count=n, bitorder="little")
    return ret #np.array([f'{i:0b}'.zfill(n)[::-1] for i in range(2**n)])


def trunk_states(state: np.array) -> np.array:
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


def ssr_to_fss(state: np.array) -> np.array:
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


def random_theta(n: int, sparsity: float) -> np.array:
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


def reachable_states(n: int) -> np.array:
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


def finite_sample(p_th: np.array, k: int) -> np.array:
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


def split_data(dat: pd.DataFrame) -> tuple:
    """Splits the whole dataset into subsets, based on their type

    Args:
        dat (pd.DataFrame): dataset with tumor genotypes as rows

    Returns:
        tuple: tuple of 4 subsets
    """
    if dat.index.isin([(0, "absent")]).any():
        prim_only =  jnp.array(dat.loc[(0, "absent")].to_numpy(dtype=np.int8))
    else:
        prim_only = None
    if dat.index.isin([(0, "isMetastasis")]).any():
        met_only = jnp.array(dat.loc[(0, "isMetastasis")].to_numpy(dtype=np.int8))
    else:
        met_only = None
    if dat.index.isin([(0, "present")]).any():
        prim_met = jnp.array(dat.loc[(0, "present")].to_numpy(dtype=np.int8))
    else:
        prim_met = None
    if dat.index.isin([(1, "isPaired")]).any():
        coupled = dat.loc[(1, "isPaired")].to_numpy(dtype=np.int8)
        coupled = jnp.array(coupled[coupled.sum(axis=1) > 1, ])
    else:
        coupled = None
    return prim_only, met_only, prim_met, coupled


def cross_val(dat: pd.DataFrame, splits: jnp.array, nfolds: int, start_params: jnp.array, m_p_corr: float, n: int) -> float:
    """Performs nfolds-crossvalidation across a parameter range in splits 

    Args:
        dat (pd.DataFrame): Input data
        splits (jnp.array): hyperparameter range to test
        nfolds (int): number of folds (subgroups) to split dat into
        start_params (jnp.array): parameters for the model
        m_p_corr (float): correction factor to account for poverrepresentation of mets
        n (int): number of mutation events

    Returns:
        float: best hyperparameter
    """
    ndat = dat.shape[0]
    dat = dat.reset_index()
    shuffled = dat.sample(frac=1)
    runs_constrained = np.zeros((nfolds, splits.shape[0]))
    batch_size = np.ceil(ndat/nfolds)
    
    logging.info(f"Crossvalidation started")
    for i in range(nfolds):
        start = batch_size*i
        stop = np.min((batch_size*(i+1), ndat))

        train_inds = np.concatenate((np.arange(start), np.arange(stop, ndat)))
        train = shuffled.iloc[train_inds,:]
        train = train.set_index(["paired", "metaStatus"])
        train_prim_only, train_met_only, train_prim_met, train_coupled = split_data(train)
        
        test_inds = np.arange(start, stop)
        test = shuffled.iloc[test_inds, :]
        test = test.set_index(["paired", "metaStatus"])
        test_prim_only, test_met_only, test_prim_met, test_coupled = split_data(test)
        for j in range(splits.size):
            mhn_diag_con = opt.minimize(reg_opt.value_grad, x0 = start_params,#
                             args = (train_prim_only, train_coupled, train_prim_met, train_met_only, n-1,#
                                     splits[j], splits[j], m_p_corr), method = "L-BFGS-B", jac = True,#
                             options={"maxiter":10000, "disp":False, "ftol":1e-04})

            runs_constrained[i,j] = reg_opt.value_grad(mhn_diag_con.x, test_prim_only, test_coupled, test_prim_met,#
                                                       test_met_only, n-1, 0., 0., m_p_corr)[0]
            
            logging.info(f"Diag constrained, Split: {splits[j]}, Fold: {i}, Score: {runs_constrained[i,j]}")

    diag_penal_scores = runs_constrained.mean(axis=1)
    best_diag_score = np.min(diag_penal_scores)
    best_diag_penal = splits[np.argmin(diag_penal_scores)]
    
    logging.info(f"Crossvalidation finished")
    logging.info(f"Highest likelihood score: {best_diag_score} (Diag penalized)")
    logging.info(f"Best Lambda: {best_diag_penal} (Diag penalized)")
    
    return best_diag_penal

def indep(dat_singles: jnp.array, dat_coupled: jnp.array) -> jnp.array:
    """Generates a diagonal theta matrix with log odds ratios as entries

    Args:
        dat_singles (jnp.array): 2d array with uncoupled datapoints as rows
        dat_coupled (jnp.array): 2d array with coupled datapoints as tows

    Returns:
        jnp.array: independence model
    """
    n = (dat_singles.shape[1] - 1)//2
    n_coupled = dat_coupled.shape[0]
    n_singles = dat_singles.shape[0]
    theta = jnp.zeros((n + 1,n + 1))
    for i in range(n):
        mut_count = jnp.sum(dat_singles.at[:,2*i].get() + dat_singles.at[:, 2*i+1].get())
        mut_count += jnp.sum(dat_coupled.at[:,2*i].get() + dat_coupled.at[:, 2*i+1].get())
        if mut_count == 0:
            theta = theta.at[i,i].set(-1e10)
        else:
            theta = theta.at[i,i].set(jnp.log(mut_count/(2 * n_coupled + n_singles - mut_count + 1e-10)))
    seed_count = jnp.sum(dat_singles.at[:,-1].get()) + n_coupled
    theta = theta.at[n,n].set(jnp.log(seed_count/(n_coupled + n_singles - seed_count + 1e-10)))
    return theta


def plot_theta(theta_in: pd.DataFrame, alpha: float) -> tuple:
    """Plot theta matrix 

    Args:
        theta_in (pd.DataFrame): theta matrix with logarithmic entries
        alpha (float): alpha value for background heatmap coloring

    Returns:
        tuple: tuple of axis objects
    """
    theta = theta_in.copy()
    events = theta.columns
    n = theta.shape[0]
    
    theta[(theta.round(2) == 0) | (theta.round(2) == -0)] = np.nan
    theta_diag = np.diag(theta.copy()).reshape((-1,1))
    theta[theta.abs() <= alpha] = np.nan
    np.fill_diagonal(theta.values, np.nan)

    plt.style.use("ggplot")
    f, (ax, ax2) = plt.subplots(1, 2, figsize=(19,15), gridspec_kw={'width_ratios': [6, 1]})
    f.tight_layout()
    ax.matshow(theta[theta<0], cmap="Blues_r")
    ax.matshow(theta[theta>0], cmap="Reds")
    ax2.matshow(theta_diag, cmap="coolwarm")
    ax.set_xticks(range(theta.shape[1]), events, fontsize=14, rotation=90)
    ax.set_yticks(range(theta.shape[1]), events, fontsize=14)
    ax2.set_yticks(range(theta.shape[1]), events, fontsize=14)
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.set_xticks([])

    for i in range(n):
        for j in range(n):
            if np.isnan(theta.iloc[i,j]) == False:
                c = np.round(theta.iloc[i,j].round(2), 2)
            else:
                c = ""
            ax.text(j, i, str(c), va='center', ha='center')
        ax2.text(0, i, np.round(theta_diag[i,0],3), va='center', ha='center')
    return (ax, ax2)

def p_unobs_seeding(log_theta: jnp.array, lam1: jnp.array,  dat_obs: jnp.array) -> jnp.array:
    """
        Returns the probability that tumor dat_obs has spawned an unobserved metastasis

    Args:
        log_theta (jnp.array): matrix of logarithmic entries of theta
        lam1 (jnp.arry): Rate of first sampling (non logarithmic)
        dat_obs (jnp.array): Primary tumor genotype as bitstring

    Returns:
        jnp.array: _description_
    """
    dat_mod = dat_obs.at[-1].set(1)
    m =  dat_mod.sum()
    p0 = jnp.zeros(2**m)
    p0 = p0.at[0].set(1.0)
    pth = lam1 * mhn.R_inv_vec(log_theta, p0, lam1,  dat_mod, False)
    probs = pth.reshape((-1, 2), order="F").at[-1,:].get()
    return probs.at[0].get() / probs.sum()