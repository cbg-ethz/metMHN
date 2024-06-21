import argparse

parser = argparse.ArgumentParser(description="Apply metMHN to a specified dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-cv", action="store_true", help="Perform crossvalidation")
parser.add_argument("-cv_start", action="store", default=1e-05, type=float,
                    help="Lower limit of hyperparameter range")
parser.add_argument("-cv_end", action="store", default=1e-02, type=float,
                    help="Upper limit of hyperparameter range")
parser.add_argument("-cv_folds", action="store", default=5, type=int, 
                    help="Number of crossvalidation folds")
parser.add_argument("-cv_splits", action="store", default=5, type=int, 
                    help="Number of hyperparameters to test")
parser.add_argument("-pm_ratio", action="store", type=float, default=None, 
                    help="Expected ratio of PTs to MTs")
parser.add_argument("-lam", action="store", default=1e-04, type=float, 
                    help="Weight of penalization for inference")
parser.add_argument("-logs", action="store", default="./inf.dat", type=str, 
                    help="relative path to log-file destination")
parser.add_argument("-seed", action="store", default=42, type=int, 
                    help="Seed for random number generator")
parser.add_argument("source-annot", help="Relative path to the data annotation file")
parser.add_argument("source-data", help="Relative path to the data file")
parser.add_argument("dest", help="Relative path to file save destination")
args = parser.parse_args()
config = vars(args)

import metmhn.regularized_optimization as reg_opt
import metmhn.Utilityfunctions as utils

import pandas as pd
import warnings
import jax.numpy as jnp
import jax.random as jrp
import numpy as np
import jax as jax
jax.config.update("jax_enable_x64", True)

import logging

logging.basicConfig(filename=config['logs'],
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filemode='w', 
                    level=logging.INFO, 
                    force=True,
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

# Read in raw data
annot_data = pd.read_csv(config['source-annot'])
mut_data = pd.read_csv(config['source-data'])
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], 
               on=["patientID", "patientID"])
muts = list(dat.columns[1:-4])

# Label each datapoint with a numeric value according to its sequencetype
dat["type"] = dat.apply(utils.categorize, axis=1)
# Add the seeding event
dat["Seeding"] = dat["type"].apply(lambda x: pd.NA if pd.isna(x) else 0 if x == 0 else 1)
dat["M.AgeAtSeqRep"] = pd.to_numeric(dat["M.AgeAtSeqRep"], errors='coerce')
dat["P.AgeAtSeqRep"] = pd.to_numeric(dat["P.AgeAtSeqRep"], errors='coerce')
# Define the order of diagnosis for paired datapoints
dat["diag_order"] = dat["M.AgeAtSeqRep"] - dat["P.AgeAtSeqRep"]
dat["diag_order"] = dat["diag_order"].apply(lambda x: pd.NA if pd.isna(x) else 2 if x < 0 else 1 if x > 0 else 0) 
dat["diag_order"] = dat["diag_order"].astype(pd.Int64Dtype())

events_data = muts+["Seeding"]

# Only use datapoints where the state of the seeding is known
cleaned = dat.loc[~pd.isna(dat["type"]), muts+["Seeding", "diag_order", "type"]]
dat = jnp.array(cleaned.to_numpy(dtype=np.int8, na_value=-99))

events_plot = []
for elem in cleaned.columns[:-3].to_list()[::2]:
    full_mut_id = elem.split(".")
    events_plot.append(full_mut_id[1])
events_plot.append("Seeding")

if config["pm_ratio"] is None:
    perc_met = dat[:,-3].sum()/(dat.shape[0] - dat[:,-3].sum())
else:
    perc_met = config["pm_ratio"]

if config['cv']:
    # Perform crossvalidation
    log_lams = np.linspace(np.log10(config['cv_start']), 
                           np.log10(config['cv_end']), 
                           config['cv_splits'])
    lams = 10**log_lams
    penal_weights = utils.cross_val(dat=dat, 
                                penal_fun=reg_opt.symmetric_penal, 
                                splits=lams, 
                                n_folds=config['cv_folds'], 
                                m_p_corr=perc_met, 
                                seed=jrp.PRNGKey(config['seed']))

    # The cross_val function returns a n_folds x log_lams.size shaped dataframe
    penal = lams[np.argmax(np.mean(penal_weights, axis=0))]
else:
    penal = config['lam']


# Learn the actual model
th_init, dp_init, dm_init = utils.indep(dat)
theta, d_p, d_m= reg_opt.learn_mhn(th_init=th_init, 
                                   dp_init=dp_init,
                                   dm_init=dm_init,
                                   dat=dat,
                                   perc_met=perc_met,
                                   penal=reg_opt.symmetric_penal,
                                   w_penal=penal
                                   )

th_plot = np.row_stack((d_p.reshape((1,-1)), 
                        d_m.reshape((1,-1)), 
                        theta))

df2 = pd.DataFrame(th_plot, columns=events_plot)
df2.to_csv(config["dest"])