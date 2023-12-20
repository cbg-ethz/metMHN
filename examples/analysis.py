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
parser.add_argument("-pm_ratio", action="store", type=float, default="0.5", 
                    help="Expected ratio of PTs to MTs")
parser.add_argument("-lam", action="store", default=1e-04, type=float, 
                    help="Weight of penalization for inference")
parser.add_argument("-logs", action="store", default="./inf.dat", type=str, 
                    help="relative path to log-file destination")
parser.add_argument("source-annot", help="Relative path to the data annotation file")
parser.add_argument("source-data", help="Relative path to the data file")
parser.add_argument("dest", help="Relative path to file save destination")
args = parser.parse_args()
config = vars(args)

import metmhn.regularized_optimization as reg_opt
import metmhn.Utilityfunctions as utils

import pandas as pd
import warnings
warnings.simplefilter(action='ignore', 
                      category=pd.errors.PerformanceWarning)

import jax.numpy as jnp
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

annot_data = pd.read_csv(config['source-annot'])
mut_data = pd.read_csv(config['source-data'])
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], 
               on=["patientID", "patientID"])
muts = list(dat.columns[1:-4])
dat["type"] = dat.apply(utils.categorize, axis=1)
dat["Seeding"] = dat.apply(utils.add_seeding, axis=1)
dat.loc[dat["M.AgeAtSeqRep"] == "No metastasis included", "M.AgeAtSeqRep"] = pd.NA
dat.loc[dat["P.AgeAtSeqRep"] == "No primary included", "P.AgeAtSeqRep"] = pd.NA
dat["P.AgeAtSeqRep"] = dat["P.AgeAtSeqRep"].astype(pd.Int64Dtype())
dat["M.AgeAtSeqRep"] = dat["M.AgeAtSeqRep"].astype(pd.Int64Dtype())
dat["diag_order"] = dat["M.AgeAtSeqRep"] - dat["P.AgeAtSeqRep"]
dat["diag_order"] = dat["diag_order"].apply(lambda x: pd.NA if pd.isna(x) else 2 if x < 0 else 1 if x > 0 else 0) 
dat["diag_order"] = dat["diag_order"].astype(pd.Int64Dtype())
events_data = muts+["Seeding"]
# Only use datapoints where the state of the metastasis is known
cleaned = dat.loc[dat["type"].isin([0,1,2,3]), muts+["Seeding", "type", "diag_order"]]
cleaned.drop(cleaned[cleaned.iloc[:,:-2].sum(axis=1)<1].index, inplace=True)
dat_prim_nomet, dat_prim_met, dat_met_only, dat_coupled = utils.split_data(cleaned, events_data+["diag_order"])

events_plot = []
for elem in cleaned.columns[:-3].to_list()[::2]:
    full_mut_id = elem.split(".")
    events_plot.append(full_mut_id[1])
events_plot.append("Seeding")

if config['cv']:
    log_lams = np.linspace(np.log(config['cv_start']), 
                           np.log(config['cv_end']), 
                           config['cv_splits'])
    lams = 10**log_lams
    penal = utils.cross_val(cleaned.copy(), events_data, lams, 
                            config['cv_folds'], config['pm_ratio'])
else:
    penal = config['lam']

th_init, dp_init, dm_init = utils.indep(jnp.array(cleaned[events_data].to_numpy()), dat_coupled.shape[0])
theta, d_p, d_m= reg_opt.learn_mhn(th_init, dp_init, dm_init, dat_prim_nomet, dat_prim_met, dat_met_only,
                                    dat_coupled, config['pm_ratio'], penal)

th_plot = np.row_stack((d_p.reshape((1,-1)), 
                        d_m.reshape((1,-1)), 
                        theta))

df2 = pd.DataFrame(th_plot, columns=events_plot)
df2.to_csv(config["dest"])