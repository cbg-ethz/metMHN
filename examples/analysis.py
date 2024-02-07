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
parser.add_argument("-n_jobs", action="store", type=int, default=1, 
                    help="Number of processes to use in crossvalidation. Should be at most equal to the number of folds")
parser.add_argument("-pm_ratio", action="store", type=float, default=None, 
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
#muts = list(dat.columns[1:-4])

muts = ['P.TP53 (M)', 'M.TP53 (M)', 'P.KRAS (M)', 'M.KRAS (M)', 'P.EGFR (M)', 'M.EGFR (M)', 'P.STK11 (M)', 'M.STK11 (M)', 'P.KEAP1 (M)', 'M.KEAP1 (M)', 
        'P.RBM10 (M)', 'M.RBM10 (M)', 'P.SMARCA4 (M)', 'M.SMARCA4 (M)', 'P.ATM (M)', 'M.ATM (M)', 'P.NF1 (M)', 'M.NF1 (M)', 'P.PTPRD (M)', 'M.PTPRD (M)', 
        'P.PTPRT (M)', 'M.PTPRT (M)', 'P.ARID1A (M)', 'M.ARID1A (M)', 'P.BRAF (M)', 'M.BRAF (M)', 'P.PIK3CA (M)', 'M.PIK3CA (M)', 'P.EPHA3 (M)', 'M.EPHA3 (M)', 
        'P.FAT1 (M)', 'M.FAT1 (M)', 'P.SETD2 (M)', 'M.SETD2 (M)', 'P.RB1 (M)', 'M.RB1 (M)', 'P.MET (M)', 'M.MET (M)', 'P.KMT2C (M)', 'M.KMT2C (M)']

# Label each datapoint with a numeric value according to its sequencetype
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
cleaned = dat.loc[dat["type"].isin([0,1,2,3]), muts+["Seeding", "diag_order", "type"]]
# Remove completely empty datapoints
cleaned.drop(cleaned[cleaned.iloc[:,:-2].sum(axis=1)<1].index, inplace=True)
dat = jnp.array(cleaned.to_numpy(dtype=np.int8, na_value=-99))

events_plot = []
for elem in cleaned.columns[:-3].to_list()[::2]:
    full_mut_id = elem.split(".")
    events_plot.append(full_mut_id[1])
events_plot.append("Seeding")

if config['cv']:
    log_lams = np.linspace(np.log10(config['cv_start']), 
                           np.log10(config['cv_end']), 
                           config['cv_splits'])
    lams = 10**log_lams
    penal = utils.cross_val(dat=dat, 
                            penal_fun=reg_opt.symmetric_penal, 
                            splits=lams, 
                            n_folds=config['cv_folds'], 
                            m_p_corr=config['pm_ratio'], 
                            n_jobs=config['n_jobs'],
                            seed=42)
else:
    penal = config['lam']

if config["pm_ratio"] is None:
    perc_met = dat[:,-3].sum()/(dat.shape[0] - dat[:,-3].sum())
else:
    perc_met = config["pm_ratio"]

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