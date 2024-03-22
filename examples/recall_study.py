import argparse

parser = argparse.ArgumentParser(description="Apply metMHN to a specified dataset",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-n_dat", action="store", help="Number of datapoints to simulate", type=int, default=5000)
parser.add_argument("-n_reps", action="store", help="Number of models to learn", type = int, default=100)
parser.add_argument("-logs", action="store", default="./inf.dat", type=str, 
                    help="relative path to log-file destination")
parser.add_argument("-seed", action="store", help="Seed for random number generator", type = int, default=42)
args = parser.parse_args()
config = vars(args)

import jax.random as jrp
import jax.numpy as jnp
import numpy as np
import pandas as pd
import metmhn.simulations as simul
import metmhn.Utilityfunctions as utils
import metmhn.regularized_optimization as reg_opt

import logging

logging.basicConfig(filename=config['logs'],
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    filemode='w', 
                    level=logging.INFO, 
                    force=True,
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )

def create_dat(dat_sim_full, n_em, n_nm, n_c, n_pm, key):
    key, sub_key = jrp.split(key)
    dat_po = dat_sim_full[dat_sim_full[:,-2]==0,:]
    # Discard empty datapoints
    dat_po = dat_po[dat_po.sum(axis=1)!=0,:]
    po_inds = jrp.choice(sub_key, jnp.arange(dat_po.shape[0]), shape=[n_nm], replace=False)
    dat_po = jnp.hstack([dat_po[po_inds,:], jnp.array([0]*n_nm).reshape((n_nm,1))])
    dat_em = dat_sim_full[dat_sim_full[:,-2]!=0,:]
    em_inds = jrp.choice(sub_key, jnp.arange(dat_em.shape[0]), shape=[n_em], replace=False)
    dat_c = jnp.hstack([dat_em[em_inds[:n_c],:], jnp.array([3]*n_c).reshape((n_c,1))])
    dat_mo = jnp.hstack([dat_em[em_inds[n_c:n_c+n_mo],:], jnp.array([2]*n_mo).reshape((n_mo,1))])
    dat_pm = jnp.hstack([dat_em[em_inds[n_c+n_mo:],:], jnp.array([1]*n_pm).reshape((n_pm,1))])
    dat_full = jnp.vstack([dat_po, dat_c, dat_mo, dat_pm])
    return dat_full

# Learn the groundtruthmodel from real data

# Load raw data
mut_handle = "../data/luad/G14_LUAD_Events.csv"
annot_handle = "../data/luad/G14_LUAD_sampleSelection.csv"
annot_data = pd.read_csv(annot_handle)
mut_data = pd.read_csv(mut_handle)
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], 
               on=["patientID", "patientID"])

# Only select mutations for inference
muts = ['P.TP53 (M)', 'M.TP53 (M)', 'P.KRAS (M)', 'M.KRAS (M)', 'P.EGFR (M)', 'M.EGFR (M)', 
        'P.STK11 (M)', 'M.STK11 (M)', 'P.KEAP1 (M)', 'M.KEAP1 (M)', 'P.RBM10 (M)', 'M.RBM10 (M)', 
        'P.SMARCA4 (M)', 'M.SMARCA4 (M)', 'P.ATM (M)', 'M.ATM (M)', 'P.NF1 (M)', 'M.NF1 (M)', 
        'P.PTPRD (M)', 'M.PTPRD (M)', 'P.PTPRT (M)', 'M.PTPRT (M)', 'P.ARID1A (M)', 'M.ARID1A (M)', 
        'P.BRAF (M)', 'M.BRAF (M)', 'P.PIK3CA (M)', 'M.PIK3CA (M)', 'P.EPHA3 (M)', 'M.EPHA3 (M)', 
        'P.FAT1 (M)', 'M.FAT1 (M)', 'P.SETD2 (M)', 'M.SETD2 (M)', 'P.RB1 (M)', 'M.RB1 (M)', 
        'P.MET (M)', 'M.MET (M)', 'P.KMT2C (M)', 'M.KMT2C (M)']

# Label data according to their observational sequence and type
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

# Estimate the groundtruth model
m_p_corr = 0.65
th_init, dp_init, dm_init = utils.indep(dat)
theta_gt, d_p_gt, d_m_gt= reg_opt.learn_mhn(th_init=th_init, 
                                   dp_init=dp_init,
                                   dm_init=dm_init,
                                   dat=dat,
                                   perc_met=m_p_corr,
                                   penal=reg_opt.symmetric_penal,
                                   w_penal=0.005,
                                   opt_ftol=1e-05
                                   )
# Constants
n_events = theta_gt.shape[0]
n_reps = config["n_reps"]
learned_models =  np.zeros((n_reps, n_events+2, n_events))

# Subsample simulate data to mimic composition of real dataset
n_dat = config["n_dat"]
frac_nm = 0.115 # Fraction of never metastasisizng tumors
n_nm = int(np.round(frac_nm*n_dat)) 
n_em = n_dat - n_nm
n_c = int(np.round(0.107*n_em)) # Number of coupled datapoints
n_pm = int(np.round(0.386*n_em)) # Number of PT-only metassaized datapoints
n_mo = n_em - n_c - n_pm # Number of MT-only datapoints

# Sample datapoints according to the ground truth model
key = jrp.PRNGKey(config["seed"])
key, sub_key = jrp.split(key)
dat_sim_full = simul.simulate_dat_jax(jnp.array(theta_gt), 
                                      jnp.array(d_p_gt), 
                                      jnp.array(d_m_gt), 
                                      10000, 
                                      sub_key)

key, sub_key = jrp.split(key)
dat_full = create_dat(dat_sim_full, n_em, n_nm, n_c, n_pm, sub_key)

# Run a crossval once to determine the best lambda
log_lams = np.linspace(-3.5, -2.5, 5)
lams = 10**log_lams
m_p_corr = 0.65
penal_weight = utils.cross_val(dat=dat_full, 
                  penal_fun=reg_opt.symmetric_penal, 
                  splits=lams, 
                  n_folds=5, 
                  m_p_corr=0.65, 
                  seed=4209)

for i in range(n_reps):
    key, sub_key = jrp.split(key)
    dat = simul.simulate_dat_jax(jnp.array(theta_gt), 
                                 jnp.array(d_p_gt), 
                                 jnp.array(d_m_gt), 
                                 50000, 
                                 sub_key)
    key, sub_key = jrp.split(key)
    dat_sampled = create_dat(dat_sim_full, n_em, n_nm, n_c, n_pm, sub_key)
    th_init, dp_init, dm_init = utils.indep(dat_sampled)
    theta_inf_sim, d_p_inf_sim, d_m_inf_sim= reg_opt.learn_mhn(th_init=th_init, 
                                   dp_init=dp_init,
                                   dm_init=dm_init,
                                   dat=dat_sampled,
                                   perc_met=m_p_corr,
                                   penal=reg_opt.symmetric_penal,
                                   w_penal=penal_weight,
                                   opt_ftol=1e-05
                                   )
    learned_models[i,:,:] = np.row_stack((d_p_inf_sim.reshape((1,-1)), d_m_inf_sim.reshape((1,-1)), theta_inf_sim))
    logging.info(f"finished rep {i} out of {n_reps}")

df = pd.DataFrame(learned_models.reshape(-1, n_events))
df.to_csv('../results/recall_study_'+str(n_dat)+'.csv')