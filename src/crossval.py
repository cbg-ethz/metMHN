import logging
logging.basicConfig(filename='crossval.log', level=logging.INFO)
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
import scipy.optimize as opt
import regularized_optimization as reg_opt
import Utilityfunctions as utils
import jax.numpy as jnp
import numpy as np
import jax as jax
import matplotlib.pyplot as plt
import ssr_likelihood_jax as lik
jax.config.update("jax_enable_x64", True)

#mut_handle = "../data/paad/G12_PAADPANET_PM_z10_EventsAEP.csv"
#mut_handle = "../data/paad/G12_PAADPANET_PM_z10_Events_Mut50_Full.csv"
mut_handle = "../data/paad/G12_PAADPANET_PM_z10_Events_30and5_Full.csv"
annot_handle = "../data/paad/sampleSelection.txt"
#mut_handle = "../data/luad/G13_LUAD_PM_z1_Events_20and8_Full.csv"
#annot_handle = "../data/luad/G13_LUAD_PM_z1_sampleSelection_20and8.csv"
annot_data = pd.read_csv(annot_handle, sep="\t")
mut_data = pd.read_csv(mut_handle)
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], on=["patientID", "patientID"])

# Remove datapoints, that consist solely of NaNs
dat = dat.loc[dat.iloc[:,1:-3].isna().all(axis=1) == False, :]
dat = dat.loc[(dat.iloc[:,1:-3].sum(axis=1) > 0), :]

start = 'P.Mut.KRAS'
stop = 'M.Mut.PTPRT'
mult = dat.set_index(["paired", "metaStatus"])
cleaned = mult.loc[zip(*[[0,0,0,1],["present", "absent", "isMetastasis", "isPaired"]]), start:stop]
cleaned = cleaned.sort_index()
cleaned.loc[(0, ["present", "isMetastasis"]), "Seeding"] = 1
cleaned.loc[(0, "absent"), "Seeding"] = 0
cleaned.loc[(1, "isPaired"), "Seeding"] = 1
events = cleaned.columns.to_list()[::2]
dat_prim_nomet, dat_met_only, dat_prim_met, dat_coupled = utils.split_data(cleaned)

n = (cleaned.shape[1]-1)//2
n += 1
lam1_start = np.log(30/162)
lam2_start = np.log(30/162) # observed mean time to second diagnosis is 87/162 days
m_p_corr = 0.8
indep = utils.indep(jnp.vstack((dat_met_only, dat_prim_met, dat_prim_nomet)), dat_coupled)
start_params = np.append(indep, [lam1_start, lam2_start])

logging.info("Started")

diag_penal, def_penal = utils.cross_val(cleaned, np.linspace(1e-04, 1e-02, 5), 5, start_params, m_p_corr, n)
x = opt.minimize(reg_opt.value_grad, x0 = start_params,#
                 args = (dat_prim_nomet, dat_coupled, dat_prim_met, dat_met_only, n-1, diag_penal, diag_penal,  m_p_corr),#
                 method = "L-BFGS-B", jac = True, options={"maxiter":10000, "disp":True, "ftol":1e-04})
print(1/jnp.exp(x.x[-2]))
df2 = pd.DataFrame(x.x[:-2].reshape((n, n)), columns=events, index=events)
theta = df2.copy()
df2["Sampling"] = np.append(np.array(x.x[-2:]), np.zeros(n-2))
df2.to_csv("../results/paad/paad_mixed_08_crossval.csv")