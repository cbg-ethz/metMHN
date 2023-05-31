import pandas as pd
import scipy.optimize as opt
import regularized_optimization as reg_opt
import Utilityfunctions as utils
import jax.numpy as jnp
import numpy as np
import jax as jax
jax.config.update("jax_enable_x64", True)

mut_handle = "~/MetaMHN/data/G12_PAADPANET_PM_z10_Events.csv"
annot_handle = "~/MetaMHN/data/sampleSelection.txt"
annot_data = pd.read_csv(annot_handle, sep="\t")
mut_data = pd.read_csv(mut_handle)
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], on=["patientID", "patientID"])

# Remove datapoints, that consist solely of NaNs
dat = dat.loc[dat.iloc[:,1:-3].isna().all(axis=1) == False, :]
dat = dat.loc[(dat.iloc[:,1:-3].sum(axis=1) > 0), :]
events = list(dat.columns[1:-4:2])
events.append("Seeding")

# split in train and test dataset
dat_train = dat.sample(frac=0.6, replace=False)
dat_test = dat.drop(dat_train.index)

train_prim_nomet, train_prim_met, train_met_only, train_coupled = utils.split_data(dat_train)
test_prim_nomet, test_prim_met, test_met_only, test_coupled = utils.split_data(dat_test)

dat_train = jnp.vstack((train_prim_nomet, train_prim_met, train_coupled, train_met_only))
n = 22
lam1_start = np.log(30/391)
lam2_start = np.log(30/391) # observed mean time to second diagnosis is 391 days

# Initialization of model params
indep = utils.indep(jnp.array(dat_train))
indep = indep.at[np.diag_indices(n)].add(lam2_start) # Assumption diagnosis and progression rates are on the same scale
start_params = np.append(indep, [lam1_start, lam2_start])
w_prim = 36/77
w_met = 41/77
#weights = np.array([w_prim, w_met/3, w_met/3, w_met/3])
weights = np.array([1, 1, 1, 1])
penal_vals = [0.001, 0.005, 0.08, 0.01, 0.05, 0.1]
scores = np.zeros(len(penal_vals))

for i in range(len(penal_vals)):
    x = opt.minimize(reg_opt.log_lik, x0 = start_params, args = (train_prim_met, train_prim_nomet, train_coupled, train_met_only, penal_vals[i], weights), 
                    method = "L-BFGS-B", jac = reg_opt.grad, options={"maxiter":100000, "disp":True, "ftol":1e-04})
    scores[i] = reg_opt.log_lik(x.x, test_prim_met, test_prim_nomet, test_coupled, test_met_only, 0.0)

# Best value for penal
i = scores.argmin()
print("Best lam:", penal_vals[i])
full_prim_nomet, full_prim_met, full_met_only, full_coupled = utils.split_data(dat)
dat = jnp.vstack((full_prim_nomet, full_prim_met, full_coupled, full_met_only))
indep = utils.indep(jnp.array(dat))
x = opt.minimize(reg_opt.log_lik, x0 = start_params, args = (full_prim_met, full_prim_nomet, full_coupled, full_met_only, penal_vals[i], weights), 
                method = "L-BFGS-B", jac = reg_opt.grad, options={"maxiter":100000, "disp":True, "ftol":1e-04})
df = pd.DataFrame(x.x[:-2].reshape((n,n)), columns=events, index=events)
lams = np.zeros(df.shape[0])
lams[0] = x.x[-2]
lams[1] = x.x[-1]
df["lams"] = lams
df.to_csv("~/MetaMHN/results/prad_"+ "data_ratio" + ".csv")