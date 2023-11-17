import metmhn.regularized_optimization as reg_opt
import metmhn.Utilityfunctions as utils

import pandas as pd
import scipy.optimize as opt
import jax.numpy as jnp
import numpy as np
import jax as jax
jax.config.update("jax_enable_x64", True)
import logging
logging.basicConfig(filename='example.log', filemode='w', level=logging.INFO, force=True)

#mut_handle = "../data/luad/G13_LUAD_PM_v2_Events_20and15_Full.csv"
#annot_handle = "../data/luad/G13_LUAD_PM_v2_sampleSelection_20and15.csv"
annot_handle = "../data/paad/G13_PAADPANET_PM_v2_sampleSelection_30and15.csv"
mut_handle = "../data/paad/G13_PAADPANET_PM_v2_Events_30and15_Full.csv"

annot_data = pd.read_csv(annot_handle)
mut_data = pd.read_csv(mut_handle)
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], on=["patientID", "patientID"])

annot_data = pd.read_csv(annot_handle)
mut_data = pd.read_csv(mut_handle)
mut_data.rename(columns={"Unnamed: 0":"patientID"}, inplace = True)
dat = pd.merge(mut_data, annot_data.loc[:, ['patientID', 'metaStatus']], 
               on=["patientID", "patientID"])
muts = dat.columns[1:-4]
# Label each datapoint with a numeric value according to its sequencetype
dat["type"] = dat.apply(utils.categorize, axis=1)
dat["Seeding"] = dat.apply(utils.add_seeding, axis=1)
events_data = muts+["Seeding"]

# Only use datapoints where the state of the metastasis is known
cleaned = dat.loc[dat["type"].isin([0,1,2,3]), muts+["Seeding", "type"]]
cleaned.drop(cleaned[cleaned.iloc[:,:-1].sum(axis=1)<1].index, inplace=True)
dat_prim_nomet, dat_prim_met, dat_met_only, dat_coupled = utils.split_data(cleaned, events_data)

penal1 = 0.00129 #L1 penalty on off-diagonals
m_p_corr = 0.83
th_init, fd_init, sd_init = utils.indep(jnp.array(cleaned.to_numpy()), dat_coupled.shape[0])
theta, fd_effects, sd_effects = reg_opt.learn_mhn(th_init, fd_init, sd_init, dat_prim_nomet, dat_prim_met, 
                                                  dat_met_only, dat_coupled, m_p_corr, penal1)
th_plot = np.row_stack((fd_effects.reshape((1,-1)), 
                    sd_effects.reshape((1,-1)), 
                    theta))
events = []
for elem in cleaned.columns.to_list()[::2]:
    iwas = elem.split(".")
    if len(iwas) > 2:
        if iwas[1]  == 'Mut':
            events.append(iwas[2] + " (M)")
        else:
            events.append(iwas[1] + " (" + iwas[2] + ")")
    elif len(iwas) == 2:
        events.append(iwas[1])
    else:
        events.append("Seeding")
df2 = pd.DataFrame(th_plot, columns=events)
df2.to_csv("../results/paad/full_30mut_15cnv_00129.csv")