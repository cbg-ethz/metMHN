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
#mut_handle = "../data/paad/G12_PAADPANET_PM_z10_EventsAEP.csv"
#mut_handle = "../data/paad/G12_PAADPANET_PM_z10_Events_Mut50_Full.csv"
#mut_handle = "../data/paad/G12_PAADPANET_PM_z10_Events_30and5_Full.csv"
#annot_handle = "../data/paad/G12_PAADPANET_PM_z10_sampleSelection.csv"
#paads_only =  pd.read_csv("../data/paad/ptMap.csv")
#mut_handle = "../data/luad/G13_LUAD_PM_v2_Events_20and15_Full.csv"
#annot_handle = "../data/luad/G13_LUAD_PM_v2_sampleSelection_20and15.csv"
#annot_handle = "../data/luad/G13_LUAD_PM_z1_sampleSelection_20and8.csv"
#mut_handle = "../data/luad/G13_LUAD_PM_z1_Events_20and8_Full.csv"
annot_handle = "../data/paad/G13_PAADPANET_PM_v2_sampleSelection_30and15.csv"
mut_handle = "../data/paad/G13_PAADPANET_PM_v2_Events_30and15_Full.csv"

annot_data = pd.read_csv(annot_handle)
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
dat_prim_nomet, dat_met_only, dat_prim_met, dat_coupled = utils.split_data(cleaned)
print(dat_prim_nomet.shape[0], dat_prim_met.shape[0], dat_coupled.shape[0], dat_met_only.shape[0])
penal1 = 0.0007 #L1 penalty on off-diagonals
m_p_corr = 0.65
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
df2.to_csv("../results/luad/paired_20_8_001.csv")