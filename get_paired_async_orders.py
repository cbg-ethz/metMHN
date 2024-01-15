import pandas as pd
import numpy as np
from metmhn.model import MetMHN


log_theta_path = R"results/prad/prad_g14_0001.csv"
data_path = R"data/prad/G14_PRAD_events.csv"

log_theta = pd.read_csv(
    log_theta_path, index_col=0)

obs1 = log_theta.iloc[0].to_numpy()
obs2 = log_theta.iloc[1].to_numpy()
log_theta = log_theta.drop(index=[0, 1]).to_numpy()

mmhn = MetMHN(log_theta=log_theta, obs1=obs1, obs2=obs2)

data = pd.read_csv(data_path, index_col=0)
paired = data[data["paired"] == 1]
paired = paired.dropna().astype(int)
paired["first_obs"] = paired["P.AgeAtSeqRep"] - paired["M.AgeAtSeqRep"]
paired["n_events"] = paired.loc[:, ~paired.columns.isin(
    ["first_obs", "P.AgeAtSeqRep", "M.AgeAtSeqRep"])].sum(axis=1)

for i, (pat, row) in enumerate(paired.sort_values("n_events").iterrows()):
    print(f"{i:4}/{len(paired)}. {row['n_events']:2} events.", end="\r")
    with open(log_theta_path[:-4] + "orders.txt", "a") as file:
        if row["first_obs"] == 0:
            first_obs = "sync"
        elif row["first_obs"] > 0:
            first_obs = "PT"
        else:
            first_obs = "Met"
        file.write(pat + "\t"
                   + str(mmhn.likeliest_order(
                       state=row[paired.columns[:-4]].to_numpy(),
                       first_obs=first_obs
                   )[0]) + "\n")
