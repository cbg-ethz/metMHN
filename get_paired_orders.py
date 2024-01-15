import pandas as pd
from metmhn.model import MetMHN
import os

# set paths
log_theta_path = R"results/prad/prad_g14_0001.csv"
data_path = R"data/prad/G14_PRAD_events.csv"
output_path = log_theta_path[:-4] + "orders.txt"

# get patient ids of already processed patients
if os.path.isfile(output_path):
    already_processed = pd.read_table(output_path, index_col=0).index.to_list()

# setup MetMHN
log_theta = pd.read_csv(
    log_theta_path, index_col=0)

obs1 = log_theta.iloc[0].to_numpy()
obs2 = log_theta.iloc[1].to_numpy()
log_theta = log_theta.drop(index=[0, 1]).to_numpy()

mmhn = MetMHN(log_theta=log_theta, obs1=obs1, obs2=obs2)

# get paired patients, drop NaNs, compute sequencing age difference
data = pd.read_csv(data_path, index_col=0)
paired = data[data["paired"] == 1]
paired = paired.dropna().astype(int)
paired["first_obs"] = paired["P.AgeAtSeqRep"] - paired["M.AgeAtSeqRep"]
paired["n_events"] = paired.loc[:, ~paired.columns.isin(
    ["first_obs", "P.AgeAtSeqRep", "M.AgeAtSeqRep"])].sum(axis=1)

paired = paired.loc[[i for i in paired.index if i not in already_processed]]

for i, (pat, row) in enumerate(paired.sort_values("n_events").iterrows()):
    print(f"{i:4}/{len(paired)}. {row['n_events']:2} events.", end="\r")
    with open(output_path, "a") as file:
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
