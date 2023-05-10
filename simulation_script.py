from src import simulations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

log_theta = pd.read_csv("results/paad/paad_muts_30_5_008_old.csv", index_col=0)
lam1, lam2 = log_theta.pop("Sampling").iloc[:2]
events = log_theta.columns
log_theta = log_theta.to_numpy()

data = np.vstack([simulations.single_traject_timed(log_theta=log_theta, t_obs=np.inf, rng=np.random.default_rng()) for _ in range(1000000)])

np.save("simulations.npy", data)