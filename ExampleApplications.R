
source("UtilityFunctions.R")
source("ModelConstruction.R")
source("Likelihood_Metastasis.R")
source("RegularizedOptimization.R")


#Simulation-------------------------
set.seed(1)

#Create a true MHN with random parameters (in log-space)
Theta.true <- Random.Theta(n=3, sparsity=0.50)
pTh <- Generate.pTh(Theta.true)

#Estimate the model from an empirical sample
pD  <- Finite.Sample(pTh, 1500)
Theta.hat <- Learn.MHN(pD, lambda=1/500)
pTh_hat <- Generate.pTh(Theta.hat)
KL.Div(pTh, pTh_hat)
#Given the true distribution, parameters can often be recovered exactly
Theta.rec <- Learn.MHN(pTh, lambda=0, reltol=1e-13)

#Cancer Progression Data----------------








