source("UtilityFunctions.R")
source("ModelConstruction.R")
source("Likelihood_Metastasis.R")
source("RegularizedOptimization.R")

registerDoFuture()
plan(sequential)

#Simulation-------------------------
set.seed(1)

#Create a true MHN with random parameters (in log-space)
Theta.true <- Random.Theta(n=8, sparsity=0.50)
pTh <- Generate.pTh(Theta.true)

#Estimate the model from an empirical sample
pD  <- Finite.Sample(pTh, 1500)
start.time <- Sys.time()
Theta.hat <- Learn.MHN(pD, lambda=1/500)
end.time <- Sys.time()
time.taken <- round(difftime(end.time, start.time, units = "secs"), digits = 2)
cat("Time taken for estimating Theta.hat is ", as.character(time.taken), " seconds\n")
pTh_hat <- Generate.pTh(Theta.hat)
KL.Div(pTh, pTh_hat)
#Given the true distribution, parameters can often be recovered exactly
start.time <- Sys.time()
Theta.rec <- Learn.MHN(pTh, lambda=0, reltol=1e-13)
end.time <- Sys.time()
time.taken <- round(difftime(end.time, start.time, units = "secs"), digits = 2)
cat("Time taken for estimating Theta.hat is ", as.character(time.taken), " seconds\n")


#Cancer Progression Data----------------




