#Smooth approximation of the L1 penalty on Theta.
#(to be replaced with OWL-QN)
L1 <- function(Theta, eps=1e-05){
  diag(Theta) <- 0 
  sum(sqrt(Theta^2 + eps))
}

#Derivative of L1 penalty
L1_ <- function(Theta, eps=1e-05){
  diag(Theta) <- 0
  Theta / sqrt(Theta^2 + eps)
}


#Regularized Score 
Score.Reg <- function(Theta, pD, lambda){
  
  #Reshape parameters as matrix, after internal handling by BFGS as vectors.
  n <- sqrt(length(Theta))
  Theta <- matrix(Theta,nrow=n,ncol=n)  
  
  Score(Theta,pD) - lambda*L1(Theta)
} 

#Regularized Gradient
Grad.Reg  <- function(Theta, pD, lambda){
  n <- sqrt(length(Theta))
  Theta <- matrix(Theta,nrow=n,ncol=n)  
  
  Grad(Theta,pD) - lambda*L1_(Theta)
} 


#Learn an MHN from data.
Learn.MHN <- function(pD, init=NULL, lambda=0 ,maxit=5000, trace=0, reltol=1e-07, round=T){
  n <- log(length(pD), base=2)
  n <- (n + 1) / 2
  
  #Initialize the parameters from the independence model
  if(is.null(init)){
    init <- Learn.Indep(pD)
  } 
  
  opt <- optim(as.vector(init), fn=Score.Reg, gr=Grad.Reg, pD, lambda, method="BFGS",
               control=list(fnscale=-1,trace=trace,maxit=maxit,reltol=reltol))
  
  Theta <- matrix(opt$par,nrow=n,ncol=n)
  
  if(round){
    Theta <- round(Theta,2)
  } 
  
  return(Theta)
}


