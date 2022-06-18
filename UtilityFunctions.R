#Convert a state from a bit-vector to a natural number.
State.to.Int <- function(x){
  x <- as.logical(rev(x))
  packBits(rev(c(rep(FALSE, 32 - length(x)%%32), x)), type="integer") + 1
}

#Convert a data matrix, where each row is the bit-vector of a state,
#to a probability distribution as a vector.
Data.to.pD <- function(Data){
  Data <- as.matrix(Data)
  n <- ncol(Data)
  N <- 2^n
  
  Data <- apply(Data, 1, State.to.Int)

  pD <- tabulate(Data, nbins=N)
  pD <- pD/sum(pD)
  
  return(pD)
}

#Simulate an empirical sample from a probability distribution.
Finite.Sample <- function(pTh, k){
  N <- length(pTh)
  tabulate(sample(1:N, k, prob=pTh, replace=T), nbins=N) / k
}

#Kullback-Leibler divergence from model distribution q to true distribution p
KL.Div <- function(p,q){
  log_p <- log(p)
  log_p[is.infinite(log_p)] <- 0
  log_q <- log(q)
  log_q[is.infinite(log_q)] <- 0
  as.numeric(p%*%log_p - p%*%log_q)
}

# Kronicker Delta Function
Kronicker.Delta <- function(x, y) {
  if (x == y) {
    return(1)
  } else {
    return(0)
  }
}

# Turns out this function is the same as Kronicker Delta
# also this function should be named Delta 
Sigma <- function(x, y) {
  if (x == y) return(1)
  else return(0)
}


# Generate the genotype names based on the order in the MHN paper
# Also adapted to suit the metastasis need.
genoNames <- function(n) {
  N <- 2 * n - 1
  tau <- matrix(1, N, N)
  geno <- matrix(0,2^N, N)
  for(i in 1:N){
    geno[,i] <- diag(factorF(tau[,i],i, add=FALSE))
  }
  x <- apply(geno, 1, paste, collapse="")
  x <- gsub('(.{2})', '\\1|', x)
  # x <- gsub('.{1}$','', x)
  return(x)
}

factorF <- function(tauI, i, add=T){
  n <- length(tauI)
  if(i==n){
    fTmp <- matrix(c(0,0,0,tauI[n]),2,2)
    oneTmp <- matrix(c(1,0,0,0),2,2)
  }
  else{
    fTmp <- matrix(c(1,0,0,tauI[n]), 2, 2)
    oneTmp <- matrix(c(1,0,0,1), 2, 2)
  }
  for (j in (n-1):1){
    if(i == j){
      fTmp <- kronecker(fTmp, matrix(c(0,0,0,tauI[j]),2,2))
      oneTmp <- kronecker(oneTmp, matrix(c(1,0,0,0),2,2))
    }
    else{
      fTmp <- kronecker(fTmp, matrix(c(1,0,0,tauI[j]),2,2))
      oneTmp <- kronecker(oneTmp, matrix(c(1,0,0,1), 2, 2))
    }
  }
  return(fTmp+add*oneTmp)
} 

get_mutation_process <- function(Q) {
  # sink("mutation_process_for_Q.txt")
  mutation_process <- c()
  n <- nrow(Q)
  for(i in 1 : n) {
    for(j in 1 : n) {
      if(Q[i, j] > 0) {
        from <- colnames(Q)[j]
        to <- rownames(Q)[i]
        # cat(from, "->", to, ":", Q[i, j], "\n")
        mutation_process <- c(mutation_process, paste(from, "->", to, ":", Q[i, j], "\n"))
      }
    }
  }
  # sink()
  # file.show("mutation_process_for_Q.txt")
  return(mutation_process)
}
