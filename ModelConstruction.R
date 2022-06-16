source("UtilityFunctions.R")
require("Matrix")

#Create a random MHN with (log-transformed) parameters Theta.
#Sparsity is given as percentage.
Random.Theta <- function(n, sparsity=0){
  Theta  <- matrix(0,nrow=n,ncol=n)
  
  diag(Theta)  <- NA
  nonZeros <- sample(which(!is.na(Theta)), size=(n^2 - n)*(1 - sparsity))
  
  Theta[nonZeros] <- rnorm(length(nonZeros))
  diag(Theta) <- rnorm(n)
  
  return(round(Theta,2))
} 

# ------------------------------------ Start Biuld.Q_Metastasis --------------------------
#Build the transition rate matrix Q from Theta
Build.Q_Metastasis <- function (Theta) {
  return(create_Q(Theta, T, F))
}

create_Q_sync <- function(Theta, diag = F, transp = F) {
  Theta <- exp(Theta)
  n <- nrow(Theta)
  
  # Construction of Q sync
  # ---------------------------------------------------------------------------
  Q_sync <- matrix(0, 2^(2 * (n-1)), 2^(2 *(n-1)))
  for(i in 1 : (n - 1)) {
    # \otimes_{j < i}
    j <- 1
    otimes_j_smaller_i <- 1
    while(j < i) {
      tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, Theta[i, j]), byrow = TRUE, nrow = 4, ncol = 4)
      otimes_j_smaller_i <-  tensor_core_at_j %x% otimes_j_smaller_i # please note that the kronnecker product in R is in "reverse" of MHN's paper
      j <- j + 1
      
    }
    
    # tensor core at the middle of the equation
    if(diag) {
      if(!transp) {
        tensor_core_at_i <- matrix(c(-Theta[i,i], 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     Theta[i, i], 0, 0, 0), byrow = TRUE, nrow = 4)
      } else {
        tensor_core_at_i <- matrix(c(-Theta[i,i], 0, 0, Theta[i, i],
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0), byrow = TRUE, nrow = 4)
      }
      
    } else {
      if(!transp) {
        tensor_core_at_i <- matrix(c(0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     Theta[i, i], 0, 0, 0), byrow = TRUE, nrow = 4)
      } else {
        tensor_core_at_i <- matrix(c(0, 0, 0, Theta[i, i],
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0), byrow = TRUE, nrow = 4)
      }
      
    }
    
    
    
    # \otimes_{j > i}
    j <- i + 1
    otimes_j_larger_i <- 1
    while(j < n) {
      tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, Theta[i, j]), byrow = TRUE, nrow = 4)
      otimes_j_larger_i <- tensor_core_at_j %x% otimes_j_larger_i
      j <- j + 1
    }
    
    Q_i <- otimes_j_larger_i %x% (tensor_core_at_i %x% otimes_j_smaller_i)
    
    Q_sync <- Q_i + Q_sync
  }
  Q_sync <- matrix(c(1,0,
                     0,0), byrow = T, ncol = 2) %x% Q_sync
  
  j <- 1
  otimes_j_smaller_i <- 1
  while(j < n) {
    tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, 0,
                                 0, 0, 0, Theta[n, j]), byrow = TRUE, nrow = 4, ncol = 4)
    otimes_j_smaller_i <- tensor_core_at_j %x% otimes_j_smaller_i
    j <- j + 1
  }
  if(diag) {
    if(!transp) {
      tensor_core_at_n <- matrix(c(-Theta[n, n], 0, Theta[n, n], 0), byrow = T, ncol = 2)
    } else {
      tensor_core_at_n <- matrix(c(-Theta[n, n], Theta[n, n], 0, 0), byrow = T, ncol = 2)
    }
  } else {
    if (!transp) {
      tensor_core_at_n <- matrix(c(0, 0, Theta[n,n], 0), byrow = T, ncol = 2)
    } else {
      tensor_core_at_n <- matrix(c(0, Theta[n, n], 0, 0), byrow = T, ncol = 2)
    }
  }
  
  Q_sync <- Q_sync + tensor_core_at_n %x% otimes_j_smaller_i
  
  return(Q_sync)
}

create_Q_async_primary <- function(Theta, diag = F, transp = F) {
  Theta <- exp(Theta)
  n <- nrow(Theta)
  Q_async_primary <- matrix(0, 2^(2 *(n-1)), 2^(2*(n-1)))
  for(i in 1 : (n-1)) {
    # \otimes_{j < i} gene j's impact on gene i
    j <- 1
    otimes_j_smaller_i <- 1
    while(j < i) {
      tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                   0, Theta[i, j], 0,0,
                                   0,0,1,0,
                                   0,0,0,Theta[i,j]), byrow = TRUE, nrow = 4)
      otimes_j_smaller_i <- tensor_core_at_j %x% otimes_j_smaller_i
      j <- j + 1
    }
    
    # gene i;s generator
    if(diag) {
      if (transp) {
        tensor_core_at_i <- matrix(c(-Theta[i, i], Theta[i, i], 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, -Theta[i, i], Theta[i, i],
                                     0, 0, 0, 0), byrow = TRUE, nrow = 4)
        
      } else {
        tensor_core_at_i <- matrix(c(-Theta[i, i], 0, 0, 0,
                                     Theta[i, i], 0, 0, 0,
                                     0, 0, -Theta[i, i], 0,
                                     0, 0, Theta[i, i], 0), byrow = TRUE, nrow = 4)
        
      }
    } else {
      if (transp) {
        tensor_core_at_i <- matrix(c(0, Theta[i, i], 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, Theta[i, i],
                                     0, 0, 0, 0), byrow = TRUE, nrow = 4)
      } else {
        tensor_core_at_i <- matrix(c(0, 0, 0, 0,
                                     Theta[i, i], 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, Theta[i, i], 0), byrow = TRUE, nrow = 4)
      }
    }
    
    
    # \otimes_{j > i} gene j's impact on gene i
    j <- i + 1
    otimes_j_larger_i <- 1
    while(j < n) {
      tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                   0, Theta[i, j], 0,0,
                                   0,0,1,0,
                                   0,0,0,Theta[i,j]), byrow = TRUE, nrow = 4)
      otimes_j_larger_i <- tensor_core_at_j %x% otimes_j_larger_i
      j <- j + 1
    }
    Q_async_primary_i <- otimes_j_larger_i %x% (tensor_core_at_i %x% otimes_j_smaller_i)
    Q_async_primary <- Q_async_primary + Q_async_primary_i
  }
  Q_async_primary <- matrix(c(0, 0, 0, 1), byrow = TRUE, nrow = 2) %x% Q_async_primary
  return(Q_async_primary)
}


create_Q_async_metastasis <- function(Theta, diag = F, transp = F) {
  Theta <- exp(Theta)
  n <- nrow(Theta)
  Q_async_metastasis <- matrix(0, 2^(2 *(n-1)), 2^(2*(n-1)))
  for(i in 1 : (n-1)) {
    # \otimes_{j < i} gene j's impact on gene i
    j <- 1
    otimes_j_smaller_i <- 1
    while(j < i) {
      tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, Theta[i, j], 0,
                                   0, 0, 0, Theta[i, j]), byrow = T, nrow = 4)
      otimes_j_smaller_i <- tensor_core_at_j %x% otimes_j_smaller_i
      j <- j + 1
    }
    
    # gene i generator 
    if(diag) {
      if(transp) {
        tensor_core_at_i <- matrix(c(-Theta[i, i] * Theta[i, n], 0, Theta[i, i] * Theta[i, n], 0, 
                                     0, -Theta[i, i] * Theta[i, n], 0, Theta[i, i] * Theta[i, n],
                                     0, 0, 0, 0,
                                     0, 0, 0, 0), byrow = TRUE, nrow = 4)
        
      } else {
        tensor_core_at_i <- matrix(c(-Theta[i, i] * Theta[i, n], 0, 0, 0, 
                                     0, -Theta[i, i] * Theta[i, n], 0, 0,
                                     Theta[i, i] * Theta[i, n], 0, 0, 0,
                                     0, Theta[i, i] * Theta[i, n], 0, 0), byrow = TRUE, nrow = 4)
      } 
    } else {
      if(transp) {
        tensor_core_at_i <- matrix(c(0, 0, Theta[i, i] * Theta[i, n], 0, 
                                     0, 0, 0, Theta[i, i] * Theta[i, n],
                                     0, 0, 0, 0,
                                     0, 0, 0, 0), byrow = TRUE, nrow = 4)
      } else {
        tensor_core_at_i <- matrix(c(0, 0, 0, 0, 
                                     0, 0, 0, 0,
                                     Theta[i, i] * Theta[i, n], 0, 0, 0,
                                     0, Theta[i, i] * Theta[i, n], 0, 0), byrow = TRUE, nrow = 4)
      }
    }
    
    
    # \otimes_{j > i} gene j's impact on gene i
    j <- i + 1
    otimes_j_larger_i <- 1
    while(j < n) {
      tensor_core_at_j <- matrix(c(1, 0, 0, 0,
                                   0, 1, 0, 0,
                                   0, 0, Theta[i, j], 0,
                                   0, 0, 0, Theta[i, j]), byrow = TRUE, nrow = 4)
      otimes_j_larger_i <- tensor_core_at_j %x% otimes_j_larger_i
      j <- j + 1
    }
    Q_async_metastasis_i <- otimes_j_larger_i %x% (tensor_core_at_i %x% otimes_j_smaller_i)
    Q_async_metastasis <- Q_async_metastasis + Q_async_metastasis_i
  }
  Q_async_metastasis <- matrix(c(0, 0, 0, 1), byrow = TRUE, nrow = 2) %x% Q_async_metastasis
  return(Q_async_metastasis)
}

# create Q based on Theta and two indicators'
# diag means we want to include the diagonal of the Q matrix
# transp means we want to transpose the Q matrix
# This function is used for testing Q%*%vec 
# Build.Q_Metastasis is just a special case of this function 
# when we set diag = T and transp = F
create_Q <- function(Theta, diag = F, transp = F) {
  Q <- create_Q_sync(Theta, diag, transp) + create_Q_async_metastasis(Theta, diag, transp) + create_Q_async_primary(Theta, diag, transp)
  return(Q)
}

# Code for testing
# n <- 5
# Theta <- log(matrix(c(1:25), byrow = T, ncol = n))
# Q <- create_Q(Theta, T, F)
# colnames(Q) <- genoNames(n)
# rownames(Q) <- genoNames(n)
# get_mutation_process(Q)
# --------------------------- End of Build.Q_Metastasis---------------------------------------

#Get the diagonal of Q. 
Q.Diag <- function(Theta){
  n <- ncol(Theta)
  dg <- rep(0, 2^n)
  
  for(i in 1:n){
    dg <- dg - Q.Subdiag(Theta, i)
  }
  
  return(dg)
}

# ----------------- Start Biuld Q.Diag.Metastasis --------------------------------------------

Q.Diag.Metastasis <- function(Theta) {
  result <- Q.Diag.Metastasis.syn(Theta) + Q.Diag.Metastasis.asyn.primary(Theta) + Q.Diag.Metastasis.asyn.metastasis(Theta)
  return(result)
}

Q.Diag.Metastasis.syn <- function(Theta){
  n <- ncol(Theta)
  y <- rep(0, 2^(2*n - 1))
  x <- matrix(rep(1, 2^(2*n - 1)), ncol = 1)
  
  for(i in 1:n){ #Should be parallelized
    y <- y + Kron.vec.Q.Diag.sync(Theta, i, x)
  }    
  
  return(y)
}

Kron.vec.Q.Diag.sync <- function(Theta, i, x) {
  Theta_i <- exp(Theta[i,]) #extract relevant Thetas and convert from log-space
  n <- length(Theta_i)
  
  if (i == n) {
    # Code that deals with the second part of the tensor structure
    for(j in 1 : (n-1)) {
      x = matrix(x, byrow = T, ncol = 4)
      x[, 2] <- 0
      x[, 3] <- 0
      x[, 4] <- Theta_i[j] * x[, 4]
    }
    
    x = matrix(x, byrow = T, ncol = 2)
    x[, 1] <- -Theta_i[n] * x[, 1]
    x[, 2] <- 0
    return(as.vector(x))
  }
  
  for(j in 1:n) {
    if (j == n) {
      x = matrix(x, byrow = T, ncol = 2)
    } else {
      x = matrix(x, byrow = T, ncol = 4)
    }
    
    
    if(i == j) {
      x[, 1] <- -Theta_i[i] * x[, 1]
      x[, 2] <- 0
      x[, 3] <- 0
      x[, 4] <- 0
    } else if(j == n) {
      x[, 2] <- 0
    } else {
      x[, 2] <- 0
      x[, 3] <- 0
      x[, 4] <- Theta_i[j] * x[, 4]
    }
  }
  
  return(as.vector(x))
}

Q.Diag.Metastasis.asyn.primary <- function(Theta){
  n <- ncol(Theta)
  y <- rep(0, 2^(2*n - 1))
  x <- matrix(rep(1, 2^(2*n - 1)), ncol = 1)
  
  for(i in 1:(n-1)){ #Should be parallelized
    y <- y + Kron.vec.Q.Diag.async.primary(Theta, i, x)
  }    
  
  return(y)
}

Kron.vec.Q.Diag.async.primary <- function(Theta, i, x) {
  Theta_i <- exp(Theta[i, ])
  n <- length(Theta_i)
  
  for(j in 1:n) {
    if(j == n) {
      x = matrix(x, byrow = T, ncol = 2)
    } else {
      x = matrix(x, byrow = T, ncol = 4)
    }
    
    if (j == i) {
      x[, 1] <- -Theta_i[i] * x[, 1]
      x[, 2] <- 0
      x[, 3] <- -Theta_i[i] * x[, 3]
      x[, 4] <- 0
    } else if(j == n) {
      x[, 1] <- 0
    } else {
      x[, 2] <- Theta_i[j] * x[, 2]
      x[, 4] <- Theta_i[j] * x[, 4]
    }
  }
  
  return(as.vector(x))
}

Q.Diag.Metastasis.asyn.metastasis <- function(Theta){
  n <- ncol(Theta)
  y <- rep(0, 2^(2*n - 1))
  x <- matrix(rep(1, 2^(2*n - 1)), ncol = 1)
  
  for(i in 1:(n-1)){ #Should be parallelized
    y <- y + Kron.vec.Q.Diag.async.metastasis(Theta, i, x)
  }    
  
  return(y)
}

Kron.vec.Q.Diag.async.metastasis <- function(Theta, i, x) {
  Theta_i <- exp(Theta[i, ])
  n <- length(Theta_i)
  
  for(j in 1:n) {
    if(j == n) {
      x = matrix(x, byrow = T, ncol = 2)
    } else {
      x = matrix(x, byrow = T, ncol = 4) 
    }
    
    
    if(j == i) {
      x[, 1] <- -Theta_i[i] * Theta_i[n] * x[, 1]
      x[, 2] <- -Theta_i[i] * Theta_i[n] * x[, 2]
      x[, 3] <- 0
      x[, 4] <- 0
    } else if (j == n) {
      x[, 1] <- 0
    } else {
      x[, 3] <- Theta_i[j] * x[, 3]
      x[, 4] <- Theta_i[j] * x[, 4]
    }
  }
  
  return(as.vector(x))
}



# Code for testing
# n <- 7
# Theta <- log(matrix(1:n^2, byrow = T, nrow = n))
# diag_from_Build_Q <- diag(Build.Q_Metastasis(Theta = Theta))
# diag_from_Q.Diag <- Q.Diag.Metastasis(Theta)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, diag_from_Build_Q, diag_from_Q.Diag)) == length(diag_from_Q.Diag)

#-------------------------------- The end of Q.Diag ------------------------------------------

#Learn an independence model from the data distribution, which assumes that no events interact. 
#Used to initialize the parameters of the actual model before optimization.
Learn.Indep <- function(pD){
  n <- log(length(pD), base=2)
  n <- (n + 1) / 2
  Theta <- matrix(0, nrow=n, ncol=n)
  
  for(i in 1:n){
    pD <- matrix(pD, nrow=2^(2*n-2), ncol=2, byrow=T)    
    
    perc <- sum(pD[,2])
    Theta[i,i] <- log(perc/(1-perc))
  }
  
  return(round(Theta,2))
}


