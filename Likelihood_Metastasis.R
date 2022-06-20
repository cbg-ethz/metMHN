# File Name: Likelihood_Metastasis
# Created By: Chenxi Nie (聂晨晞)
# Date: 23-05-2022
# Function: This is a file that mimic Likelihood.R But in Meta_MHN
source("UtilityFunctions.R")
source("ModelConstruction.R")
require(doParallel)


# ------------------------ Q.Sync ----------------------------------------------
Q.sync.vec <- function(Theta, x, diag=F, transp=F){
  n <- ncol(Theta)
  y <- rep(0, 2^(2*n-1))
  
  for(i in 1:n){ #Should be parallelized
    y <- y + Kron.vec.Q.sync(Theta, i, x, diag, transp)
  }    
  
  return(y)
}

# Kron.vec for Q.sync
Kron.vec.Q.sync <- function(Theta, i, x, diag=F, transp=F) {
  Theta_i <- exp(Theta[i,]) #extract relevant Thetas and convert from log-space
  n <- length(Theta_i)
  
  if (i == n) {
    # Code that deals with i = n
    for(j in 1 : (n - 1)) {
      x = matrix(x, byrow = T, ncol = 4)
      x[, 2] <- 0
      x[, 3] <- 0
      x[, 4] <- Theta_i[j] * x[, 4]
    }
    
    x = matrix(x, byrow = T, ncol = 2)
    if(!transp) {
      if(!diag) {
        # transp = F, diag = F
        x[, 2] <- Theta_i[n] * x[, 1]
        x[, 1] <- 0
      } else {
        # transp = F, diag = T
        x[, 1] <- -Theta_i[n] * x[, 1]
        x[, 2] <- -x[, 1]
      }
    } else {
      if(!diag) {
        #transp = T, diag = F
        x[, 1] <- Theta_i[n] * x[, 2]
        x[, 2] <- 0
      } else {
        #transp = T, diag = T
        x[, 1] <- Theta_i[n] * (x[, 2] - x[, 1])
        x[, 2] <- 0
      }
    }
    
    return(as.vector(x))
  }
  
  # Code for i < n
  for(j in 1 : n) {
    if(j == n) {
      x = matrix(x, byrow = T, ncol = 2)
    } else {
      x = matrix(x, byrow = T, ncol = 4)
    }
    
    if(i == j) {
      if(!transp) {
        if(!diag) {
          # diag = F, transp = F
          x[, 4] <- Theta_i[i] * x[, 1]
          x[, 1] <- 0 * x[, 1]
          x[, 2] <- 0 * x[, 1]
          x[, 3] <- 0 * x[, 1]
        } else {
          # diag = T, transp = F
          x[, 1] <- -Theta_i[i] * x[, 1]
          x[, 2] <- 0 * x[, 1]
          x[, 3] <- 0 * x[, 1]
          x[, 4] <- -x[, 1]
        }
        
      } else {
        if(!diag) {
          # diag = F, transp = T
          x[, 1] <- Theta_i[i] * x[, 4]
          x[, 2] <- 0
          x[, 3] <- 0
          x[, 4] <- 0
        } else {
          # diag = T, transp = T
          x[, 1] <- -Theta_i[i]* x[, 1] + Theta_i[i] * x[, 4]
          x[, 2] <- 0 * x[,2]
          x[, 3] <- 0
          x[, 4] <- 0
        }
      }
    }
    else if(j == n) {
      x[, 2] <- 0
    } else{
      x[, 2] <- x[, 2] * 0
      x[, 3] <- x[, 3] * 0
      x[, 4] <- x[, 4] * Theta_i[j] 
    }
  }
  return(as.vector(x))
}

# Code for testing
# n <- 3
# Theta <- log(matrix(runif(n^2, 1, 99), byrow = T, ncol = n))
# x <- matrix(rnorm(2^(2*n-1)), ncol = 1)
# 
# Q_T_T_x <- create_Q_sync(Theta, T, T) %*% x
# Q_T_F_x <- create_Q_sync(Theta, T, F) %*% x
# Q_F_T_x <- create_Q_sync(Theta, F, T) %*% x
# Q_F_F_x <- create_Q_sync(Theta, F, F) %*% x
# 
# Q_T_T_x_2 <- Q.sync.vec(Theta, x, T, T)
# Q_T_F_x_2 <- Q.sync.vec(Theta, x, T, F)
# Q_F_T_x_2 <- Q.sync.vec(Theta, x, F, T)
# Q_F_F_x_2 <- Q.sync.vec(Theta, x, F, F)
# 
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_T_T_x, Q_T_T_x_2)) == length(Q_T_T_x_2)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_T_F_x, Q_T_F_x_2)) == length(Q_T_F_x_2)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_F_T_x, Q_F_T_x_2)) == length(Q_F_T_x_2)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_F_F_x, Q_F_F_x_2)) == length(Q_F_F_x_2)

#------------------------ Q.async.primary --------------------------------------
Q.async.primary.vec <- function(Theta, x, diag=F, transp=F){
  n <- ncol(Theta)
  y <- rep(0, 2^(2*n-1))
  
  for(i in 1:(n-1)){ #Should be parallelized
    y <- y + Kron.vec.Q.async.primary(Theta, i, x, diag, transp)
  }    
  
  return(y)
}


Kron.vec.Q.async.primary <- function(Theta, i, x, diag = F, transp = F) {
  Theta_i <- exp(Theta[i,]) #extract relevant Thetas and convert from log-space
  n <- length(Theta_i)
  
  for(j in 1:n) {
    if (j == n) {
      x <- matrix(x, byrow = T, ncol = 2)
    } else {
      x <- matrix(x, byrow = T, ncol = 4)
    }
   
    if (i == j) {
      if(!diag) {
        if(!transp) {
          # diag = F, transp = F
          x[, 2] <- Theta_i[i] * x[, 1]
          x[, 1] <- 0
          x[, 4] <- Theta_i[i] * x[, 3]
          x[, 3] <- 0
        } else {
          # diag = F, transp = T
          x[, 1] <- Theta_i[i] * x[, 2]
          x[, 2] <- 0
          x[, 3] <- Theta_i[i] * x[, 4]
          x[, 4] <- 0
          
        }
      } else {
        if(!transp) {
          # diag = T, transp = F
          x[, 1] <- -Theta_i[i] * x[, 1]
          x[, 2] <- -x[, 1]
          x[, 3] <- -Theta_i[i] * x[, 3]
          x[, 4] <- -x[, 3]
        } else {
          # diaf = T, transp = T
          x[, 1] <- -Theta_i[i] * x[, 1] + Theta_i[i] * x[, 2]
          x[, 2] <- 0
          x[, 3] <- -Theta_i[i] * x[, 3] + Theta_i[i] * x[, 4]
          x[, 4] <- 0
        }
      }
    }
    else if (j == n) {
      x[, 1] <- 0
    } else {
      x[, 2] <- Theta_i[j] * x[, 2]
      x[, 4] <- Theta_i[j] * x[, 4]
    }
  }
  return(as.vector(x))
}

# Code for testing
# n <- 7
# Theta <- log(matrix(runif(n^2, 1, 99), byrow = T, ncol = n))
# x <- matrix(rnorm(2^(2 * n - 1)), ncol = 1)
# Q_async_primary_T_T_x <- create_Q_async_primary(Theta, T, T)%*%x
# Q_async_primary_T_F_x <- create_Q_async_primary(Theta, T, F)%*%x
# Q_async_primary_F_T_x <- create_Q_async_primary(Theta, F, T)%*%x
# Q_async_primary_F_F_x <- create_Q_async_primary(Theta, F, F)%*%x
# Q_async_primary_vec_T_T <- Q.async.primary.vec(Theta, x, T,T)
# Q_async_primary_vec_T_F <- Q.async.primary.vec(Theta, x, T,F)
# Q_async_primary_vec_F_T <- Q.async.primary.vec(Theta, x, F,T)
# Q_async_primary_vec_F_F <- Q.async.primary.vec(Theta, x, F,F)
# 
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_primary_T_T_x, Q_async_primary_vec_T_T)) == length(Q_async_primary_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_primary_T_F_x, Q_async_primary_vec_T_F)) == length(Q_async_primary_vec_T_F)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_primary_F_T_x, Q_async_primary_vec_F_T)) == length(Q_async_primary_vec_F_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_primary_F_F_x, Q_async_primary_vec_F_F)) == length(Q_async_primary_vec_T_F)

# ---------------------- Q.async.metastasis ------------------------------------
Q.async.metastasis.vec <- function(Theta, x, diag=F, transp = F) {
  n <- ncol(Theta)
  y <- rep(0, 2^(2*n-1))
  
  for(i in 1:(n-1)){ #Should be parallelized
    y <- y + Kron.vec.Q.async.metastasis(Theta, i, x, diag, transp)
  }    
  
  return(y)
}

Kron.vec.Q.async.metastasis <- function(Theta, i, x, diag=F, transp=F) {
  Theta_i <- exp(Theta[i,]) #extract relevant Thetas and convert from log-space
  n <- length(Theta_i)
  
  for(j in 1 : n){
    if(j == n) {
      x <- matrix(x, byrow = T, ncol = 2)
    } else {
      x <- matrix(x, byrow = T, ncol = 4)
    }
    
    if (i == j) {
      if(!diag) {
        if (!transp) {
          # diag = F, transp = F
          x[, 3] <- Theta_i[i] * Theta_i[n] * x[, 1]
          x[, 1] <- 0
          x[, 4] <- Theta_i[i] * Theta_i[n] * x[, 2]
          x[, 2] <- 0
        } else {
          # diag = F, transp = T
          x[, 1] <- Theta_i[i] * Theta_i[n] * x[, 3]
          x[, 2] <- Theta_i[i] * Theta_i[n] * x[, 4]
          x[, 3] <- 0
          x[, 4] <- 0
        }
      } else {
        if (!transp) {
          # diag = T, transp = F
          x[, 1] <- -Theta_i[i] * Theta_i[n] * x[, 1]
          x[, 2] <- -Theta_i[i] * Theta_i[n] * x[, 2]
          x[, 3] <- -x[, 1]
          x[, 4] <- -x[, 2]
        } else {
          # diag = T, transp = T
          x[, 1] <- -Theta_i[i] * Theta_i[n] * x[, 1] + Theta_i[i] * Theta_i[n] * x[, 3]
          x[, 2] <- -Theta_i[i] * Theta_i[n] * x[, 2] + Theta_i[i] * Theta_i[n] * x[, 4]
          x[, 3] <- 0
          x[, 4] <- 0
        }
      }
    } else if (j == n) {
      x[, 1] <- 0
    } else {
      x[, 3] <- Theta_i[j] * x[, 3]
      x[, 4] <- Theta_i[j] * x[, 4]
    }
  }
  return(as.vector(x))
}

# code used for testing
# n <- 7
# Theta <- log(matrix(runif(n^2, 1, 99), byrow = T, ncol = n))
# x <- matrix(rnorm(2^(2 * n - 1)), ncol = 1)
# Q_async_metastasis_T_T_x <- create_Q_async_metastasis(Theta, T, T)%*%x
# Q_async_metastasis_T_F_x <- create_Q_async_metastasis(Theta, T, F)%*%x
# Q_async_metastasis_F_T_x <- create_Q_async_metastasis(Theta, F, T)%*%x
# Q_async_metastasis_F_F_x <- create_Q_async_metastasis(Theta, F, F)%*%x
# Q_async_metastasis_vec_T_T <- Q.async.metastasis.vec(Theta, x, T,T)
# Q_async_metastasis_vec_T_F <- Q.async.metastasis.vec(Theta, x, T,F)
# Q_async_metastasis_vec_F_T <- Q.async.metastasis.vec(Theta, x, F,T)
# Q_async_metastasis_vec_F_F <- Q.async.metastasis.vec(Theta, x, F,F)
# 
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_metastasis_T_T_x, Q_async_metastasis_vec_T_T)) == length(Q_async_metastasis_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_metastasis_T_F_x, Q_async_metastasis_vec_T_F)) == length(Q_async_metastasis_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_metastasis_F_T_x, Q_async_metastasis_vec_F_T)) == length(Q_async_metastasis_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_async_metastasis_F_F_x, Q_async_metastasis_vec_F_F)) == length(Q_async_metastasis_vec_T_T)

# ------------------------------ Put it all together ---------------------------------
Q.vec <- function(Theta, x, diag=F, transp=F){
  y_sync <- Q.sync.vec(Theta, x, diag, transp)
  y_async_p <- Q.async.primary.vec(Theta, x, diag, transp)
  y_async_m <- Q.async.metastasis.vec(Theta, x, diag, transp)
  y <- y_sync + y_async_p + y_async_m
  return(y)
}

# Code for testing
# n <- 7
# Theta <- log(matrix(runif(n^2, 1, 99), byrow = T, ncol = n))
# x <- matrix(rnorm(2^(2 * n-1)), ncol = 1)
# Q_T_T_x <- create_Q(Theta, T, T) %*% x
# Q_T_F_x <- create_Q(Theta, T, F) %*% x
# Q_F_T_x <- create_Q(Theta, F, T) %*% x
# Q_F_F_x <- create_Q(Theta, F, F) %*% x
# Q_vec_T_T <- Q.vec(Theta, x, T, T)
# Q_vec_T_F <- Q.vec(Theta, x, T, F)
# Q_vec_F_T <- Q.vec(Theta, x, F, T)
# Q_vec_F_F <- Q.vec(Theta, x, F, F)
# 
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_T_T_x, Q_vec_T_T)) == length(Q_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_T_F_x, Q_vec_T_F)) == length(Q_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_F_T_x, Q_vec_F_T)) == length(Q_vec_T_T)
# sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, Q_F_F_x, Q_vec_F_F)) == length(Q_vec_T_T)

# BenchMarking for Q.vec function
# ns <- 2:14
# time <- c()
# 
# for (n in ns) {
#   print(n)
#   Theta <- log(matrix(runif(n^2, 1, 99), byrow = T, ncol = n))
#   x <- matrix(rnorm(2^(2 * n)), ncol = 1)
#   start <- Sys.time()
#   Q_vec_F_T <- Q.vec(Theta, x, F, T)
#   end <- Sys.time()
#   time <- c(time, difftime(end, start, units = "secs")[[1]])
# }
# 
# plot(ns, time, type = "b", pch = 19, xlab = "n", ylab = "time in seconds")

# ----------------- other functions in Likelihood.R adapted to meta_MHN---------------------------

#Solves [-Q+I]x = b using the Jacobi method. 
#Convergence is guaranteed for 2*n iterations
Jacobi <- function(Theta, b, transp=F, x=NULL){
  n <- ncol(Theta)
  if(is.null(x)) x <- rep(1,2^(2 * n - 1))/(2^(2 * n - 1))
  
  dg <- -Q.Diag.Metastasis(Theta) + 1
  
  criterion <- 2 * n
  for(i in 1:(criterion)){
    x <- b + Q.vec(Theta, x, diag=F, transp)
    x <- x/dg
  }
  
  return(x)
}

# test the new Jacobi function
# n <- 3
# Theta <- log(matrix(1:n^2, byrow = T, ncol = n))
# p0 <- c(1, rep(0, 2^(2*n - 1) - 1))
# x <- Jacobi(Theta, p0, F)
# Q <- Build.Q_Metastasis(Theta)
# I <- diag(2^{2 * n - 1})
# p02 <- (-Q + I) %*% x


#Generate the probability distribution from a model Theta.
Generate.pTh <- function(Theta, p0 = NULL){
  n <- ncol(Theta)
  if(is.null(p0)) p0 <- c(1, rep(0, 2^(2 * n - 1) - 1))
  
  return(Jacobi(Theta,p0))
}

#Direct copy from Likelihood.R
#Log-likelihood Score
Score <- function(Theta, pD){
  pTh <- Generate.pTh(Theta)
  pTh[pTh == 0] <- 1e-20
  # Just a test
  log_pTh <- log(pTh)
  log_pTh[is.infinite(log_pTh)] <- 0
  as.numeric(pD %*% log(pTh)) 
  # as.numeric(pD%*%log_pTh)
}

# -------------------------- Gradient Related Code ----------------------------------------------------------------------------
#Gradient of the Score wrt Theta. Implements equation (7)
#Adapted to Meta_MHN
Grad <- function(Theta, pD){
  n <- sqrt(length(Theta))
  Theta <- matrix(Theta,nrow=n,ncol=n)
  
  p0    <- c(1, rep(0,2^(2 * n - 1) - 1))
  
  pTh <- Jacobi(Theta, p0)
  pD_pTh <- pD/pTh
  pD_pTh[is.na(pD_pTh)] = 0
  q   <- Jacobi(Theta, pD_pTh, transp=T)
  
  # start.time <- Sys.time()
  # G <- matrix(0,nrow=n,ncol=n)
  # 
  # # Single thread G calculation
  # # Maybe think about a new algorithm
  # for(i in 1:n){
  #   for(j in 1:n) {
  #     G[i, j] <- t(q) %*% dQ.vec(Theta, pTh,i , j)
  #   }
  # }
  # end.time<-Sys.time()
  # time.taken <- round(difftime(end.time, start.time, units = "secs"), digits = 2)

  
  # Nested foreach loop
  # Replace the double for loop with foreach
  # Special thanks to this post
  # https://cran.r-project.org/web/packages/foreach/vignettes/nested.html
  # start.time <- Sys.time()
  # G <-
  #   foreach(i = 1:n, .combine="rbind",.export = c("dQ.vec", "dQ.sync.vec", "dQ.async.primary.vec","dQ.async.metastasis.vec", "Kronicker.Delta")) %:%
  #     foreach(j = 1:n, .combine = "c", .export = c("dQ.vec", "dQ.sync.vec", "dQ.async.primary.vec","dQ.async.metastasis.vec", "Kronicker.Delta")) %dopar% {
  #       t(q) %*% dQ.vec(Theta, pTh,i , j)
  #     }
  # end.time <- Sys.time()
  # time.taken2 <- round(difftime(end.time, start.time, units = "secs"), digits = 2)

  is <-rep(0, n^2); js <- rep(0, n^2)
  for(i_prime in 1:n^2) {
    if (i_prime %% n == 0) {
      i = i_prime %/% n
    } else {
      i = i_prime %/% n + 1
    }
    j = i_prime - (i-1) * n
    is[i_prime] <- i
    js[i_prime] <- j
  }

  # flatted foreach loop
  G <- foreach(i_prime = 1:n^2, .combine = "c") %dopar% {
    i <- is[i_prime]
    j <- js[i_prime]
    t(q) %*% dQ.vec(Theta, pTh, i, j)
  }
  G <- matrix(G, byrow = T, ncol = n, nrow = n)
  return(G)
}

# dQ/dTheta[i, j] %*% vec written as sum of kronecker products
# This function is a sum of Q_sync, Q_async_pri and Q_async_meta matrixes
dQ.vec<- function(Theta, p, i, j){
  result <- dQ.sync.vec(Theta, p, i, j) + dQ.async.primary.vec(Theta, p, i, j) + dQ.async.metastasis.vec(Theta, p, i, j)
  return(result)
}

# Code for testing
# n <- 7
# Theta <- log(matrix(runif(n^2, 1, 99), byrow = T, ncol = n))
# p <- matrix(rnorm(2^(2 * n-1)), ncol = 1)
# for(i in 1:n) {
#   for(j in 1:n) {
#     dQ_x <- create_dQ_Theta_ij(Theta, i, j) %*% p
#     dQ_vec <- dQ.vec(Theta, p, i, j)
#     print(sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, dQ_x, dQ_vec)) == length(dQ_vec))
#   }
# }


# dQ_sync/ dTheta[i, j] %*% vec written as sum of kronecker products
dQ.sync.vec<- function(Theta, p, i, j) {
  Theta_i <- exp(Theta[i, ])
  n <- length(Theta_i)
  
  if(i < n) {
    for(k in 1 : n) {
      if(k == n) {
        p = matrix(p, byrow = T, ncol = 2)
      } else{
        p = matrix(p, byrow = T, ncol = 4)
      }
      
      if(k == i) {
        p[, 1] <- -Theta_i[i] * (1 - Kronicker.Delta(j, n)) * p[, 1]
        p[, 2] <- 0
        p[, 3] <- 0
        p[, 4] <- -p[, 1]
      } else if(k == n) {
        p[, 2] <- 0
      } else {
        p[, 1] <- (1 - Kronicker.Delta(k, j)) * p[, 1]
        p[, 2] <- 0
        p[, 3] <- 0
        p[, 4] <- Theta_i[k] * p[, 4]
      }
    }
  } else {
    for(k in 1:n) {
      if(k == n) {
        p = matrix(p, byrow = T, ncol = 2)
      } else {
        p = matrix(p, byrow = T, ncol = 4)
      }
      
      if(k < n) {
        p[, 1] <- (1 - Kronicker.Delta(k, j)) * p[, 1]
        p[, 2] <- 0
        p[, 3] <- 0
        p[, 4] <- Theta_i[k] * p[, 4]
      } else {
        p[, 1] <- -Theta_i[n] * p[, 1]
        p[, 2] <- -p[, 1]
      }
    }
  }
  return(as.vector(p))
}

# Code for testing
# n <- 7
# Theta <- log(matrix(1:n^2, byrow = T, ncol = n))
# p <- matrix(1:2^(2*n-1),ncol = 1)
# for(i in 1:n) {
#   for(j in 1:n) {
#     dQ_x <- create_dQ_sync_ij(Theta, i, j) %*% p
#     dQ_vec <- dQ.sync.vec(Theta, p, i, j)
#     print(sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, dQ_x, dQ_vec)) == length(dQ_vec))
#   }
# }


# dQ_async_primary / dTheta[i, j] %*% vec written as sum of kronecker products
dQ.async.primary.vec <- function(Theta, p, i, j) {
  Theta_i = exp(Theta[i, ])
  n <- length(Theta_i)
  
  if(i < n) {
    for(k in 1:n) {
      if(k == n) {
        p = matrix(p, byrow = T, ncol = 2)
      } else {
        p = matrix(p, byrow = T, ncol = 4)
      }

      if(i == k) {
        p[, 1] <- -Theta_i[i] * (1 - Kronicker.Delta(j, n)) * p[, 1]
        p[, 2] <- -p[, 1]
        p[, 3] <- -Theta_i[i] * (1 - Kronicker.Delta(j, n)) * p[, 3]
        p[, 4] <- -p[, 3]
      } else if(k == n) {
        p[, 1] <- 0
      } else {
        p[, 1] <- p[, 1] * (1 - Kronicker.Delta(k, j))
        p[, 2] <- p[, 2] * Theta_i[k]
        p[, 3] <- p[, 3] * (1 - Kronicker.Delta(k, j))
        p[, 4] <- p[, 4] * Theta_i[k]
      }
    }
  } else {
    p <- rep(0, 2^{2 * n - 1})
  }

  return(as.vector(p))
}

# Code for testing
# n <- 7
# Theta <- log(matrix(1:n^2, byrow = T, ncol = n))
# p <- matrix(1:2^(2*n-1),ncol = 1)
# for(i in 1:n) {
#   for(j in 1:n) {
#     dQ_x <- create_dQ_async_primary_ij(Theta, i, j) %*% p
#     dQ_vec <- dQ.async.primary.vec(Theta, p, i, j)
#     print(sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, dQ_x, dQ_vec)) == length(dQ_vec))
#   }
# }


# dQ_async_metastasis / dTheta[i, j] %*% vec wriiten as sum of kronecker products
dQ.async.metastasis.vec <- function(Theta, p, i, j) {
  Theta_i = exp(Theta[i, ])
  n <- length(Theta_i) 
  
  if(i < n) {
    for(k in 1 : n) {
      if(k == n) {
        p = matrix(p, byrow = T, ncol = 2)
      } else {
        p = matrix(p, byrow = T, ncol = 4)
      }
      
      if(i == k) {
        p[, 1] <- -Theta_i[i] * Theta_i[n] * p[, 1]
        p[, 2] <- -Theta_i[i] * Theta_i[n] * p[, 2]
        p[, 3] <- -p[, 1]
        p[, 4] <- -p[, 2]
      } else if(k == n) {
        p[, 1] <- 0
      } else {
        p[, 1] <- (1 - Kronicker.Delta(k, j)) * p[, 1]
        p[, 2] <- (1 - Kronicker.Delta(k, j)) * p[, 2]
        p[, 3] <- Theta_i[k] * p[, 3]
        p[, 4] <- Theta_i[k] * p[, 4]
      }
    }
  } else {
    p <- rep(0, 2^{2 * n - 1})
  }

  return(as.vector(p))
}

# Code for testing
# n <- 7
# Theta <- log(matrix(1:n^2, byrow = T, ncol = n))
# p <- matrix(1:2^(2*n-1),ncol = 1)
# for(i in 1:n) {
#   for(j in 1:n) {
#     dQ_x <- create_dQ_async_metastasis_ij(Theta, i, j) %*% p
#     dQ_vec <- dQ.async.metastasis.vec(Theta, p, i, j)
#     print(sum(mapply(function(x, y) {isTRUE(all.equal(x, y))}, dQ_x, dQ_vec)) == length(dQ_vec))
#   }
# }

#------------------------------------------------------------------------------
# Functions to create dQ/dTheta[i, j] 
# these Functions are not necessary for Likelihood_Metastasis.R to work,
# but they are good for debugging and I would keep them for now

create_dQ_Theta_ij <- function(Theta, i, j) {
  dQ_Theta_ij <- create_dQ_sync_ij(Theta, i, j) + create_dQ_async_primary_ij(Theta, i, j) + create_dQ_async_metastasis_ij(Theta, i, j)
  return(dQ_Theta_ij)
}

create_dQ_sync_ij <- function(Theta, i, j) {
  Theta <- exp(Theta)
  n <- nrow(Theta)
  if (i < n) {
    # i < n
    # \otimes k < i
    k <- 1
    otimes_k_smaller_i <- 1
    while(k < i) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(k, j), 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, Theta[i, k]), byrow = T, nrow = 4)
      otimes_k_smaller_i <- tensor_core_at_k %x% otimes_k_smaller_i
      k <- k + 1
    }
    
    tensor_core_at_i <- matrix(c(-Theta[i, i] * (1 - Kronicker.Delta(j, n)), 0, 0, 0, 
                                                 0, 0, 0, 0,
                                                 0, 0, 0, 0,
                                  Theta[i, i] * (1 - Kronicker.Delta(j, n)), 0, 0, 0), byrow = T, ncol = 4)
    
    k <- i + 1
    otimes_k_larger_i <- 1
    while(k < n) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(k, j), 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, Theta[i, k]), byrow = T, nrow = 4)
      otimes_k_larger_i <- tensor_core_at_k %x% otimes_k_larger_i
      k <- k + 1
    }
    tensor_core_at_n <- matrix(c(1, 0, 0, 0), byrow = T, ncol = 2)
    dQ_sync_ij <- tensor_core_at_n %x% (otimes_k_larger_i %x% (tensor_core_at_i %x% otimes_k_smaller_i))
  } else {
    # i = n
    k <- 1
    otimes_k_smaller_i <- 1
    while(k < n) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(k, j), 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, 0,
                                   0, 0, 0, Theta[i, k]), byrow = T, nrow = 4)
      otimes_k_smaller_i <- tensor_core_at_k %x% otimes_k_smaller_i
      k <- k + 1
    }
    tensor_core_at_n <- matrix(c(-Theta[n, n], 0, Theta[n, n], 0), byrow = T, nrow = 2)
    dQ_sync_ij <- tensor_core_at_n %x% otimes_k_smaller_i
  }
  
  return(dQ_sync_ij)
}

create_dQ_async_primary_ij <- function(Theta, i, j) {
  Theta <- exp(Theta)
  n <- nrow(Theta)
  
  if(i < n) {
    # \otimes k < i
    k <- 1
    otimes_k_smaller_i <- 1
    while(k < i) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(k, j), 0, 0, 0,
                                   0, Theta[i, k], 0, 0,
                                   0, 0, 1-Kronicker.Delta(k, j), 0,
                                   0, 0, 0, Theta[i,k]), byrow = T, nrow = 4)
      otimes_k_smaller_i <- tensor_core_at_k %x% otimes_k_smaller_i
      k <- k + 1
    }
    
    tensor_core_at_i <- matrix(c(-Theta[i, i] * (1 - Kronicker.Delta(j, n)), 0, 0, 0,
                                 Theta[i, i] * (1 - Kronicker.Delta(j, n)), 0, 0, 0,
                                 0, 0, -Theta[i, i] * (1 - Kronicker.Delta(j, n)), 0,
                                 0, 0, Theta[i, i] * (1 - Kronicker.Delta(j, n)), 0), byrow = T, ncol = 4)
    
    # \otimes k > i
    k <- i + 1
    otimes_k_larger_i <- 1
    while(k < n) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(k, j), 0, 0, 0,
                                   0, Theta[i, k], 0, 0,
                                   0, 0, 1-Kronicker.Delta(k, j), 0,
                                   0, 0, 0, Theta[i,k]), byrow = T, nrow = 4)
      otimes_k_larger_i <- tensor_core_at_k %x% otimes_k_larger_i
      k <- k + 1
    }
    
    tensor_core_at_n <- matrix(c(0, 0, 0, 1), byrow = T, ncol = 2)
    dQ_async_primary_i_j <- tensor_core_at_n %x% (otimes_k_larger_i %x% (tensor_core_at_i %x% otimes_k_smaller_i))
  } else {
    dQ_async_primary_i_j <- matrix(0, nrow = 2^(2 * n - 1), ncol = 2^(2 * n - 1))
  }
  return(dQ_async_primary_i_j)
}

create_dQ_async_metastasis_ij <- function(Theta, i, j) {
  Theta <- exp(Theta)
  n <- ncol(Theta) 
  
  if(i < n) {
    # i < n
    
    # \otimes k < i
    k <- 1
    otimes_k_smaller_i <- 1
    while(k < i) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(j, k), 0, 0, 0,
                                   0, 1-Kronicker.Delta(j, k), 0, 0,
                                   0, 0, Theta[i, k], 0,
                                   0, 0, 0, Theta[i, k]), byrow = T, ncol = 4)
      otimes_k_smaller_i <- tensor_core_at_k %x% otimes_k_smaller_i
      k <- k + 1
    }
    
    tensor_core_at_i <- matrix(c(-Theta[i, i] * Theta[i, n], 0, 0, 0,
                                 0, -Theta[i, i] * Theta[i, n], 0, 0,
                                 Theta[i, i] * Theta[i, n], 0, 0, 0,
                                 0, Theta[i, i] * Theta[i, n], 0, 0), byrow = T, nrow = 4)
    
    # \otimes k > i
    k <- k + 1
    otimes_k_larger_i <- 1
    while(k < n) {
      tensor_core_at_k <- matrix(c(1-Kronicker.Delta(j, k), 0, 0, 0,
                                   0, 1-Kronicker.Delta(j, k), 0, 0,
                                   0, 0, Theta[i, k], 0,
                                   0, 0, 0, Theta[i, k]), byrow = T, ncol = 4)
      otimes_k_larger_i <- tensor_core_at_k %x% otimes_k_larger_i
      k <- k + 1
    }
    
    tensor_core_at_n <- matrix(c(0, 0, 0, 1), ncol = 2, byrow = T)
    
    dQ_async_metastasis_ij <- tensor_core_at_n %x% (otimes_k_larger_i %x% (tensor_core_at_i %x% otimes_k_smaller_i))
    
  } else {
    # i == n
    dQ_async_metastasis_ij <- matrix(0, ncol = 2^(2 * n - 1), nrow = 2^(2 * n - 1))
  }
  
  return(dQ_async_metastasis_ij)
}

# Code for testing
# n <- 5
# Theta <- log(matrix(1:n^2, byrow = T, ncol = n))
# Q <- create_Q(Theta, T, F)
# rownames(Q) <- genoNames(n) 
# colnames(Q) <- genoNames(n)
# mutation_process_Q <- get_mutation_process(Q)
# 
# for(i in 1:n) {
#   for(j in 1:n) {
#     dQ_i_j <- create_dQ_async_metastasis_ij(Theta, i, j)
#     rownames(dQ_i_j) <- genoNames(n) 
#     colnames(dQ_i_j) <- genoNames(n)
#     mutation_process_i_j <- get_mutation_process(dQ_i_j)
#     flag <- TRUE
#     for(dQ in mutation_process_i_j) {
#       flag <- flag & (dQ %in% mutation_process_Q)
#     }
#     print(flag)
#   }
# }



