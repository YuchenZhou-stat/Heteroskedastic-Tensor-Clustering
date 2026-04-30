library(rTensor)
library(gtools)
file.sources <- list.files(c("LICORS/R"), 
                           pattern="*.R$", full.names=TRUE, 
                           ignore.case=TRUE)
sapply(file.sources,source,.GlobalEnv)
library(mclust)

#data_generating_from Han et al. 2022
TBM_hetero.generator <- function(p, r, delta = 1, sigma = 1, seed){
  set.seed(seed)
  try(if(length(p)!=length(r) | any(r>p)) stop("invalid input: Y should be of tensor type."))
  
  try(if(delta <= 0) stop("Invalid delta."))
  
  # Generate core tensor.
  d = length(p)
  #S.array = array(c(1,2,2,4,2,4,4,8), dim = c(2,2,2))
  S.array = array(rnorm(prod(r)), dim=r) # Core tensor
  #S.array = array(1 : prod(r), dim=r) # Core tensor
  S = as.tensor(S.array)
  # adjust minimal separation.
  Sk = k_unfold(S, 1)@data
  delta.min = sqrt(sum((Sk[1,] - Sk[2,])^2))
  for (i in 1:d){
    Sk = k_unfold(S, i)@data
    for (k1 in 1:(r[i]-1)){
      for (k2 in (k1+1):r[i]){
        delta.min = min(delta.min, sqrt(sum((Sk[k1,] - Sk[k2,])^2)))
      }
    }
  }
  S = S * delta / delta.min
  
  # Generate random labels 
  Z = list()
  z = list()
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = rep(1:r[i], each = floor(p[i]/r[i]))
    zi = c(zi, rep(1, p[i]-length(zi)))
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  
  a = list()
  for (i in 1:d){
    ai = matrix(runif(p[i],0,2), ncol = 1)
    a = c(a, list(ai))
  }
  A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  E = as.tensor(array(rnorm(prod(p)), dim=p)) # Noise tensor
  Y = X + sigma * A * E
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}


TBM_hetero.generator.ill.conditioned <- function(p, r, kappa, delta = 1, sigma = 1, seed){
  set.seed(seed)
  try(if(length(p)!=length(r) | any(r>p)) stop("invalid input: Y should be of tensor type."))
  
  try(if(delta <= 0) stop("Invalid delta."))
  
  # Generate core tensor.
  d = length(p)
  S.array = array(rep(0, prod(r)), dim=r)
  S.array[1,1,1] = kappa
  for (i in 2:min(r)){
    S.array[i,i,i] = 1
  }
  #for (i in 1:r[1]){
  #  for (j in 1:r[2]){
  #    for (k in 1:r[3]){
  #      S.array[i,j,k] = i*j*k
  #    }
  #  }
  #}
  S = as.tensor(S.array)
  # adjust minimal separation.
  Sk = k_unfold(S, 1)@data
  delta.min = sqrt(sum((Sk[1,] - Sk[2,])^2))
  for (i in 1:d){
    Sk = k_unfold(S, i)@data
    for (k1 in 1:(r[i]-1)){
      for (k2 in (k1+1):r[i]){
        delta.min = min(delta.min, sqrt(sum((Sk[k1,] - Sk[k2,])^2)))
      }
    }
  }
  S = S * delta / delta.min
  
  # Generate random labels 
  Z = list()
  z = list()
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = rep(1:r[i], each = floor(p[i]/r[i]))
    zi = c(zi, rep(1, p[i]-length(zi)))
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  
  a = list()
  for (i in 1:d){
    ai = matrix(runif(p[i],0,2), ncol = 1)
    a = c(a, list(ai))
  }
  A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  E = as.tensor(array(rnorm(prod(p)), dim=p)) # Noise tensor
  #Y = X
  Y = X + sigma * A * E
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}



#data generating which allows unbalanced cluster sizes
TBM_hetero.generator.unbalanced <- function(p, r, prop_list, delta = 1, sigma = 1, seed){
  set.seed(seed)
  try(if(length(p)!=length(r) | any(r>p)) stop("invalid input: Y should be of tensor type."))
  
  try(if(delta <= 0) stop("Invalid delta."))
  
  # Generate core tensor.
  d = length(p)
  #S.array = array(c(1,2,2,4,2,4,4,8), dim = c(2,2,2))
  S.array = array(rnorm(prod(r)), dim=r) # Core tensor
  #S.array = array(1 : prod(r), dim=r) # Core tensor
  S = as.tensor(S.array)
  # adjust minimal separation.
  Sk = k_unfold(S, 1)@data
  delta.min = sqrt(sum((Sk[1,] - Sk[2,])^2))
  for (i in 1:d){
    Sk = k_unfold(S, i)@data
    for (k1 in 1:(r[i]-1)){
      for (k2 in (k1+1):r[i]){
        delta.min = min(delta.min, sqrt(sum((Sk[k1,] - Sk[k2,])^2)))
      }
    }
  }
  S = S * delta / delta.min
  
  # Generate random labels 
  Z = list()
  z = list()
  w = label_generator(prop_list, p, r)
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = w[[i]]
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  
  a = list()
  for (i in 1:d){
    ai = matrix(runif(p[i],0,2), ncol = 1)
    a = c(a, list(ai))
  }
  A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  E = as.tensor(array(rnorm(prod(p)), dim=p)) # Noise tensor
  Y = X + sigma * A * E
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}

TBM_hetero.generator.rank1 <- function(p, r, prop_list, delta = 1, sigma = 1, seed){
  set.seed(seed)
  try(if(length(p)!=length(r) | any(r>p)) stop("invalid input: Y should be of tensor type."))
  
  try(if(delta <= 0) stop("Invalid delta."))
  
  # Generate core tensor.
  d = length(p)
  #S.array = array(c(1,2,2,4,2,4,4,8), dim = c(2,2,2))
  S.array = optimal_rank1_tensor_d(r,1) # Core tensor
  #S.array = array(1 : prod(r), dim=r) # Core tensor
  S = as.tensor(S.array)
  # adjust minimal separation.
  Sk = k_unfold(S, 1)@data
  delta.min = sqrt(sum((Sk[1,] - Sk[2,])^2))
  for (i in 1:d){
    Sk = k_unfold(S, i)@data
    for (k1 in 1:(r[i]-1)){
      for (k2 in (k1+1):r[i]){
        delta.min = min(delta.min, sqrt(sum((Sk[k1,] - Sk[k2,])^2)))
      }
    }
  }
  S = S * delta / delta.min
  
  # Generate random labels 
  Z = list()
  z = list()
  w = label_generator(prop_list, p, r)
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = w[[i]]
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  
  a = list()
  for (i in 1:d){
    ai = matrix(runif(p[i],0,2), ncol = 1)
    a = c(a, list(ai))
  }
  A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  E = as.tensor(array(rnorm(prod(p)), dim=p)) # Noise tensor
  Y = X + sigma * A * E
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}

TBM_hetero.generator.unbalanced2 <- function(p, r, prop_list, hetero, delta = 1, sigma = 1, seed){
  set.seed(seed)
  try(if(length(p)!=length(r) | any(r>p)) stop("invalid input: Y should be of tensor type."))
  
  try(if(delta <= 0) stop("Invalid delta."))
  
  # Generate core tensor.
  d = length(p)
  #S.array = array(c(1,2,2,4,2,4,4,8), dim = c(2,2,2))
  S.array = array(rnorm(prod(r)), dim=r) # Core tensor
  #S.array = array(1 : prod(r), dim=r) # Core tensor
  S = as.tensor(S.array)
  # adjust minimal separation.
  Sk = k_unfold(S, 1)@data
  delta.min = sqrt(sum((Sk[1,] - Sk[2,])^2))
  for (i in 1:d){
    Sk = k_unfold(S, i)@data
    for (k1 in 1:(r[i]-1)){
      for (k2 in (k1+1):r[i]){
        delta.min = min(delta.min, sqrt(sum((Sk[k1,] - Sk[k2,])^2)))
      }
    }
  }
  S = S * delta / delta.min
  
  # Generate random labels 
  Z = list()
  z = list()
  w = label_generator(prop_list, p, r)
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = w[[i]]
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  
  a = list()
  for (i in 1:d){
    ai = sqrt(matrix(runif(p[i],2-hetero,2+hetero), ncol = 1))
    a = c(a, list(ai))
  }
  A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  E = as.tensor(array(rnorm(prod(p)), dim=p)) # Noise tensor
  Y = X + sigma * A * E
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}


TBM.generator.bernoulli <- function(p, r, q1, q2, seed){
  set.seed(seed)
  # Generate core tensor.
  d = length(p)
  
  S.array = array(rep(q2, prod(r)), dim=r)
  for (i in 1:r[1]){
    #S.array[i,i,i] = q1
    S.array[i,i,i] = q1*(1 - (i-1)/(2*(r[1]-1)))
    #S.array[i,i,i] = q1*i
    #S.array[i,i,i] = q1*(1 - (i-1)/6)
  }
  S = as.tensor(S.array) 
  
  
  #S.array = array(runif(prod(r)), dim=r) # Core tensor
  #S = as.tensor(S.array) 
  # do the scaling
  #a = list()
  #for (i in 1:d){
  #  ai = matrix(runif(r[i],0,1), ncol = 1)
  #  a = c(a, list(ai))
  #}
  #A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  #S = S*A/max((S*A)@data)
  #S = S/max(S@data) #standardize S
  #S = S * scale 
  
  # Generate random labels 
  Z = list()
  z = list()
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = rep(1:r[i], each = floor(p[i]/r[i]))
    zi = c(zi, rep(1, p[i]-length(zi)))
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  
  # Generate random labels 
  #Z = list()
  #z = list()
  #Z1 = matrix(0, p[1], r[1])
  #z1 = rep(1:r[1], each = floor(p[1]/(r[1])))
  #z1 = c(z1, rep(1, p[1]-length(z1)))
  #z1 = permute(z1[1:p[1]])
  #for (j in 1:p[1]){
  #  Z1[j, z1[j]] = 1
  #} 
  #for (i in 1:d){
  #  Zi = Z1
  #  zi = z1
  #  Z = c(Z, list(Zi))
  #  z = c(z, list(zi))
  #}
  X = ttl(S, Z, 1:d)
  Y = as.tensor(array(rbinom(prod(p),1,c(X@data)),dim = p))
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}


TBM.generator.bernoulli.unbalanced <- function(p, r, scale, prop_list, seed){
  set.seed(seed)
  # Generate core tensor.
  d = length(p)
  u = seq(0.1, 10, length.out = r[1])
  v = seq(0.1, 10, length.out = r[2])
  w = seq(0.1, 10, length.out = r[3])
  #r_min = min(r[1], r[2], r[3])
  S.array = array(rep(0, prod(r)), dim=r)
  for (i in 1:r[1]){
    for (j in 1:r[2]){
      for (k in 1:r[3]){
        S.array[i,j,k] = scale*(u[i]+v[j]+w[k])
      }
    }
  }
  #for (i in 1:r_min){
    #S.array[i,i,i] = q1
    #S.array[i,i,i] = q1*(1 - (i-1)/(2*(r_min-1)))
    #S.array[i,i,i] = q1*i
    #S.array[i,i,i] = q1*(1 - (i-1)/6)
  #}
  S = as.tensor(S.array) 
  
  
  #S.array = array(runif(prod(r)), dim=r) # Core tensor
  #S = as.tensor(S.array) 
  # do the scaling
  #a = list()
  #for (i in 1:d){
  #  ai = matrix(runif(r[i],0,1), ncol = 1)
  #  a = c(a, list(ai))
  #}
  #A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  #S = S*A/max((S*A)@data)
  #S = S/max(S@data) #standardize S
  #S = S * scale 
  
  # Generate random labels 
  Z = list()
  z = list()
  w = label_generator(prop_list, p, r)
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = w[[i]]
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  Y = as.tensor(array(rbinom(prod(p),1,c(X@data)),dim = p))
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}


TBM.generator.bernoulli.random <- function(p, r, scale, seed){
  set.seed(seed)
  # Generate core tensor.
  d = length(p)
  
  
  S.array = array(runif(prod(r)), dim=r) # Core tensor
  S = as.tensor(S.array) 
  # do the scaling
  a = list()
  for (i in 1:d){
    ai = matrix(runif(r[i],0,1), ncol = 1)
    a = c(a, list(ai))
  }
  A = ttl(as.tensor(array(1, dim = rep(1,d))), a, 1:d)
  S = S*A/max((S*A)@data)
  S = S/max(S@data) #standardize S
  S = S * scale 
  
  # Generate random labels 
  Z = list()
  z = list()
  for (i in 1:d){
    Zi = matrix(0, p[i], r[i])
    zi = rep(1:r[i], each = floor(p[i]/r[i]))
    zi = c(zi, rep(1, p[i]-length(zi)))
    zi = permute(zi[1:p[i]])
    for (j in 1:p[i]){
      Zi[j, zi[j]] = 1
    }
    Z = c(Z, list(Zi))
    z = c(z, list(zi))
  }
  X = ttl(S, Z, 1:d)
  Y = as.tensor(array(rbinom(prod(p),1,c(X@data)),dim = p))
  return(list("tensor" = Y, "labels" = z, "signal" = X))
}


#HeteroPCA
HeteroPCA <- function(Y, r){
  iter_hetero = 100
  G_hetero = Y%*%t(Y) - diag(diag(Y%*%t(Y)))
  for (i in 1:iter_hetero){
    U_hetero = svd(G_hetero)$u[,1:r]
    G_hetero = G_hetero - diag(diag(G_hetero)) + diag(diag(U_hetero%*%t(U_hetero)%*%G_hetero))
  }
  return(U_hetero)
}

#HeteroPCA + kmeans
HeteroPCA.SC <- function(Y, r){
  try(if(missing("Y")) stop("missing argument: Y is required as the tensor type."))
  try(if(missing("r")) stop("missing argument: r is required as a scalar or vector."))
  try(if(class(Y) != "Tensor") stop("invalid input: Y should be of tensor type."))
  p = dim(Y)
  d = length(p)
  iter = 10
  Y1 = k_unfold(Y, 1)@data
  
  if(is.atomic(r) && length(r)==1){
    r = rep(r, d)
  }
  
  U_t = list();
  PC = list()
  for (i in 1:d){#Initialization via Truncated Deflated-HeteroPCA
    PC[[i]] <- HeteroPCA(k_unfold(Y, i)@data,r[i])
    U_t = c(U_t, list(t(PC[[i]])))
  }
  
  #for(i in 1:d){#Projection
  #  A = ttl(Y, U_t[-i], (1:d)[-i])
  #  A_matrix = k_unfold(A, i)@data
  #  svd.result = svd(A_matrix)
  #  U_t[[i]] = t(svd.result$u[,1:r[i]])
  #}
  
  z_h = list(); Kmatrix = list();
  for (i in 1:d){
    Kmatrix[[i]] = ttl(Y,U_t[-i],(1:d)[-i])
    Kmatrix[[i]] = t(U_t[[i]]) %*% U_t[[i]] %*% k_unfold(Kmatrix[[i]],i)@data
    z_h = c(z_h, list(kmeanspp(Kmatrix[[i]], r[i])$cluster))
    #z_hetero = c(z_hetero, list(kmeans(Kmatrix[[i]], centers = r[i])$cluster))
  }
  
  return(list(z_h, PC, Kmatrix))
}

#Thresholded Deflated-HeteroPCA
Deflated_HeteroPCA <- function(Y, r, iter, tau){
  G = Y%*%t(Y) - diag(diag(Y%*%t(Y)))
  r_select = 1
  for(i in 1:r){
    if(svd(G)$d[1] > 4*svd(G)$d[i]){
      break
    }
    else if((svd(G)$d[i] > (r/(r-1))*svd(G)$d[i+1]) || (i == r)){
      r_select = i
    }
  }
  if(r_select >= 1){
    U1_hat = svd(G)$u[,1:r_select]
    for (i in 1:iter){
      G = G - diag(diag(G)) + diag(diag(U1_hat%*%t(U1_hat)%*%G))
      U1_hat = svd(G)$u[,1:r_select]
    }
    G = G - diag(diag(G)) + diag(diag(U1_hat%*%t(U1_hat)%*%G))
    U_hat = U1_hat
  }
  while((r - r_select >= 1) & (svd(G)$d[r_select + 1] > tau)){
    r_temp = r_select
    for (i in (r_temp+1):r){
      if(svd(G)$d[r_temp+1] > 4*svd(G)$d[i]){
        break
      }
      else if((svd(G)$d[i] > (r/(r-1))*svd(G)$d[i+1]) || (i == r)){
        r_select = i
      }
    }
    for (i in 1:iter){
      U_hat = svd(G)$u[,1:r_select]
      G = G - diag(diag(G)) + diag(diag(U_hat%*%t(U_hat)%*%G))
    }
  }
  return(U_hat)
}

#Deflated-HeteroPCA + kmeans
Deflated_HeteroPCA.SC <- function(Y, r){
  try(if(missing("Y")) stop("missing argument: Y is required as the tensor type."))
  try(if(missing("r")) stop("missing argument: r is required as a scalar or vector."))
  try(if(class(Y) != "Tensor") stop("invalid input: Y should be of tensor type."))
  p = dim(Y)
  d = length(p)
  iter = 10
  Y1 = k_unfold(Y, 1)@data
  tau = (1.1)*(svd(Y1)$d[r[1]+1]/sqrt(prod(p)/p[1]))^2*(prod(p))^(1/2)
  #tau = 0
  
  if(is.atomic(r) && length(r)==1){
    r = rep(r, d)
  }
  
  U_t = list();
  PC = list()
  for (i in 1:d){#Initialization via Truncated Deflated-HeteroPCA
    PC[[i]] <- Deflated_HeteroPCA(k_unfold(Y, i)@data,r[i], iter, tau)
    U_t = c(U_t, list(t(PC[[i]])))
  }
  
  #for(i in 1:d){#Projection
  #  A = ttl(Y, U_t[-i], (1:d)[-i])
  #  A_matrix = k_unfold(A, i)@data
  #  svd.result = svd(A_matrix)
  #  U_t[[i]] = t(svd.result$u[,1:r[i]])
  #}
  
  z_hetero = list(); Kmatrix = list();
  #z_refine = list();
  for (i in 1:d){
    Kmatrix[[i]] = ttl(Y,U_t[-i],(1:d)[-i])
    Kmatrix[[i]] = t(U_t[[i]]) %*% U_t[[i]] %*% k_unfold(Kmatrix[[i]],i)@data
    kmeansout <- kmeanspp(Kmatrix[[i]], r[i])
    z_hetero = c(z_hetero, list(kmeansout$cluster))
    #kmeansout2 <- kmeans(Kmatrix[[i]], center = kmeansout$centers, r[i])
    #z_refine = c(z_refine, list(kmeansout2$cluster))
    #z_hetero = c(z_hetero, list(kmeans(Kmatrix[[i]], centers = r[i])$cluster))
    #zi = rep(0, p[i])
    #for (j in 1:p[i]){
    #  dist = Inf
    #  nearest = 1
    #  for (k in 1:r[i]){
    #    dist_k = sqrt(sum((Kmatrix[[i]][j,] - kmeansout$centers[k,])^2))
    #    if(dist_k < dist){
    #      nearest = k
    #      dist = dist_k
    #    }
    #  }
    #  zi[j] = nearest
    #}
    #z_refine = c(z_refine, list(zi))
  }
  
  return(list(z_hetero, PC, Kmatrix))
}


#Deflated-HeteroPCA + kmeans with different thresholding choices 
Deflated_HeteroPCA.SC.tau <- function(Y, r, constant){
  try(if(missing("Y")) stop("missing argument: Y is required as the tensor type."))
  try(if(missing("r")) stop("missing argument: r is required as a scalar or vector."))
  try(if(class(Y) != "Tensor") stop("invalid input: Y should be of tensor type."))
  p = dim(Y)
  d = length(p)
  iter = 10
  Y1 = k_unfold(Y, 1)@data
  tau = constant*(svd(Y1)$d[r[1]+1]/sqrt(prod(p)/p[1]))^2*(prod(p))^(1/2)
  #tau = 0
  
  if(is.atomic(r) && length(r)==1){
    r = rep(r, d)
  }
  
  U_t = list();
  PC = list()
  for (i in 1:d){#Initialization via Truncated Deflated-HeteroPCA
    PC[[i]] <- Deflated_HeteroPCA(k_unfold(Y, i)@data,r[i], iter, tau)
    U_t = c(U_t, list(t(PC[[i]])))
  }
  
  #for(i in 1:d){#Projection
  #  A = ttl(Y, U_t[-i], (1:d)[-i])
  #  A_matrix = k_unfold(A, i)@data
  #  svd.result = svd(A_matrix)
  #  U_t[[i]] = t(svd.result$u[,1:r[i]])
  #}
  
  z_hetero = list(); Kmatrix = list();
  #z_refine = list();
  for (i in 1:d){
    Kmatrix[[i]] = ttl(Y,U_t[-i],(1:d)[-i])
    Kmatrix[[i]] = t(U_t[[i]]) %*% U_t[[i]] %*% k_unfold(Kmatrix[[i]],i)@data
    kmeansout <- kmeanspp(Kmatrix[[i]], r[i])
    z_hetero = c(z_hetero, list(kmeansout$cluster))
    #kmeansout2 <- kmeans(Kmatrix[[i]], center = kmeansout$centers, r[i])
    #z_refine = c(z_refine, list(kmeansout2$cluster))
    #z_hetero = c(z_hetero, list(kmeans(Kmatrix[[i]], centers = r[i])$cluster))
    #zi = rep(0, p[i])
    #for (j in 1:p[i]){
    #  dist = Inf
    #  nearest = 1
    #  for (k in 1:r[i]){
    #    dist_k = sqrt(sum((Kmatrix[[i]][j,] - kmeansout$centers[k,])^2))
    #    if(dist_k < dist){
    #      nearest = k
    #      dist = dist_k
    #    }
    #  }
    #  zi[j] = nearest
    #}
    #z_refine = c(z_refine, list(zi))
  }
  
  return(list(z_hetero, PC, Kmatrix))
}



#' Transform a label vector to a weighted matrix
#' Non-exported.
#' @param z vector of clustering labels
#' @return Corresponding weighted matrix
toweightmatrix <- function(z){
  d = length(z)
  W = list()
  for (i in 1:d){
    zi = z[[i]]
    pi = length(zi)
    ri = length(unique(zi))
    Wi = matrix(0, ri, pi)
    for (k in 1:ri){
      zi.subset = which(zi==k)
      Wi[k,zi.subset] = 1/length(zi.subset)
    }
    W = c(W, list(Wi))
  }
  return(W)
}

#' Transform a label vector to a membership matrix.
#' Non-exported.
#' @param z vector of clustering labels.
#' @return Corresponding membership matrix.
tomembershipmatrix <- function(z){
  d = length(z)
  Z = list()
  for (i in 1:d){
    zi = z[[i]]
    pi = length(zi)
    ri = length(unique(zi))
    Zi = matrix(0, pi, ri)
    for (k in 1:ri){
      zi.subset = which(zi==k)
      Zi[zi.subset,k] = 1
    }
    Z = c(Z, list(Zi))
  }
  return(Z)
}

#' Matricized tensor spectral clustering using kmeans++.
#' @param Y Observed tensor.
#' @param r Vector of cluster numbers.
#' @return clustering labels.
#' @export
SC <- function(Y, r){
  try(if(missing("Y")) stop("missing argument: Y is required as the tensor type."))
  try(if(missing("r")) stop("missing argument: r is required as a scalar or vector."))
  try(if(class(Y) != "Tensor") stop("invalid input: Y should be of tensor type."))
  p = dim(Y)
  d = length(p)
  
  if(is.atomic(r) && length(r)==1){
    r = rep(r, d)
  }
  
  z_0 = list();
  for (i in 1:d){
    MY = k_unfold(Y, i)@data
    V_0 = svd(MY)$v[,1:r[i]]
    A.bar = MY %*% V_0
    z_0 = c(z_0, list(kmeanspp(A.bar, r[i])$cluster))
  }
  
  return(z_0)
}

#' High-order tensor spectral clustering.
#' @param Y Observed tensor.
#' @param r Vector of cluster numbers.
#' @return clustering labels.
#' @export
HO.SC <- function(Y, r){
  try(if(missing("Y")) stop("missing argument: Y is required as the tensor type."))
  try(if(missing("r")) stop("missing argument: r is required as a scalar or vector."))
  try(if(class(Y) != "Tensor") stop("invalid input: Y should be of tensor type."))
  p = dim(Y)
  d = length(p)
  
  if(is.atomic(r) && length(r)==1){
    r = rep(r, d)
  }
  
  U_t = list();
  for (i in 1:d){#Initialization
    MY = k_unfold(Y, i)@data
    U_t = c(U_t, list(t(svd(MY%*%t(MY))$u[,1:r[i]])))
  }
  
  for(i in 1:d){#Projection
    A = ttl(Y, U_t[-i], (1:d)[-i])
    A_matrix = k_unfold(A, i)@data
    svd.result = svd(A_matrix)
    U_t[[i]] = t(svd.result$u[,1:r[i]])
  }
  
  z_0 = list();
  for (i in 1:d){
    Kmatrix = ttl(Y,U_t[-i],(1:d)[-i])
    Kmatrix = t(U_t[[i]]) %*% U_t[[i]] %*% k_unfold(Kmatrix,i)@data
    z_0 = c(z_0, list(kmeanspp(Kmatrix, r[i])$cluster))
  }
  
  return(z_0)
}


#' High-order Lloyd Algorithm.
#' @param Y Observed tensor.
#' @param z Vector of initialized labels. 
#' @param t_max maximum number of iterations.
#' @return Estimated clustering labels.
#' @export
HO.Lloyd <- function(Y, z, t_max = 10){
  try(if(missing("Y")) stop("missing argument: Y is required as the tensor type."))
  try(if(missing("z")) stop("missing argument: z is required as a vector."))
  try(if(class(Y) != "Tensor") stop("invalid input: Y should be of tensor type."))
  d = length(z)
  r = rep(0, d)
  for (i in 1:d){
    r[i] = length(unique(z[[i]]))
  }
  for (iter in 1:t_max){
    # create weight matrix
    W = toweightmatrix(z)
    z = list()
    S.hat = ttl(Y, W, (1:d))
    for (i in 1:d){
      S.hat.matrix = k_unfold(S.hat, i)@data
      A.matrix = k_unfold(ttl(Y, W[-i], (1:d)[-i]) , i)@data
      zi = rep(0, nrow(A.matrix))
      for (j in 1:nrow(A.matrix)){
        dist = Inf
        nearest = 1
        for (k in 1:nrow(S.hat.matrix)){
          dist_k = sqrt(sum((A.matrix[j,] - S.hat.matrix[k,])^2))
          if(dist_k < dist){
            nearest = k
            dist = dist_k
          }
        }
        zi[j] = nearest
      }
      z = c(z, list(zi))
    }
  }
  
  return(z)
  
}

#' Calculate ARI of two sets of labels.
#' @param z First clustering vector.
#' @param z.hat Second clustering vector.
#' @param mode Averaged/Minimal ARI.
#' @return Calculated ARI.
ARI <- function(z,z.hat,mode=c("averaged","minimal")){
  d = length(z)
  if (mode == "minimal"){
    ari = 1
    for (i in 1:d){
      ari = min(ari, adjustedRandIndex(z[[i]],z.hat[[i]]))
    }
  }
  else{
    ari = rep(0,d)
    for (i in 1:d){
      ari[i] = adjustedRandIndex(z[[i]],z.hat[[i]])
    }
    ari = sum(ari)/d
  }
  return(ari)
}

# function that takes p_i, prop_i, k_i as input
cluster_label_generator <- function(prop, p){
  k = length(prop)
  size = floor(prop[1:(k-1)] * p)
  size[k] = p-sum(size[1:(k-1)])
  z <- rep(1:k, size)
  return(z)
}

# function that generates labels
label_generator <- function(prop_list, p_list, k_list){
  z <- list()
  for(d in 1:3){
    #call the function above to generate zd.
    z <- c(z, list(cluster_label_generator(prop_list[[d]], p_list[d])))
  }
  return(z)
}

optimal_rank1_tensor_d <- function(r, sigma) {
  d <- length(r)
  
  factors <- lapply(r, function(rm) {
    u <- (1:rm) - (rm + 1) / 2
    sigma^(1/d) * u / sqrt(sum(u^2))
  })
  
  T <- factors[[1]]
  dim(T) <- c(length(T), rep(1, d - 1))
  
  if (d >= 2) {
    for (m in 2:d) {
      T <- outer(T, factors[[m]])
    }
  }
  
  dim(T) <- r
  return(T)
}