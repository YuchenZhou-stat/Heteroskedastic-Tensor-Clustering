setwd("/Users/yczhou/Desktop/tensor_clustering/experiments_revision")
source("tensor_clustering_method.R")
#library(doParallel)

## simulated data.
set.seed(2023)

replicates = 100

k_1 = 3
k_2 = 3
k_3 = 3
d = 3
k = c(k_1, k_2, k_3)
n_1 = 150
n_2 = 150
n_3 = 150
#n_1 = 100
#n_2 = 100
#n_3 = 100
kappa = 10

#k_1 = 5
#k_2 = 4
#k_3 = 3
#d = 3
#k = c(k_1, k_2, k_3)
#n_1 = 50
#n_2 = 100
#n_3 = 150


n = c(n_1, n_2, n_3)
delta_low = 0.1
delta_upper = 0.8
delta.candidate = seq(delta_low,delta_upper,0.05)
sigma = 1
filename <- paste("SNR", "n_1", n_1,
                  "n_2", n_2,
                  "n_3", n_3,
                  "k_1", k_1, 
                  "k_2", k_2, 
                  "k_3", k_3, 
                  "kappa", kappa,
                  "simu", replicates, 'd', d ,'delta_low',delta_low,'delta_upper',delta_upper,sep = "_")
file = paste("results_new/",filename, ".csv", sep = "")
methods <- c("H", "H_HLloyd", "Deflated_Hetero", "De_HLloyd")
#n.index = 1
#delta.index = 1

#for (n.index in 1:length(n.candidate)){
for (delta.index in 1:length(delta.candidate)){
  sum_time <- 0
  time <- rep(0, 4)
  for(i in  1:replicates){
    start_t <- Sys.time()
    seed = i + 2023
    set.seed(seed)
    #n = rep(n.candidate[n.index],d)
    data = TBM_hetero.generator.ill.conditioned(n, k, kappa, 40*(n[1]*n[2]*n[3])^(-delta.candidate[delta.index]/3), sigma, seed)
    
    time_tmp <- Sys.time()
    z.HeteroSC = HeteroPCA.SC(data$tensor, k)[[1]]
    time[1] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.H = HO.Lloyd(data$tensor, z.HeteroSC)
    time[2] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    myout <- Deflated_HeteroPCA.SC(data$tensor, k)
    z_hetero = myout[[1]]
    time[3] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.hetero = HO.Lloyd(data$tensor, z_hetero)
    time[4] <- Sys.time() - time_tmp
    
    H_err = 1 - ARI(z.HeteroSC, data$labels, mode="averaged")
    Lloyd_H_err = 1 - ARI(z.Lloyd.H, data$labels, mode="averaged")
    hetero_err = 1 - ARI(z_hetero, data$labels, mode="averaged")
    Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, data$labels, mode="averaged")
    error = c(H_err,Lloyd_H_err,hetero_err,Lloyd_hetero_err)
    comp_time <- Sys.time() - start_t
    results <- NULL
    for(j in 1:4){
      results <- rbind(results,data.frame(seed = i, n_1 = n_1, n_2 = n_2, n_3 = n_3, delta = delta.candidate[delta.index], 
                                          error = error[j], method = methods[j], comp_time = time[j]))
    }
    if(file.exists(file)){
      write.table(results, file = file, row.names = FALSE, col.names = FALSE, append = TRUE)
    }else{
      write.table(results, file = file, row.names = FALSE)
    }
    sum_time <- sum_time + comp_time
  }
  print(sprintf("finished n1 = %d, n2 = %d, n3 = %d, delta = %f, within %s seconds", n_1, n_2, n_3, delta.candidate[delta.index], sum_time))
}

