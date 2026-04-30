setwd("/Users/yczhou/Desktop/tensor_clustering/experiments_revision")
source("tensor_clustering_method.R")
#library(doParallel)

## simulated data.
set.seed(2023)

replicates = 50

k_total = 4
d = 3
k = rep(k_total,d)
n_1 = 200
n_2 = 200
n_3 = 200
n = c(n_1, n_2, n_3)

scale_min = 4
scale_max = 8
scale.candidate = seq(scale_min,scale_max,0.25)
filename <- paste("STBM", 
                  "n1", n_1, 
                  "n2", n_2,
                  "n3", n_3, 
                  "k1", k_total, 
                  "k2", k_total, 
                  "k3", k_total,  
                  "simu", replicates, 'd', d ,'scale_min',scale_min,'scale_max',scale_max,sep = "_")
file = paste("results/",filename, ".csv", sep = "")
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
#n.index = 1
#scale.index = 1

  for (scale.index in 1:length(scale.candidate)){
    sum_time <- 0
    time <- rep(0, 4)
    for(i in  1:replicates){
      start_t <- Sys.time()
      seed = i + 2023
      set.seed(seed)
      data = TBM.generator.bernoulli(n, k, (10)*scale.candidate[scale.index]/((n[1]*n[2]*n[3])^(1/2)), (0.1)*scale.candidate[scale.index]/((n[1]*n[2]*n[3])^(1/2)), seed)
      
      time_tmp <- Sys.time()
      z.HOSC = HO.SC(data$tensor, k)
      time[1] <- Sys.time() - time_tmp
      
      time_tmp <- Sys.time()
      z.Lloyd.HOSC = HO.Lloyd(data$tensor, z.HOSC)
      time[2] <- Sys.time() - time_tmp
      
      time_tmp <- Sys.time()
      myout <- Deflated_HeteroPCA.SC(data$tensor, k)
      z_hetero = myout[[1]]
      time[3] <- Sys.time() - time_tmp
      
      time_tmp <- Sys.time()
      z.Lloyd.hetero = HO.Lloyd(data$tensor, z_hetero)
      time[4] <- Sys.time() - time_tmp
      
      HOSC_err = 1 - ARI(z.HOSC, data$labels, mode="averaged")
      Lloyd_err = 1 - ARI(z.Lloyd.HOSC, data$labels, mode="averaged")
      hetero_err = 1 - ARI(z_hetero, data$labels, mode="averaged")
      Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, data$labels, mode="averaged")
      error = c(HOSC_err,Lloyd_err,hetero_err,Lloyd_hetero_err)
      comp_time <- Sys.time() - start_t
      results <- NULL
      for(j in 1:4){
        results <- rbind(results,data.frame(seed = i, n_1 = n_1, n_2 = n_2, n_3 = n_3, scale = scale.candidate[scale.index], 
                                            error = error[j], method = methods[j], comp_time = time[j]))
      }
      if(file.exists(file)){
        write.table(results, file = file, row.names = FALSE, col.names = FALSE, append = TRUE)
      }else{
        write.table(results, file = file, row.names = FALSE)
      }
      sum_time <- sum_time + comp_time
    }
    print(sprintf("finished scale = %f, within %s seconds", scale.candidate[scale.index], sum_time))
  }


prop_list <- list(c(0.4,0.3,0.2,0.1), c(0.4,0.3,0.2,0.1), c(0.4,0.3,0.2,0.1))
k_1 = 4
k_2 = 4
k_3 = 4
d = 3
k = c(k_1, k_2, k_3)
n_1 = 50
n_2 = 50
n_3 = 500
n = c(n_1, n_2, n_3)
scale_min = 2
scale_max = 10

scale.candidate = seq(scale_min,scale_max,0.5)
filename <- paste("STBM", "unbalanced", "n1", n_1, 
                  "n2", n_2,
                  "n3", n_3,
                  "k1", k_1, 
                  "k2", k_2, 
                  "k3", k_3, 
                  "simu", replicates, 'd', d ,'scale_min',scale_min,'scale_max',scale_max,sep = "_")
file = paste("results_new/",filename, ".csv", sep = "")
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
#n.index = 1
#scale.index = 1

#for (n.index in 1:length(n.candidate)){
for (scale.index in 1:length(scale.candidate)){
  sum_time <- 0
  time <- rep(0, 4)
  for(i in  1:replicates){
    start_t <- Sys.time()
    seed = i + 2023
    set.seed(seed)
    data = TBM.generator.bernoulli.unbalanced(n, k, scale.candidate[scale.index]/((n[1]*n[2]*n[3])^(1/2)), prop_list, seed)
    
    time_tmp <- Sys.time()
    z.HOSC = HO.SC(data$tensor, k)
    time[1] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.HOSC = HO.Lloyd(data$tensor, z.HOSC)
    time[2] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    myout <- Deflated_HeteroPCA.SC(data$tensor, k)
    z_hetero = myout[[1]]
    time[3] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.hetero = HO.Lloyd(data$tensor, z_hetero)
    time[4] <- Sys.time() - time_tmp
    
    HOSC_err = 1 - ARI(z.HOSC, data$labels, mode="averaged")
    Lloyd_err = 1 - ARI(z.Lloyd.HOSC, data$labels, mode="averaged")
    hetero_err = 1 - ARI(z_hetero, data$labels, mode="averaged")
    Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, data$labels, mode="averaged")
    error = c(HOSC_err,Lloyd_err,hetero_err,Lloyd_hetero_err)
    comp_time <- Sys.time() - start_t
    results <- NULL
    for(j in 1:4){
      results <- rbind(results,data.frame(seed = i, n_1 = n_1, n_2 = n_2, n_3 = n_3, scale = scale.candidate[scale.index], 
                                          error = error[j], method = methods[j], comp_time = time[j]))
    }
    if(file.exists(file)){
      write.table(results, file = file, row.names = FALSE, col.names = FALSE, append = TRUE)
    }else{
      write.table(results, file = file, row.names = FALSE)
    }
    sum_time <- sum_time + comp_time
  }
  print(sprintf("finished n1 = %d, n2 = %d, n3 = %d, scale = %f, within %s seconds", n_1, n_2, n_3, scale.candidate[scale.index], sum_time))
}
#}
