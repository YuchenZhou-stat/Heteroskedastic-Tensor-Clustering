setwd("/Users/yczhou/Desktop/tensor_clustering/experiments_revision")
source("tensor_clustering_method.R")
#library(doParallel)

## simulated data.
set.seed(2023)

replicates = 100

prop_list <- list(c(0.2,0.2,0.2,0.2,0.2), c(0.2,0.2,0.2,0.2,0.2), c(0.2,0.2,0.2,0.2,0.2))

k_1 = 5
k_2 = 5
k_3 = 5
d = 3
k = c(k_1, k_2, k_3)
n_1 = 150
n_2 = 150
n_3 = 150
#k_1 = 3
#k_2 = 3
#k_3 = 3
#d = 3
#k = c(k_1, k_2, k_3)
#n_1 = 100
#n_2 = 100
#n_3 = 100

#prop_list <- list(c(1/3, 1/3, 1/3), c(1/3, 1/3, 1/3), c(1/3, 1/3, 1/3))

#k_1 = 2
#k_2 = 3
#k_3 = 4
#d = 3
#k = c(k_1, k_2, k_3)
#n.candidate = 150
#n_1 = 30
#n_2 = 100
#n_3 = 200

#k_1 = 3
#k_2 = 4
#k_3 = 5
#d = 3
#k = c(k_1, k_2, k_3)
#n.candidate = 150
#n_1 = 30
#n_2 = 100
#n_3 = 200

#prop_list <- list(c(0.4,0.3,0.3), c(0.4,0.2,0.2,0.2), c(0.4,0.15,0.15,0.15,0.15))
#prop_list <- list(c(0.4,0.2,0.2, 0.1, 0.1), c(0.4,0.3,0.2,0.1), c(0.5,0.3,0.2))

#k_1 = 5
#k_2 = 4
#k_3 = 3
#d = 3
#k = c(k_1, k_2, k_3)
#n_1 = 50
#n_2 = 100
#n_3 = 150

n = c(n_1, n_2, n_3)
delta_low = 0.6
delta_upper = 0.9
delta.candidate = seq(delta_low,delta_upper,0.025)
sigma = 1
filename <- paste("SNR", "unbalanced", "n1", n_1, 
                  "n2", n_2,
                  "n3", n_3,
                  "k1", k_1, 
                  "k2", k_2, 
                  "k3", k_3, 
                  "simu", replicates, 'd', d ,'delta_low',delta_low,'delta_upper',delta_upper, "constant_choice", sep = "_")
file = paste("results_new/",filename, ".csv", sep = "")
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd", "Hetero_1.2", "He_HLloyd_1.2", "Hetero_1.3", "He_HLloyd_1.3", "Hetero_1.4", "He_HLloyd_1.4", "Hetero_1.5", "He_HLloyd_1.5")
#n.index = 1
#delta.index = 1

#for (n.index in 1:length(n.candidate)){
for (delta.index in 1:length(delta.candidate)){
  sum_time <- 0
  time <- rep(0, 12)
  for(i in  1:replicates){
    start_t <- Sys.time()
    seed = i + 2023
    #n = rep(n.candidate[n.index],d)
    data = TBM_hetero.generator.unbalanced(n, k, prop_list, 40*(n[1]*n[2]*n[3])^(-delta.candidate[delta.index]/3), sigma, seed)
    
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
    
    time_tmp <- Sys.time()
    out0 <- Deflated_HeteroPCA.SC.tau(data$tensor, k, 0)
    z_hetero0 = out0[[1]]
    time[5] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.hetero0 = HO.Lloyd(data$tensor, z_hetero0)
    time[6] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    out2 <- Deflated_HeteroPCA.SC.tau(data$tensor, k, 2)
    z_hetero2 = out2[[1]]
    time[7] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.hetero2 = HO.Lloyd(data$tensor, z_hetero2)
    time[8] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    out5 <- Deflated_HeteroPCA.SC.tau(data$tensor, k, 5)
    z_hetero5 = out5[[1]]
    time[9] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.hetero5 = HO.Lloyd(data$tensor, z_hetero5)
    time[10] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    out10 <- Deflated_HeteroPCA.SC.tau(data$tensor, k, 10)
    z_hetero10 = out10[[1]]
    time[11] <- Sys.time() - time_tmp
    
    time_tmp <- Sys.time()
    z.Lloyd.hetero10 = HO.Lloyd(data$tensor, z_hetero10)
    time[12] <- Sys.time() - time_tmp
    
    HOSC_err = 1 - ARI(z.HOSC, data$labels, mode="averaged")
    Lloyd_err = 1 - ARI(z.Lloyd.HOSC, data$labels, mode="averaged")
    hetero_err = 1 - ARI(z_hetero, data$labels, mode="averaged")
    Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, data$labels, mode="averaged")
    hetero_err_0 = 1 - ARI(z_hetero0, data$labels, mode="averaged")
    Lloyd_hetero_err_0 = 1 - ARI(z.Lloyd.hetero0, data$labels, mode="averaged")
    hetero_err_2 = 1 - ARI(z_hetero2, data$labels, mode="averaged")
    Lloyd_hetero_err_2 = 1 - ARI(z.Lloyd.hetero2, data$labels, mode="averaged")
    hetero_err_5 = 1 - ARI(z_hetero5, data$labels, mode="averaged")
    Lloyd_hetero_err_5 = 1 - ARI(z.Lloyd.hetero5, data$labels, mode="averaged")
    hetero_err_10 = 1 - ARI(z_hetero10, data$labels, mode="averaged")
    Lloyd_hetero_err_10 = 1 - ARI(z.Lloyd.hetero10, data$labels, mode="averaged")
    error = c(HOSC_err,Lloyd_err,hetero_err,Lloyd_hetero_err,hetero_err_0,Lloyd_hetero_err_0,hetero_err_2,Lloyd_hetero_err_2,hetero_err_5,Lloyd_hetero_err_5,hetero_err_10,Lloyd_hetero_err_10)
    comp_time <- Sys.time() - start_t
    results <- NULL
    for(j in 1:12){
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
#}


results <- read.csv(file, sep = "")

results_summary <- results %>%
  group_by(n_1, n_2, n_3, delta, method)%>%
  summarise(error_mean = mean(error), error_sd = sd(error), recovery_rate = 1 - mean(error!=0))
filename_summary <- paste("result_summary_new/", paste(filename, "summary.csv", sep = "_"),sep = "")
write.csv(results_summary,file = filename_summary)