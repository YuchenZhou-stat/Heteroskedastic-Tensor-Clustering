setwd("/Users/yczhou/Desktop/tensor_clustering/experiments_revision")
source("tensor_clustering_method.R")
#library(doParallel)

## simulated data.
set.seed(2023)

replicates = 100
const = 70

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

#k_1 = 3
#k_2 = 3
#k_3 = 3
#d = 3
#k = c(k_1, k_2, k_3)
#n_1 = 100
#n_2 = 100
#n_3 = 100

#prop_list <- list(c(1/3, 1/3, 1/3), c(1/3, 1/3, 1/3), c(1/3, 1/3, 1/3))
#prop_list <- list(c(0.5, 0.3, 0.2), c(0.5, 0.3, 0.2), c(0,5, 0.3, 0.2))

#prop_list <- list(c(0.4,0.2,0.2, 0.1, 0.1), c(0.4,0.3,0.2,0.1), c(0.5,0.3,0.2))

#k_1 = 5
#k_2 = 4
#k_3 = 3
#d = 3
#k = c(k_1, k_2, k_3)
#n_1 = 50
#n_2 = 100
#n_3 = 150
prop_list <- list(c(0.2,0.2,0.2,0.2,0.2), c(0.2,0.2,0.2,0.2,0.2), c(0.2,0.2,0.2,0.2,0.2))

k_1 = 5
k_2 = 5
k_3 = 5
d = 3
k = c(k_1, k_2, k_3)
n_1 = 150
n_2 = 150
n_3 = 150
#n_1 = 100
#n_2 = 100
#n_3 = 100

n = c(n_1, n_2, n_3)
hetero = 2
hetero.candidate = seq(0,hetero,0.2)
sigma = 1
filename <- paste("SNR", "unbalanced", "n1", n_1, 
                  "n2", n_2,
                  "n3", n_3,
                  "k1", k_1, 
                  "k2", k_2, 
                  "k3", k_3, 
                  "simu", replicates, 'd', d ,'hetero', hetero, 'signal_constant', const, sep = "_")
file = paste("results_new/",filename, ".csv", sep = "")
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
#i = 1
#hetero.index = 1


#for (n.index in 1:length(n.candidate)){
for (hetero.index in 1:length(hetero.candidate)){
  sum_time <- 0
  time <- rep(0, 4)
  for(i in  1:replicates){
    start_t <- Sys.time()
    seed = i + 2023
    #n = rep(n.candidate[n.index],d)
    #setting for n = 100
    data = TBM_hetero.generator.unbalanced2(n, k, prop_list, hetero.candidate[hetero.index], const*(n[1]*n[2]*n[3])^(-0.75/3), sigma, seed)
    #setting for n = 150
    #data = TBM_hetero.generator.unbalanced2(n, k, prop_list, hetero.candidate[hetero.index], 70*(n[1]*n[2]*n[3])^(-0.75/3), sigma, seed)
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
      results <- rbind(results,data.frame(seed = i, n_1 = n_1, n_2 = n_2, n_3 = n_3, hetero = hetero.candidate[hetero.index], 
                                          error = error[j], method = methods[j], comp_time = time[j]))
    }
    if(file.exists(file)){
      write.table(results, file = file, row.names = FALSE, col.names = FALSE, append = TRUE)
    }else{
      write.table(results, file = file, row.names = FALSE)
    }
    sum_time <- sum_time + comp_time
  }
  print(sprintf("finished n1 = %d, n2 = %d, n3 = %d, hetero = %f, within %s seconds", n_1, n_2, n_3, hetero.candidate[hetero.index], sum_time))
}

results <- read.csv(file, sep = "")
results_summary <- results %>%
  group_by(n_1, n_2, n_3, hetero, method)%>%
  summarise(error_mean = mean(error), error_sd = sd(error), recovery_rate = 1 - mean(error!=0))
filename_summary <- paste("result_summary_new/", paste(filename, "summary.csv", sep = "_"),sep="")
write.csv(results_summary,file = filename_summary)