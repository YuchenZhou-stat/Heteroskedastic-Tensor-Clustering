setwd("/Users/yczhou/Desktop/tensor_clustering/experiments_revision")
source("tensor_clustering_method.R")
#library(doParallel)

## simulated data.
set.seed(2023)

replicates = 100

r_total = 5
d = 3
r = rep(r_total,d)
n.candidate = 100
#delta_low = 0.7
#delta_upper = 1
delta_low = 0.5
delta_upper = 0.8
delta.candidate = seq(delta_low,delta_upper,0.025)
sigma = 1
filename <- paste("SNR", "n", paste(n.candidate, sep="", collapse="_"), "r", paste(r_total, sep="", collapse="_"), "simu", replicates, 'd', d ,'delta_low',delta_low,'delta_upper',delta_upper,sep = "_")
file = paste("results_new/",filename, ".csv", sep = "")
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
#n.index = 1
#delta.index = 1

for (n.index in 1:length(n.candidate)){
  for (delta.index in 1:length(delta.candidate)){
    sum_time <- 0
    time <- rep(0, 4)
    for(i in  1:replicates){
      start_t <- Sys.time()
      seed = i + 2023
      n = rep(n.candidate[n.index],d)
      data = TBM_hetero.generator(n, r, 40*n[1]^(-delta.candidate[delta.index]), sigma, seed)

      time_tmp <- Sys.time()
      z.HOSC = HO.SC(data$tensor, r)
      time[1] <- Sys.time() - time_tmp
      
      time_tmp <- Sys.time()
      z.Lloyd.HOSC = HO.Lloyd(data$tensor, z.HOSC)
      time[2] <- Sys.time() - time_tmp
      
      time_tmp <- Sys.time()
      myout <- Deflated_HeteroPCA.SC(data$tensor, r)
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
        results <- rbind(results,data.frame(seed = i, n = n.candidate[n.index], delta = delta.candidate[delta.index], 
                              error = error[j], method = methods[j], comp_time = time[j]))
      }
      if(file.exists(file)){
        write.table(results, file = file, row.names = FALSE, col.names = FALSE, append = TRUE)
      }else{
        write.table(results, file = file, row.names = FALSE)
      }
      sum_time <- sum_time + comp_time
    }
    print(sprintf("finished n = %d, delta = %f, within %s seconds", n.candidate[n.index], delta.candidate[delta.index], sum_time))
  }
}

results_summary <- results %>%
  group_by(n_1, n_2, n_3, delta, method)%>%
  summarise(error_mean = mean(error), error_sd = sd(error), recovery_rate = 1 - mean(error!=0))
filename_summary <- paste("result_summary_new/", paste(filename, "summary.csv", sep = "_"),sep = "")
write.csv(results_summary,file = filename_summary)