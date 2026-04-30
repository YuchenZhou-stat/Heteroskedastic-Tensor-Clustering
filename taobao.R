setwd("/Users/yczhou/Desktop/tensor_clustering/experiments_revision")
library(gridExtra)
source("tensor_clustering_method.R")
source("vis_function.R")

X = read.csv("Click_through.csv", header = F)
Y = array(0,c(100,50,24,8))
for (i in 1:nrow(X)){
  Y[X[i,1],X[i,2],X[i,3]+1,X[i,4]] = 1
}

# use clustering on all the aggregated data.
set.seed(1)
Y.sum = array(0,c(100,50,24))
for (i in 1:8){
  Y.sum = Y.sum + Y[,,,i]
}
Y.tensor = as.tensor(Y.sum)/8
tensor.data = as.tensor(Y.sum>0)
seed_list <- seq(2024,2033,length=10)
for(seed in seed_list){
  set.seed(seed)
  r_select = c(4,4,4)
  z.HOSC <- HO.SC(tensor.data,r_select)
  z.Lloyd.HOSC = HO.Lloyd(tensor.data, z.HOSC)
  z_hetero <- Deflated_HeteroPCA.SC(tensor.data,r_select)
  z_hetero <- z_hetero[[1]]
  z.Lloyd.hetero <- HO.Lloyd(tensor.data, z_hetero)
  g1 <- vis_time_cluster(z_hetero[[3]])
  g2 <- vis_time_cluster(z.Lloyd.hetero[[3]])
  
  g3 <- vis_time_cluster(z.HOSC[[3]])
  g4 <- vis_time_cluster(z.Lloyd.HOSC[[3]])
  
  ggsave(g1, filename = sprintf("taobao_fig/figVis_4clusters_%d_HHC.png",seed))
  ggsave(g2, filename = sprintf("taobao_fig/figVis_4clusters_%d_HHC_Hloyd.png",seed))
  ggsave(g3, filename = sprintf("taobao_fig/figVis_4clusters_%d_HSC.png",seed))
  ggsave(g4, filename = sprintf("taobao_fig/figVis_4clusters_%d_HSC_Hloyd.png",seed))
  
  #g <- grid.arrange(g1, g2, g3, g4, ncol = 2)
  #ggsave(g, filename = sprintf("taobao_fig/figVis_5clusters_%d.png",seed), )
}






set.seed(2024)
r_select = c(4,4,4)
z.HOSC.2024 <- HO.SC(tensor.data,r_select)
z.Lloyd.HOSC.2024 = HO.Lloyd(tensor.data, z.HOSC.2024)
z_hetero <- Deflated_HeteroPCA.SC(tensor.data,r_select)
z_hetero.2024 <- z_hetero[[1]]
z.Lloyd.hetero.2024 <- HO.Lloyd(tensor.data, z_hetero.2024)
ARI(z.Lloyd.hetero.2024, z.Lloyd.HOSC.2024, mode = "averaged")




# run 100 replicates and check if the obtained results occur most frequently
sum_time <- 0
time <- rep(0, 4)
replicates = 100
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
results <- NULL
for(i in  1:replicates){
  start_t <- Sys.time()
  seed = i + 2024
  set.seed(seed)
  
  time_tmp <- Sys.time()
  z.HOSC <- HO.SC(tensor.data,r_select)
  time[1] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z.Lloyd.HOSC = HO.Lloyd(tensor.data, z.HOSC)
  time[2] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z_hetero <- Deflated_HeteroPCA.SC(tensor.data,r_select)
  z_hetero <- z_hetero[[1]]
  time[3] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z.Lloyd.hetero <- HO.Lloyd(tensor.data, z_hetero)
  time[4] <- Sys.time() - time_tmp
  
  HOSC_err = 1 - ARI(z.HOSC, z.HOSC.2024, mode="averaged")
  Lloyd_err = 1 - ARI(z.Lloyd.HOSC, z.Lloyd.HOSC.2024, mode="averaged")
  hetero_err = 1 - ARI(z_hetero, z_hetero.2024, mode="averaged")
  Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, z.Lloyd.hetero.2024, mode="averaged")
  error = c(HOSC_err,Lloyd_err,hetero_err,Lloyd_hetero_err)
  comp_time <- Sys.time() - start_t
  for(j in 1:4){
    results <- rbind(results,data.frame(seed = i,
                                        error = error[j], method = methods[j], comp_time = time[j]))
  }
  sum_time <- sum_time + comp_time
}

############## Cluster mean estimation based on our method
W.hat.Lloyd.hetero = toweightmatrix(z.Lloyd.hetero.2024)
Z.hat.Lloyd.hetero = tomembershipmatrix(z.Lloyd.hetero.2024)
X.hat.Lloyd.hetero = ttl(ttl(tensor.data, W.hat.Lloyd.hetero ,1:3), Z.hat.Lloyd.hetero, 1:3)
tensor_mean.hetero = ttl(tensor.data, W.hat.Lloyd.hetero ,1:3)
X_inspired = ttl(tensor_mean.hetero, Z.hat.Lloyd.hetero, 1:3)

#real-data inspired simulation
sum_time <- 0
time <- rep(0, 4)
replicates = 100
n = c(100, 50, 24)
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
results_inspired1 <- NULL
for(i in  1:replicates){
  start_t <- Sys.time()
  seed = i + 2024
  set.seed(seed)
  Y_inspired = as.tensor(array(rbinom(prod(n),1,c(X.hat.Lloyd.hetero@data)),dim = n))
  
  time_tmp <- Sys.time()
  z.HOSC <- HO.SC(Y_inspired,r_select)
  time[1] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z.Lloyd.HOSC = HO.Lloyd(Y_inspired, z.HOSC)
  time[2] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z_hetero <- Deflated_HeteroPCA.SC(Y_inspired,r_select)
  z_hetero <- z_hetero[[1]]
  time[3] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z.Lloyd.hetero <- HO.Lloyd(Y_inspired, z_hetero)
  time[4] <- Sys.time() - time_tmp
  
  HOSC_err = 1 - ARI(z.HOSC, z.Lloyd.hetero.2024, mode="averaged")
  Lloyd_err = 1 - ARI(z.Lloyd.HOSC, z.Lloyd.hetero.2024, mode="averaged")
  hetero_err = 1 - ARI(z_hetero, z.Lloyd.hetero.2024, mode="averaged")
  Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, z.Lloyd.hetero.2024, mode="averaged")
  error = c(HOSC_err,Lloyd_err,hetero_err,Lloyd_hetero_err)
  comp_time <- Sys.time() - start_t
  for(j in 1:4){
    results_inspired1 <- rbind(results_inspired1,data.frame(seed = i,
                                                            error = error[j], method = methods[j], comp_time = time[j]))
  }
  sum_time <- sum_time + comp_time
}
results_summary_inspired1 <- results_inspired1 %>%
  group_by(method)%>%
  summarise(error_mean = mean(error), error_sd = sd(error), recovery_rate = 1 - mean(error!=0))
write.table(results_summary_inspired1, file = "results_summary_inspired1_taobao", sep = " ", row.names = FALSE, col.names = FALSE)



############## Cluster mean estimation based on Rungang's method. 
W.hat.Lloyd.HOSC = toweightmatrix(z.Lloyd.HOSC.2024)
Z.hat.Lloyd.HOSC = tomembershipmatrix(z.Lloyd.HOSC.2024)
X.hat.Lloyd.HOSC = ttl(ttl(tensor.data, W.hat.Lloyd.HOSC ,1:3), Z.hat.Lloyd.HOSC, 1:3)
tensor_mean = ttl(tensor.data, W.hat.Lloyd.HOSC ,1:3)

sum_time <- 0
time <- rep(0, 4)
replicates = 100
n = c(100, 50, 24)
methods <- c("HOSC", "HLloyd", "Hetero", "He_HLloyd")
results_inspired2 <- NULL
for(i in  1:replicates){
  start_t <- Sys.time()
  seed = i + 2024
  set.seed(seed)
  Y_inspired2 = as.tensor(array(rbinom(prod(n),1,c(X.hat.Lloyd.HOSC@data)),dim = n))
  
  time_tmp <- Sys.time()
  z.HOSC <- HO.SC(Y_inspired2,r_select)
  time[1] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z.Lloyd.HOSC = HO.Lloyd(Y_inspired2, z.HOSC)
  time[2] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z_hetero <- Deflated_HeteroPCA.SC(Y_inspired2,r_select)
  z_hetero <- z_hetero[[1]]
  time[3] <- Sys.time() - time_tmp
  
  time_tmp <- Sys.time()
  z.Lloyd.hetero <- HO.Lloyd(Y_inspired2, z_hetero)
  time[4] <- Sys.time() - time_tmp
  
  HOSC_err = 1 - ARI(z.HOSC, z.Lloyd.HOSC.2024, mode="averaged")
  Lloyd_err = 1 - ARI(z.Lloyd.HOSC, z.Lloyd.HOSC.2024, mode="averaged")
  hetero_err = 1 - ARI(z_hetero, z.Lloyd.HOSC.2024, mode="averaged")
  Lloyd_hetero_err = 1 - ARI(z.Lloyd.hetero, z.Lloyd.HOSC.2024, mode="averaged")
  error = c(HOSC_err,Lloyd_err,hetero_err,Lloyd_hetero_err)
  comp_time <- Sys.time() - start_t
  for(j in 1:4){
    results_inspired2 <- rbind(results_inspired2,data.frame(seed = i,
                                                            error = error[j], method = methods[j], comp_time = time[j]))
  }
  sum_time <- sum_time + comp_time
}
results_summary_inspired2 <- results_inspired2 %>%
  group_by(method)%>%
  summarise(error_mean = mean(error), error_sd = sd(error), recovery_rate = 1 - mean(error!=0))
write.table(results_summary_inspired2, file = "results_summary_inspired2_taobao", sep = " ", row.names = FALSE, col.names = FALSE)



