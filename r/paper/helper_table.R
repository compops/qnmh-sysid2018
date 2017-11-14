helper_table <- function(data, result, settings) {
  
  paramsTrace <- result$params
  iact <- rep(0, dim(paramsTrace)[2])
    
  for (k in 1:3) {
    acf_res <- acf(paramsTrace[, k], plot = FALSE, lag.max = 500)
    iact[k] <- 1 + 2 * sum(acf_res$acf)
  }
  acceptRate <- mean(result$accepted)
  fracHessiansCorrected <- result$no_hessians_corrected / settings$no_iters
  essPerSec <- result$time_per_iteration * max(iact)
  
  c(settings$simulation_name, acceptRate, fracHessiansCorrected, iact[1], iact[2], iact[3], result$time_per_iteration, essPerSec)
}