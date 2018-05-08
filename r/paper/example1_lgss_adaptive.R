setwd("~/src/qnmh-sr1/r")
library("jsonlite")
library("xtable")
options(xtable.floating = FALSE)
options(xtable.timestamp = "")
source("paper/helper_table.R")

prefix <- c("mh_asr1", "mh_als")
suffix <- c("flip_0", "replace")

noSimulations <- 25
noAlgorithms <- length(prefix)

mem_length <- array(0, dim = c(10000, noSimulations, noAlgorithms))
step_size <- array(0, dim = c(10000, noSimulations, noAlgorithms))

for (i in 1:(noAlgorithms)) {
  for (j in 1:noSimulations) {
    algorithm <- paste(paste("example1", paste(prefix[i], j-1, sep="_"), sep="_"), suffix[i], sep="_")
    
    data <- read_json(paste("../results/example1/",
                            paste(algorithm, "/data.json.gz", sep=""),
                            sep=""),
                      simplifyVector = TRUE)
    result <- read_json(paste("../results/example1/",
                              paste(algorithm, "/mcmc_output.json.gz", sep=""),
                              sep=""),
                        simplifyVector = TRUE)
    settings <- read_json(paste("../results/example1/",
                                paste(algorithm, "/settings.json.gz", sep=""),
                                sep=""),
                          simplifyVector = TRUE)
    
    
    mem_length[, j, i] <- result$mem_lengths
    step_size[, j, i] <- result$step_sizes
    print(c(i, j))
  }    
}

layout(matrix(seq(1, 2*noAlgorithms), noAlgorithms, 2, byrow = TRUE))
par(mar = c(4, 5, 0, 0))

for (i in 1:(noAlgorithms)) {
  plot(step_size[, 1, i], type="l", xlab="iteration", ylab="adapted step size", bty = "n", ylim=c(0, 1.5))
  for (j in 2:noSimulations) {
    lines(step_size[, j, i])
  }
  plot(mem_length[, 1, i], type="l", xlab="iteration", ylab="adapted memory length", bty = "n", ylim=c(0, 50))
  for (j in 2:noSimulations) {
    lines(mem_length[, j, i])
  }  
}
