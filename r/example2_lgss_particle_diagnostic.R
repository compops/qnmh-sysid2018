setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("helper_plotting.R")

algorithms <- c("example2_mh2_",
                "example2_qmh_bfgs_",
                "example2_qmh_sr1_hyb_")
offset <- c(0)
noItersToPlot <- 1000
savePlotToFile <- TRUE
paramsScale <- c(0.1, 0.6, 0.35, 0.55, 0.9, 1.05)

for (i in 1:length(algorithms)) {
  for (j in 1:length(offset)) {
    algorithm <- paste(algorithms[i], offset[j], sep="")

    data <- read_json(paste("../results/",
                      paste(algorithm, "/data.json", sep=""),
                      sep=""),
                      simplifyVector = TRUE)
    result <- read_json(paste("../results/",
                        paste(algorithm, "/mcmc_output.json", sep=""),
                        sep=""),
                        simplifyVector = TRUE)
    settings <- read_json(paste("../results/",
                          paste(algorithm, "/settings.json", sep=""),
                          sep=""),
                          simplifyVector = TRUE)

    iact <- helper_plotting(data=data,
                            result=result,
                            settings=settings,
                            algorithm=algorithm,
                            noItersToPlot=noItersToPlot,
                            savePlotToFile=savePlotToFile,
                            paramsScale=paramsScale)
  }
}