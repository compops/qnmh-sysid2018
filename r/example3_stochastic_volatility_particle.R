setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("helper_plotting.R")

algorithms <- c("example3_mh2_") #,
                #"example3_qmh_bfgs_",
                #"example3_qmh_sr1_hyb_")
offset <- c(0)
noItersToPlot <- 1000
savePlotToFile <- TRUE
paramsScale <- c(-3, 3, 0.5, 1, 0, 1)

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