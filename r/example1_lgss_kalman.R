setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("helper_plotting.R")

algorithms <- list.dirs("../results/example1/", full.names = FALSE)[-1]

offset <- c(0)
noItersToPlot <- 1000
savePlotToFile <- TRUE
paramsScale <- c(0.1, 0.6, 0.35, 0.55, 0.9, 1.05)

for (i in 1:length(algorithms)) {
  for (j in 1:length(offset)) {
    #algorithm <- paste(algorithms[i], offset[j], sep="")
    algorithm <- algorithms[i]

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

    iact <- helper_plotting(data=data,
                            result=result,
                            settings=settings,
                            algorithm=algorithm,
                            noItersToPlot=noItersToPlot,
                            savePlotToFile=savePlotToFile,
                            paramsScale=paramsScale,
                            folderToSaveTo="../results/example1/")
  }
}