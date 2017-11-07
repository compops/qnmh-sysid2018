setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("helper_plotting.R")

algorithms <- list.dirs("../results/example3/", full.names = FALSE)[-1]

offset <- c(0)
noItersToPlot <- 1000
savePlotToFile <- TRUE
paramsScale <- c(0.0, 2.0, 0.5, 1.0, 0.0, 1.0)

for (i in 1:length(algorithms)) {
  for (j in 1:length(offset)) {
    #algorithm <- paste(algorithms[i], offset[j], sep="")
    algorithm <- algorithms[i]
    
    data <- read_json(paste("../results/example3/",
                            paste(algorithm, "/data.json.gz", sep=""),
                            sep=""),
                      simplifyVector = TRUE)
    result <- read_json(paste("../results/example3/",
                              paste(algorithm, "/mcmc_output.json.gz", sep=""),
                              sep=""),
                        simplifyVector = TRUE)
    settings <- read_json(paste("../results/example3/",
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
                            folderToSaveTo="../results/example3/")
  }
}