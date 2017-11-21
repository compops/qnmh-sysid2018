setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("diagnostics/helper_plotting.R")

algorithms <- list.dirs("../results/example3/", full.names = FALSE)[-1]

offset <- c(0)
noItersToPlot <- 200
savePlotToFile <- FALSE
paramsScale <- c(0.0, 2.5, 0.85, 1.0, 0.2, 0.7, -0.2, 0.2)

for (i in 1:length(algorithms)) {
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
                          folderToSaveTo="../results/example3-diagplots/")
}

