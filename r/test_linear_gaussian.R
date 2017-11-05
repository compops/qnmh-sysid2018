setwd("~/src/qmh-sysid2018/results-tests")
library("jsonlite")
source("helper_plotting.R")

algorithms <- list.dirs(full.names = FALSE)[-1]

offset <- c(0)
noItersToPlot <- 500
savePlotToFile <- TRUE
paramsScale <- c(0.1, 0.6, 0.35, 0.55, 0.9, 1.05)

for (i in 1:length(algorithms)) {
  algorithm <- algorithms[i]

  data <- read_json(paste("../results-tests/",
                    paste(algorithm, "/data.json", sep=""),
                    sep=""),
                    simplifyVector = TRUE)
  result <- read_json(paste("../results-tests/",
                      paste(algorithm, "/mcmc_output.json", sep=""),
                      sep=""),
                      simplifyVector = TRUE)
  settings <- read_json(paste("../results-tests/",
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