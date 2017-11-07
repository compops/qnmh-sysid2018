setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("helper_plotting.R")

file_path = "../results-tests/mh-linear-gaussian/"

algorithms <- list.dirs(file_path, full.names = FALSE)[-1]

offset <- c(0)
noItersToPlot <- 1000
savePlotToFile <- TRUE
paramsScale <- c(0.1, 0.6, 0.35, 0.55, 0.9, 1.05)

for (i in 1:length(algorithms)) {
  algorithm <- algorithms[i]

  data <- read_json(paste(file_path,
                    paste(algorithm, "/data.json", sep=""),
                    sep=""),
                    simplifyVector = TRUE)
  result <- read_json(paste(file_path,
                      paste(algorithm, "/mcmc_output.json", sep=""),
                      sep=""),
                      simplifyVector = TRUE)
  settings <- read_json(paste(file_path,
                        paste(algorithm, "/settings.json", sep=""),
                        sep=""),
                        simplifyVector = TRUE)

  iact <- helper_plotting(data=data,
                          result=result,
                          settings=settings,
                          algorithm=algorithm,
                          noItersToPlot=noItersToPlot,
                          savePlotToFile=savePlotToFile,
                          paramsScale=paramsScale,
                          folderToSaveTo=file_path)
}