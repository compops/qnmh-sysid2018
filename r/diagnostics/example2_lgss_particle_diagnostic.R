setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("diagnostics/helper_plotting.R")

algorithms <- list.dirs("../results/example2/", full.names = FALSE)[-1]
algorithms <- algorithms[grepl(indexToPlot, algorithms)]

noItersToPlot <- 200
savePlotToFile <- TRUE
paramsScale <- c(0.0, 0.6, 0.3, 0.8, 0.8, 1.2)

for (i in 1:length(algorithms)) {
  algorithm <- algorithms[i]


  data <- read_json(paste("../results/example2/",
                    paste(algorithm, "/data.json.gz", sep=""),
                    sep=""),
                    simplifyVector = TRUE)
  result <- read_json(paste("../results/example2/",
                      paste(algorithm, "/mcmc_output.json.gz", sep=""),
                      sep=""),
                      simplifyVector = TRUE)
  settings <- read_json(paste("../results/example2/",
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
                          folderToSaveTo="../results/example2-diagplots/")
}
