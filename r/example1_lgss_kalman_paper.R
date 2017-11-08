setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
source("helper_plotting.R")


algorithms <-
  c("example1_mh0pre_0",
    "example1_mh1pre_0",
    "example1_mh_bfgs_0")
shortAlgorithmName <- c("pMH0", "pMH1", "qMH-dBFGS")
noItersToPlot <- 1000
savePlotToFile <- TRUE
parameterToPlot <- 2
paramsScale <- c(0.3, 0.7)

if (savePlotToFile) {
  fileName <-
    "~/projects/qnmh-sysid2018/paper/qnmh-sysid2018-draft1/figures/example1_lgss_kalman_paper.pdf"
  cairo_pdf(fileName, height = 10, width = 8)
}

layout(matrix(1:9, 3, 3, byrow = TRUE))
par(mar = c(4, 5, 1, 1))

for (i in 1:length(algorithms)) {
  algorithm <- algorithms[i]
  
  data <- read_json(paste(
    "../results/example1/",
    paste(algorithm, "/data.json.gz", sep = ""),
    sep = ""
  ),
  simplifyVector = TRUE)
  result <- read_json(paste(
    "../results/example1/",
    paste(algorithm, "/mcmc_output.json.gz", sep =
            ""),
    sep = ""
  ),
  simplifyVector = TRUE)
  settings <- read_json(paste(
    "../results/example1/",
    paste(algorithm, "/settings.json.gz", sep =
            ""),
    sep = ""
  ),
  simplifyVector = TRUE)
  
  paramsTrace <- result$params
  paramsEstMean   <- colMeans(paramsTrace)
  grid <- seq(1, noItersToPlot, 1)
  
  # Histogram of the posterior
  hist(
    paramsTrace[, parameterToPlot],
    breaks = floor(sqrt(noIters)),
    col = rgb(t(col2rgb(plotColors[i + 2])) / 256, alpha = 0.25),
    border = NA,
    xlab = paramsNames[parameterToPlot],
    ylab = "posterior estimate",
    main = "",
    ylim = c(0, 12),
    xlim = paramsScale,
    freq = FALSE
  )
  
  text(0.7, 11, shortAlgorithmName[i], pos = 2)
  
  # Add lines for the kernel density estimate of the posterior
  kde <- density(paramsTrace[, parameterToPlot],
                 kernel = "e",
                 from = paramsScale[1],
                 to = paramsScale[2])
  lines(kde, lwd = 2, col = plotColors[i + 2])
  
  # Plot the estimate of the posterior mean
  abline(v = paramsEstMean[parameterToPlot], lwd = 1, lty = "dotted")
  
  # Add lines for prior
  prior_grid <- seq(paramsScale[1], paramsScale[2], 0.01)
  prior_values = dnorm(prior_grid, 0.5, 1)
  lines(prior_grid, prior_values, col = "darkgrey", lwd = 1.5)
  
  # Plot trace of the Markov chain
  plot(
    grid,
    paramsTrace[1:noItersToPlot, parameterToPlot],
    col = plotColors[i + 2],
    type = "l",
    xlab = "iteration",
    ylab = paramsNames[k],
    ylim = paramsScale,
    bty = "n"
  )
  polygon(
    c(grid, rev(grid)),
    c(paramsTrace[1:noItersToPlot, parameterToPlot], rep(-1, length(grid))),
    border = NA,
    col = rgb(t(col2rgb(plotColors[i + 2])) / 256, alpha = 0.25)
  )
  abline(h = paramsEstMean[parameterToPlot], lwd = 1, lty = "dotted")
  
  if (length(result$iter_hessians_corrected) > 0) {
    rug(result$iter_hessians_corrected, ticksize = 0.1)
  }
  
  # Plot the autocorrelation function
  acf_res <-
    acf(paramsTrace[, parameterToPlot], plot = FALSE, lag.max = 100)
  plot(
    acf_res$lag,
    acf_res$acf,
    col = plotColors[i + 2],
    type = "l",
    xlab = "iteration",
    ylab = paramsNamesACF[parameterToPlot],
    lwd = 2,
    ylim = c(-0.2, 1),
    bty = "n"
  )
  polygon(
    c(acf_res$lag, rev(acf_res$lag)),
    c(acf_res$acf, rep(0, length(acf_res$lag))),
    border = NA,
    col = rgb(t(col2rgb(plotColors[i + 2])) / 256, alpha = 0.25)
  )
  abline(h = 1.96 / sqrt(noIters), lty = "dotted")
  abline(h = -1.96 / sqrt(noIters), lty = "dotted")
}

if (savePlotToFile) {
  dev.off()
}