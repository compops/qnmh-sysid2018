setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")

library("RColorBrewer")
plotColors = brewer.pal(8, "Dark2")
plotColors = c(plotColors, plotColors)

algorithms <-
  c("example1_mh0pre_0",
    "example1_mh1pre_0",
    "example1_mh_bfgs_0")
shortAlgorithmName <- c("pMH0", "pMH1", "qMH-dBFGS")
noItersToPlot <- 200
savePlotToFile <- TRUE
parameterToPlot <- 2
paramsScale <- c(0.4, 0.7)

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
  
#  if (i == 3) {
#    paramsTrace <- result$params[seq(1, length(result$params[,1]), 20), ]
#    paramsEstMean   <- colMeans(paramsTrace)
#  }
  grid <- seq(1, noItersToPlot, 1)
  
  # Histogram of the posterior
  hist(
    paramsTrace[, parameterToPlot],
    breaks = floor(sqrt(noIters)),
    col = rgb(t(col2rgb(plotColors[i + 5])) / 256, alpha = 0.25),
    border = NA,
    xlab = paramsNames[parameterToPlot],
    ylab = "posterior estimate",
    main = "",
    ylim = c(0, 12),
    xlim = paramsScale,
    freq = FALSE,
    cex.lab = 1.5,
    cex.axis = 1.5
  )
  
  text(0, 11, shortAlgorithmName[i], cex=2)
  
  # Add lines for the kernel density estimate of the posterior
  kde <- density(paramsTrace[, parameterToPlot],
                 kernel = "e",
                 from = paramsScale[1],
                 to = paramsScale[2])
  lines(kde, lwd = 2, col = plotColors[i + 5])
  
  # Plot the estimate of the posterior mean
  abline(v = paramsEstMean[parameterToPlot], lwd = 2, lty = "dotted")
  
  # Add lines for prior
  prior_grid <- seq(paramsScale[1], paramsScale[2], 0.01)
  prior_values = dnorm(prior_grid, 0.5, 1)
  lines(prior_grid, prior_values, col = "darkgrey", lwd = 2)
  
  # Plot trace of the Markov chain
  plot(
    grid,
    paramsTrace[1:noItersToPlot, parameterToPlot],
    col = plotColors[i + 5],
    type = "l",
    xlab = "iteration",
    ylab = paramsNames[parameterToPlot],
    ylim = paramsScale,
    bty = "n",
    cex.lab = 1.5,
    cex.axis = 1.5,
    lwd=2
  )
  polygon(
    c(grid, rev(grid)),
    c(paramsTrace[1:noItersToPlot, parameterToPlot], rep(-1, length(grid))),
    border = NA,
    col = rgb(t(col2rgb(plotColors[i + 5])) / 256, alpha = 0.25)
  )
  abline(h = paramsEstMean[parameterToPlot], lwd = 2, lty = "dotted")
  
  # Plot the autocorrelation function
  acf_res <- acf(paramsTrace[, parameterToPlot], plot = FALSE, lag.max = 500)
  plot(
    acf_res$lag,
    acf_res$acf,
    col = plotColors[i + 5],
    type = "l",
    xlab = "iteration",
    ylab = paramsNamesACF[parameterToPlot],
    lwd = 2,
    ylim = c(0, 1),
    bty = "n",
    cex.lab = 1.5,
    cex.axis = 1.5
  )
  polygon(
    c(acf_res$lag, rev(acf_res$lag)),
    c(acf_res$acf, rep(0, length(acf_res$lag))),
    border = NA,
    col = rgb(t(col2rgb(plotColors[i + 5])) / 256, alpha = 0.25)
  )
  abline(h = 1.96 / sqrt(noIters), lty = "dotted", lwd=2)
  abline(h = -1.96 / sqrt(noIters), lty = "dotted", lwd=2)
}

if (savePlotToFile) {
  dev.off()
}