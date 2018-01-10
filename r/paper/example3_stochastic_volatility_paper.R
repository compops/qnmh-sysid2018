setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
library("RColorBrewer")
plotColors = brewer.pal(8, "Dark2")
plotColors = c(plotColors, plotColors)

algorithms <-
  list.dirs("../results/example3/", full.names = FALSE)[-1]

offset <- c(0)
noItersToPlot <- 350
savePlotToFile <- TRUE
paramsScale <- c(0.5, 2.5, 0.85, 1.0, 0.3, 0.7, -0.3, 0.3)
histPosteriorScale <- c(2.0, 25, 9, 7)

algorithm <- "example3_qmh_bfgs"

data <- read_json(paste(
  "../results/example3/",
  paste(algorithm, "/data.json.gz", sep = ""),
  sep = ""
),
simplifyVector = TRUE)
result <- read_json(paste(
  "../results/example3/",
  paste(algorithm, "/mcmc_output.json.gz", sep =
          ""),
  sep = ""
),
simplifyVector = TRUE)
settings <- read_json(paste(
  "../results/example3/",
  paste(algorithm, "/settings.json.gz", sep =
          ""),
  sep = ""
),
simplifyVector = TRUE)

paramsNames <- c(expression(mu),
                 expression(phi),
                 expression(sigma[v]),
                 expression(rho))
paramsNamesACF <- c(expression("ACF of " * mu),
                    expression("ACF of " * phi),
                    expression("ACF of " * sigma[v]),
                    expression("ACF of " * rho))

obs <- data$observations
noIters <- settings$no_iters - settings$no_burnin_iters

paramsTrace <- result$params
statesTrace <- result$states

# Estimate the posterior mean and the corresponding standard deviation
paramsEstMean   <- colMeans(paramsTrace)
paramsEstStDev     <- apply(paramsTrace, 2, sd)

# Estimate the log-volatility and the corresponding standad deviation
statesEstMean    <- colMeans(statesTrace)
statesEstStDev      <- apply(statesTrace, 2, sd)

# Plot the parameter posterior estimate, solid black line indicate posterior mean
# Plot the trace of the Markov chain after burn-in, solid black line indicate posterior mean
if (savePlotToFile) {
  fileName <- "../results/example3_stochastic_volatility_paper.pdf"
  cairo_pdf(fileName, height = 10, width = 8)
}

layout(matrix(c(1, 1, 1, 1, 2, 2, 2, 2, 3, 4, 5, 6), 3, 4, byrow = TRUE))
par(mar = c(4, 5, 1, 1))

# Observations
grid <-
  seq(as.POSIXct("2015-11-07 01:00:00 CET"),
      as.POSIXct("2017-11-01 01:00:00 CET"),
      by = "1 day")
plot(
  as.Date(grid),
  obs,
  col = plotColors[1],
  cex = 0.75,
  type = "p",
  pch = 19,
  xlab = "time",
  ylab = "observations",
  ylim = c(-15, 15),
  bty = "n",
  xaxt = "n",
  cex.axis = 1.5,
  cex.lab = 1.5
)
atVector1 = seq(as.POSIXct("2015-11-07 01:00:00 CET"),
                as.POSIXct("2017-11-07 01:00:00 CET"),
                by = "1 months")
atVector2 = seq(as.POSIXct("2015-11-07 01:00:00 CET"),
                as.POSIXct("2017-11-07 01:00:00 CET"),
                by = "6 months")
axis.Date(1, at = atVector1, labels = NA)
axis.Date(1, at = atVector2, format = "%b %y", cex.axis = 1.5)

statesEstUpperCI <- 1.96 * exp(0.5 * statesEstMean)
statesEstLowerCI <- -1.96 * exp(0.5 * statesEstMean)

polygon(
  c(as.Date(grid), rev(as.Date(grid))),
  c(statesEstUpperCI, rev(statesEstLowerCI)),
  border = NA,
  col = rgb(t(col2rgb(plotColors[2])) / 256, alpha = 0.25)
)


#---------------------------------------------------------------------------
# Log-volatility
#---------------------------------------------------------------------------

plot(as.Date(grid),
  statesEstMean,
  col = plotColors[2],
  type = "l",
  xlab = "time",
  ylab = "log-volatility",
  bty = "n",
  ylim = c(-4, 6),
  cex.axis = 1.5,
  cex.lab = 1.5,
  xaxt="n"
)
axis.Date(1, at = atVector1, labels = NA)
axis.Date(1, at = atVector2, format = "%b %y", cex.axis = 1.5)

statesEstUpperCI <- statesEstMean + 1.96 * statesEstStDev
statesEstLowerCI <- statesEstMean - 1.96 * statesEstStDev

polygon(
  c(as.Date(grid), rev(as.Date(grid))),
  c(statesEstUpperCI, rev(statesEstLowerCI)),
  border = NA,
  col = rgb(t(col2rgb(plotColors[2])) / 256, alpha = 0.25)
)


#---------------------------------------------------------------------------
# Parameter posteriors
#---------------------------------------------------------------------------

grid <- seq(1, noItersToPlot, 1)
paramsScale <- matrix(paramsScale,
                      nrow = 4,
                      ncol = 2,
                      byrow = TRUE)
iact <- c()

for (k in 1:4) {
  # Histogram of the posterior
  hist(
    paramsTrace[, k],
    breaks = floor(sqrt(noIters)),
    col = rgb(t(col2rgb(plotColors[k + 2])) / 256, alpha = 0.25),
    border = NA,
    xlab = paramsNames[k],
    ylab = "posterior estimate",
    main = "",
    xlim = paramsScale[k, ],
    ylim = c(0, histPosteriorScale[k]),
    freq = FALSE,
    cex.axis = 1.5,
    cex.lab = 1.5
  )

  # Add lines for the kernel density estimate of the posterior
  kde <- density(paramsTrace[, k],
                 kernel = "e",
                 from = paramsScale[k, 1],
                 to = paramsScale[k, 2])
  lines(kde, lwd = 2, col = plotColors[k + 2])

  # Plot the estimate of the posterior mean
  abline(v = paramsEstMean[k], lwd = 2, lty = "dotted")

  # Add lines for prior
  prior_grid <- seq(paramsScale[k, 1], paramsScale[k, 2], 0.01)
  if (k == 1) {
    prior_values = dnorm(prior_grid, 0, 1)
  }
  if (k == 2) {
    prior_values = dnorm(prior_grid, 0.95, 0.05)
  }
  if (k == 3) {
    prior_values = dgamma(prior_grid, 2.0, 10.0)
  }
  if (k == 4) {
    prior_values = dnorm(prior_grid, 0.0, 1.0)
  }
  lines(prior_grid, prior_values, col = "darkgrey", lwd = 3)

}

if (savePlotToFile) {
  dev.off()
}