library("RColorBrewer")

helper_plotting <- function(data, result, settings, algorithm, noItersToPlot,
                            savePlotToFile, paramsScale, folderToSaveTo) {

  plotColors = brewer.pal(8, "Dark2");
  plotColors = c(plotColors, plotColors)

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
  paramsEstStDev  <- apply(paramsTrace, 2, sd)

  # Estimate the log-volatility and the corresponding standad deviation
  statesEstMean    <- colMeans(statesTrace)
  statesEstStDev   <- apply(statesTrace, 2, sd)

  # Plot the parameter posterior estimate, solid black line indicate posterior mean
  # Plot the trace of the Markov chain after burn-in, solid black line indicate posterior mean
  if (savePlotToFile) {
    fileName <- paste(folderToSaveTo, paste(algorithm, ".pdf", sep=""), sep="")
    cairo_pdf(fileName, height = 10, width = 8)}
  
  if (length(paramsEstMean) == 3) {
    layout(matrix(c(1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 5, 3, byrow = TRUE))
  }
  if ((length(paramsEstMean) == 3) && (sum(statesEstMean) == 0)) {
    layout(matrix(c(1, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), 4, 3, byrow = TRUE))
  }
  
  if (length(paramsEstMean) == 4) {
    layout(matrix(c(1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), 6, 3, byrow = TRUE))
  }
  par(mar = c(4, 5, 0, 0))

  # Grid for plotting the data state estimates
  yGrid <- seq(1, length(obs))
  xGrid <- seq(1, length(obs) - 1)

  #---------------------------------------------------------------------------
  # Observations
  #---------------------------------------------------------------------------
  plot(
    obs,
    col = plotColors[1],
    lwd = 1,
    type = "l",
    xlab = "time",
    ylab = "observations",
    ylim = 1.2 * range(obs),
    xlim = c(0, ceiling(length(obs) / 100) * 100),
    bty = "n"
  )
  polygon(
    c(yGrid, rev(yGrid)),
    c(obs, rep(1.2 * range(obs)[1], length(yGrid))),
    border = NA,
    col = rgb(t(col2rgb(plotColors[1])) / 256, alpha = 0.25)
  )
  offset <- diff(1.2 * range(obs)) * 0.1
  x <- ceiling(length(obs) / 100) * 100
  y1 <- 1.2 * min(obs) + 5 * offset
  y2 <- 1.2 * min(obs) + 3 * offset
  y3 <- 1.2 * min(obs) + 1 * offset
  
  text(x, y1, pos=2, labels=algorithm)
  text(x, y2, pos=2, labels=paste("acc. prob:", round(mean(result$accepted), 2)))
  text(x, y3, pos=2, labels=paste("frac. hess. corr:", round(result$no_hessians_corrected/noIters, 2)))

  #---------------------------------------------------------------------------
  # Log-volatility
  #---------------------------------------------------------------------------
  if (sum(statesEstMean) != 0) {
    statesEstUpperCI <- statesEstMean[-1] + 1.96 * statesEstStDev[-1]
    statesEstLowerCI <- statesEstMean[-1] - 1.96 * statesEstStDev[-1]  
    plot(
      statesEstMean[-1],
      col = plotColors[2],
      lwd = 1.5,
      type = "l",
      xlab = "time",
      ylab = "state estimate",
      ylim = 1.2 * range(c(statesEstUpperCI, statesEstLowerCI)),
      xlim = c(0, ceiling(length(statesEstMean[-1]) / 100) * 100),
      bty = "n"
    )
  
    polygon(
      c(xGrid, rev(xGrid)),
      c(statesEstUpperCI, rev(statesEstLowerCI)),
      border = NA,
      col = rgb(t(col2rgb(plotColors[2])) / 256, alpha = 0.25)
    )
  }
  #---------------------------------------------------------------------------
  # Parameter posteriors
  #---------------------------------------------------------------------------

  grid <- seq(1, noItersToPlot, 1)
  paramsScale <- matrix(paramsScale, nrow = length(paramsEstMean), ncol = 2, byrow = TRUE)
  iact <- c()

  for (k in 1:length(paramsEstMean)) {

    # Histogram of the posterior
    hist(
      paramsTrace[, k],
      breaks = floor(sqrt(noIters)),
      col = rgb(t(col2rgb(plotColors[k+2])) / 256, alpha = 0.25),
      border = NA,
      xlab = paramsNames[k],
      ylab = "posterior estimate",
      main = "",
      xlim = paramsScale[k,],
      freq = FALSE
    )

    # Add lines for the kernel density estimate of the posterior
    kde <- density(paramsTrace[, k],
                   kernel = "e",
                   from = paramsScale[k, 1],
                   to = paramsScale[k, 2])
    lines(kde, lwd = 2, col = plotColors[k+2])

    # Plot the estimate of the posterior mean
    abline(v = paramsEstMean[k], lwd = 1, lty = "dotted")

    # Add lines for prior
    prior_grid <- seq(paramsScale[k, 1], paramsScale[k, 2], 0.01)
    if (k==1) {prior_values = dnorm(prior_grid, 0, 1)}
    if (k==2) {prior_values = dnorm(prior_grid, 0.5, 1)}
    if (k==3) {prior_values = dgamma(prior_grid, 2.0, 2.0)}
    if (k==4) {prior_values = dnorm(prior_grid, 0.0, 1.0)}
    lines(prior_grid, prior_values, col = "darkgrey")

    # Plot trace of the Markov chain
    plot(
      grid,
      paramsTrace[1:noItersToPlot, k],
      col = plotColors[k+2],
      type = "l",
      xlab = "iteration",
      ylab = paramsNames[k],
      ylim = paramsScale[k,],
      bty = "n"
    )
    polygon(
      c(grid, rev(grid)),
      c(paramsTrace[1:noItersToPlot, k], rep(-1, length(grid))),
      border = NA,
      col = rgb(t(col2rgb(plotColors[k+2])) / 256, alpha = 0.25)
    )
    abline(h = paramsEstMean[k], lwd = 1, lty = "dotted")

    if (length(result$iter_hessians_corrected) > 0) {
      rug(result$iter_hessians_corrected, ticksize = 0.1)
    }

    # Plot the autocorrelation function
    acf_res <- acf(paramsTrace[, k], plot = FALSE, lag.max = 100)
    plot(
      acf_res$lag,
      acf_res$acf,
      col = plotColors[k+2],
      type = "l",
      xlab = "iteration",
      ylab = paramsNamesACF[k],
      lwd = 2,
      ylim = c(-0.2, 1),
      bty = "n"
    )
    polygon(
      c(acf_res$lag, rev(acf_res$lag)),
      c(acf_res$acf, rep(0, length(acf_res$lag))),
      border = NA,
      col = rgb(t(col2rgb(plotColors[k+2])) / 256, alpha = 0.25)
    )
    abline(h = 1.96 / sqrt(noIters), lty = "dotted")
    abline(h = -1.96 / sqrt(noIters), lty = "dotted")

    iact <- c(iact, 1 + 2 * sum(acf_res$acf))
    text(100, 0.9, pos=2, labels=paste("IACT:",
                                 round(1 + 2 * sum(acf_res$acf), 2)))
  }

  if (savePlotToFile) {dev.off()}

  iact
  }