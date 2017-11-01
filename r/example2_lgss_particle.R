setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
library("RColorBrewer")

plot_colors = brewer.pal(8, "Dark2");
plot_colors = c(plot_colors, plot_colors)

algorithm <- "example2_mh2"
#algorithm <- "example2_qmh_bfgs"
#algorithm <- "example2_qmh_sr1_hyb"

no_iters_to_plot <- 1000
save_plot_to_file <- TRUE

data <- read_json(paste("../results/",
                  paste(algorithm, "/data.json", sep=""),
                  sep=""),
                  simplifyVector = TRUE)
result <- read_json(paste("../results/",
                    paste(algorithm, "/mcmc_output.json", sep=""),
                    sep=""),
                    simplifyVector = TRUE)
settings <- read_json(paste("../results/",
                      paste(algorithm, "/settings.json", sep=""),
                      sep=""),
                      simplifyVector = TRUE)

########################################################################################
obs <- data$observations
no_iters <- settings$no_iters - settings$no_burnin_iters

trace_params <- result$params
trace_states <- result$states

# Estimate the posterior mean and the corresponding standard deviation
params_est_mean   <- colMeans(trace_params)
params_est_sd <- apply(trace_params, 2, sd)

# Estimate the log-volatility and the corresponding standad deviation
state_est_mean    <- colMeans(trace_states)
state_est_sd  <- apply(trace_states, 2, sd)

# Plot the parameter posterior estimate, solid black line indicate posterior mean
# Plot the trace of the Markov chain after burn-in, solid black line indicate posterior mean
if (save_plot_to_file) {cairo_pdf(paste(algorithm, ".pdf", sep=""),  height = 10, width = 8)}

layout(matrix(c(1, 1, 1, 2, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), 5, 3, byrow = TRUE))
par(mar = c(4, 5, 0, 0))

# Grid for plotting the data state estimates
gridy <- seq(1, length(obs))
gridx <- seq(1, length(obs) - 1)

#---------------------------------------------------------------------------
# Observations
#---------------------------------------------------------------------------
plot(
  obs,
  col = plot_colors[1],
  lwd = 1,
  type = "l",
  xlab = "time",
  ylab = "observations",
  ylim = c(-6, 6),
  bty = "n"
)
polygon(
  c(gridy, rev(gridy)),
  c(obs, rep(-6, length(gridy))),
  border = NA,
  col = rgb(t(col2rgb(plot_colors[1])) / 256, alpha = 0.25)
)
text(1000, -2, pos=2, labels=algorithm)
text(1000, -3.5, pos=2, labels=paste("acc. prob:", round(mean(result$accepted), 2)))
text(1000, -5, pos=2, labels=paste("frac. hess. corr:", round(result$no_hessians_corrected/no_iters, 2)))

#---------------------------------------------------------------------------
# Log-volatility
#---------------------------------------------------------------------------
plot(
  state_est_mean[-1],
  col = plot_colors[2],
  lwd = 1.5,
  type = "l",
  xlab = "time",
  ylab = "state estimate",
  ylim = c(-6, 6),
  bty = "n"
)
state_est_mean_upper <- state_est_mean[-1] + 1.96 * state_est_sd[-1]
state_est_mean_lower <- state_est_mean[-1] - 1.96 * state_est_sd[-1]

polygon(
  c(gridx, rev(gridx)),
  c(state_est_mean_upper, rev(state_est_mean_lower)),
  border = NA,
  col = rgb(t(col2rgb(plot_colors[2])) / 256, alpha = 0.25)
)

#---------------------------------------------------------------------------
# Parameter posteriors
#---------------------------------------------------------------------------

grid <- seq(1, no_iters_to_plot, 1)
params_names <- c(expression(mu), expression(phi), expression(sigma[v]))
params_names_acf <- c(expression("ACF of " * mu), expression("ACF of " * phi), expression("ACF of " * sigma[v]))
params_scale <- c(0.1, 0.6, 0.35, 0.55, 0.9, 1.05)
params_scale <- matrix(params_scale, nrow = 3, ncol = 2, byrow = TRUE)
iact <- c()

for (k in 1:3) {

  # Histogram of the posterior
  hist(
    trace_params[, k],
    breaks = floor(sqrt(no_iters)),
    col = rgb(t(col2rgb(plot_colors[k+2])) / 256, alpha = 0.25),
    border = NA,
    xlab = params_names[k],
    ylab = "posterior estimate",
    main = "",
    xlim = params_scale[k,],
    freq = FALSE
  )

  # Add lines for the kernel density estimate of the posterior
  kde <- density(trace_params[, k], kernel = "e", from = params_scale[k, 1], to = params_scale[k, 2])
  lines(kde, lwd = 2, col = plot_colors[k+2])

  # Plot the estimate of the posterior mean
  abline(v = params_est_mean[k], lwd = 1, lty = "dotted")

  # Add lines for prior
  prior_grid <- seq(params_scale[k, 1], params_scale[k, 2], 0.01)
  if (k==1) {prior_values = dnorm(prior_grid, 0, 1)}
  if (k==2) {prior_values = dnorm(prior_grid, 0.5, 1)}
  if (k==3) {prior_values = dgamma(prior_grid, 2.0, 2.0)}
  lines(prior_grid, prior_values, col = "darkgrey")

  # Plot trace of the Markov chain
  plot(
    grid,
    trace_params[1:no_iters_to_plot, k],
    col = plot_colors[k+2],
    type = "l",
    xlab = "iteration",
    ylab = params_names[k],
    ylim = params_scale[k,],
    bty = "n"
  )
  polygon(
    c(grid, rev(grid)),
    c(trace_params[1:no_iters_to_plot, k], rep(-1, length(grid))),
    border = NA,
    col = rgb(t(col2rgb(plot_colors[k+2])) / 256, alpha = 0.25)
  )
  abline(h = params_est_mean[k], lwd = 1, lty = "dotted")

  if (length(result$iter_hessians_corrected) > 0) {
    rug(result$iter_hessians_corrected, ticksize = 0.1)
  }

  # Plot the autocorrelation function
  acf_res <- acf(trace_params[, k], plot = FALSE, lag.max = 100)
  plot(
    acf_res$lag,
    acf_res$acf,
    col = plot_colors[k+2],
    type = "l",
    xlab = "iteration",
    ylab = params_names_acf[k],
    lwd = 2,
    ylim = c(-0.2, 1),
    bty = "n"
  )
  polygon(
    c(acf_res$lag, rev(acf_res$lag)),
    c(acf_res$acf, rep(0, length(acf_res$lag))),
    border = NA,
    col = rgb(t(col2rgb(plot_colors[k+2])) / 256, alpha = 0.25)
  )
  abline(h = 1.96 / sqrt(no_iters), lty = "dotted")
  abline(h = -1.96 / sqrt(no_iters), lty = "dotted")

  iact <- c(iact, 1 + 2 * sum(acf_res$acf))
  text(100, 0.9, pos=2, labels=paste("IACT:", round(1 + 2 * sum(acf_res$acf), 2)))
}

if (save_plot_to_file) {dev.off()}

iact