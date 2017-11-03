setwd("~/src/qnmh-sysid2018/r")
library("jsonlite")
library("RColorBrewer")
plotColors = brewer.pal(8, "Dark2");

d <- read_json("../results/particle_smoother_linear_gaussian.json", simplifyVector = TRUE)

layout(matrix(1:9, 3, 3, byrow = TRUE))
par(mar = c(4, 5, 1, 1))

boxplot(t(d$log_like_mu), names=round(d$grid_mu, 1), frame=FALSE, col=plotColors[1], border=plotColors[1],
        xlab=expression(mu), ylab="log-likelihood", ylim=c(-2000, -1200))
abline(v=23, lty="dotted")
boxplot(t(d$gradient_mu), names=round(d$grid_mu, 1), frame=FALSE, col=plotColors[1], border=plotColors[1],
        xlab=expression(mu), ylab="gradient", ylim=c(-400, 800))
abline(h=0, lty="dotted")
abline(v=23, lty="dotted")
boxplot(t(d$nat_gradient_mu), names=round(d$grid_mu, 1), frame=FALSE, col=plotColors[1], border=plotColors[1],
        xlab=expression(mu), ylab="natural gradient", ylim=c(-1, 1.5))
abline(h=0, lty="dotted")
abline(v=23, lty="dotted")
boxplot(t(d$log_like_phi), names=round(d$grid_phi, 1), frame=FALSE, col=plotColors[2], border=plotColors[2],
        xlab=expression(phi), ylab="log-likelihood", ylim=c(-8000, 0))
abline(v=15, lty="dotted")
boxplot(t(d$gradient_phi), names=round(d$grid_phi, 1), frame=FALSE, col=plotColors[2], border=plotColors[2],
        xlab=expression(phi), ylab="gradient", ylim=c(-1000, 1500))
abline(h=0, lty="dotted")
abline(v=15, lty="dotted")
boxplot(t(d$nat_gradient_phi), names=round(d$grid_phi, 1), frame=FALSE, col=plotColors[2], border=plotColors[2],
        xlab=expression(phi), ylab="natural gradient", ylim=c(-8, 2))
abline(h=0, lty="dotted")
abline(v=15, lty="dotted")
boxplot(t(d$log_like_sigmav), names=round(d$grid_sigmav, 1), frame=FALSE, col=plotColors[3], border=plotColors[3],
       xlab=expression(mu), ylab="log-likelihood", ylim=c(-6000, 0))
abline(v=6, lty="dotted")
boxplot(t(d$gradient_sigmav), names=round(d$grid_sigmav, 1), frame=FALSE, col=plotColors[3], border=plotColors[3],
        xlab=expression(sigma[v]), ylab="gradient", ylim=c(-1000, 5000))
abline(h=0, lty="dotted")
abline(v=6, lty="dotted")
boxplot(t(d$nat_gradient_sigmav), names=round(d$grid_sigmav, 1), frame=FALSE, col=plotColors[3], border=plotColors[3],
        xlab=expression(sigma[v]), ylab="natural gradient", ylim=c(-40, 10))
abline(h=0, lty="dotted")
abline(v=6, lty="dotted")