.R_COLORS <- TRUE # If TRUE, colorful plot are produced, otherwise a grey scale is used

## Definition of the colors to be used
# Colors for the multidimensional and multidirectional transiograms
clen <- 500
R <- abs(seq(-.5, .25, length.out = clen))+.5
G <- seq(1, .25, length.out = clen)**.5; G <- .5 + .5 * (G - min(G))/diff(range(G))
B <- 1- abs(seq(-.5, .5, length.out = clen))
# Colors for histograms
scol <- colorRampPalette(c("#FEFED8", "#F2F280"))(6)[1]
scol <- rbind(scol, colorRampPalette(c("#C4FEC8", "#90A290"))(6)[1])
scol <- rbind(scol, colorRampPalette(c("#80CFF2", "#D8DEFE"))(6)[1])
# Convert colors to RGB format...
if (.R_COLORS) {
  myRGB <- rgb(R, G, B)
  cols <- myRGB
}
else { # ... or on a grey scale!
  cols <- grey(gg <- .21 * R + .72 * G + .07 * B)
  myRGB <- col2rgb(scol) / 255
  R <- myRGB[1,]
  G <- myRGB[2,]
  B <- myRGB[3,]
  scol <- grey(gg <- 0.5 * pmax(R, G, B) + 0.5 * pmin(R, G, B))
}

## Load the spMC package, set 16 CPU cores and load the data
library(spMC)
setCores(16L)
data(ACM)

set.seed(0)

## Plot empirical transiogram for Z-axis (also for the reversible chain)
plot(transiogram(ACM$MAT3, ACM[, 1:3], dirs[, 3L], 200, 25), type = "l", ci = .99)
plot(transiogram(ACM$MAT3, ACM[, 1:3], dirs[, 3L], 200, 25, reverse = TRUE), type = "l", ci = .99)

## Fit a multidirectional transiogram for multidimensional lags
pmt <- pemt(ACM$MAT3, ACM[, 1:3], 80, max.dist = c(250, 300, 5), mle = "trm") # THE EXECUTION OF THIS FUNCTION WILL REQUIRE MORE TIME
contour(pmt, mar = .7, nlevels = 5, ask = FALSE, main = "Non-elipsoidal transiogram")
persp(pmt, mar = .7, col = cols, phi = 45, theta = -45, border = NA, shade = .125, ask = FALSE, main = "Non-elipsoidal transiogram")
image(pmt, mar = .7, col = cols, breaks = 0:500/500, nlevels = 5, ask = FALSE, main = "Non-elipsoidal transiogram")

## Fit a theorical multidimensional transiogram
prm <- multi_tpfit(ACM$MAT3, ACM[, 1:3])

## Plot model transition probabilites
# section (X, Y)
image(prm, 80, max.dist=c(250, 300, 5), which.dire = 1:2, main = "Elipsoidal transiogram",
      mar = .7, col = cols, breaks = 0:500/500, nlevels = 5)
# section (X, Z)
image(prm, 80, max.dist=c(250, 300, 5), which.dire = c(1, 3), main = "Elipsoidal transiogram",
      mar = .7, col = cols, breaks = 0:500/500, nlevels = 5)
# section (Y, Z)
image(prm, 80, max.dist=c(250, 300, 5), which.dire = 2:3, main = "Elipsoidal transiogram",
      mar = .7, col = cols, breaks = 0:500/500, nlevels = 5)

## Perspective plots of transition probabilites
# section (X, Y)
persp(prm, 80, max.dist=c(250, 300, 5), which.dire = 1:2,
      mar = .7, col = cols, phi = 45, theta = -45, border = NA)
# section (X, Z)
persp(prm, 80, max.dist=c(250, 300, 5), which.dire = c(1, 3),
      mar = .7, col = cols, phi = 45, theta = -45, border = NA)
# section (Y, Z)
persp(prm, 80, max.dist=c(250, 300, 5), which.dire = 2:3,
      mar = .7, col = cols, phi = 45, theta = -45, border = NA)

## Forecasts
gx <- seq(min(ACM$X), max(ACM$X), length.out = 21)
gy <- seq(min(ACM$Y), max(ACM$Y), length.out = 21)
gz <- seq(min(ACM$Z) / 4, max(ACM$Z), length.out = 21)

grid <- expand.grid(gx, gy, gz)
time <- list()
setCores(8L)
# Indicator Kriging
time$ik <- system.time(IK <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "ik", knn = 32L))
# CK
time$ck <- system.time(CK <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "ck", knn = 32L))
# FP
time$fp <- system.time(FP <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "path", radius = 200, fixed = TRUE))
# RP
time$rp <- system.time(RP <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "path", radius = 200, fixed = FALSE))
# MCS
time$mcs <- system.time(MCS <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "mcs"))
# MCSknn
time$mcsknn <- system.time(MCSknn <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "mcs", knn = 32L))

## Forecasting Results
# library(rgl)
# plot3d(IK$X, IK$Y, IK$Z, col = 1L+as.integer(IK$Prediction))
# plot3d(IK$X, IK$Y, IK$Z, col = 1L+as.integer(CK$Prediction))
# plot3d(IK$X, IK$Y, IK$Z, col = 1L+as.integer(FP$Prediction))
# plot3d(IK$X, IK$Y, IK$Z, col = 1L+as.integer(RP$Prediction))
# plot3d(IK$X, IK$Y, IK$Z, col = 1L+as.integer(MCS$Prediction))
# plot3d(IK$X, IK$Y, IK$Z, col = 1L+as.integer(MCSknn$Prediction))

## Evaluation of time consuming with 16 processors
setCores(16L)
time16core <- list()
# Indicator Kriging
time16core$ik <- system.time(IK <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "ik", knn = 32L))
# CK
time16core$ck <- system.time(CK <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "ck", knn = 32L))
# FP
time16core$fp <- system.time(FP <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "path", radius = 200, fixed = TRUE))
# RP
time16core$rp <- system.time(RP <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "path", radius = 200, fixed = FALSE))
# MCS
time16core$mcs <- system.time(MCS <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "mcs"))
# MCSknn
time16core$mcsknn <- system.time(MCSknn <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "mcs", knn = 32L))

## Evaluation of time consuming with one processor
setCores(1L)
time1core <- list()
# Indicator Kriging
time1core$ik <- system.time(IK <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "ik", knn = 32L))
# CK
time1core$ck <- system.time(CK <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "ck", knn = 32L))
# FP
time1core$fp <- system.time(FP <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "path", radius = 200, fixed = TRUE))
# RP
time1core$rp <- system.time(RP <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "path", radius = 200, fixed = FALSE))
# MCS
time1core$mcs <- system.time(MCS <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "mcs"))
# MCSknn
time1core$mcsknn <- system.time(MCSknn <- sim(prm, ACM$MAT3, ACM[, 1:3], grid, method = "mcs", knn = 32L))
setCores(8L)


## Algorithmic efficiency (the results may differ from those presented in the paper)

secs <- sapply(time1core, "[[", 3)         # Serial-code-execution-time elapsed
secs <- rbind(secs, sapply(time, "[[", 3)) # Parallel-code-execution-time elapsed (8 corese)
secs <- rbind(secs, sapply(time16core, "[[", 3)) # Parallel-code-execution-time elapsed (16 corese)
colnames(secs) <- toupper(names(time1core))
rownames(secs) <- c("Serial code", "Parallel code (8 cores)", "Parallel code (16 cores)")
eff <- apply(secs, 2, function(x) x[1]/x[2:3])
inf <- apply(c(8, 16) / eff - 1, 2, min) / c(8, 16)
inf[inf > 1] <- 1
eff <- rbind(eff, 1 / inf)
eff[eff < 2] <- 1
rownames(eff) <- c("Parallel code (8 cores)", "Parallel code (16 cores)", "Parallel code (Inf cores)")

par(mfrow = c(1, 1), mar = c(5.1, 4.1, 4.1, 2.1))
barplot(eff, main = "Computational speed-up", col = scol, legend.text = TRUE, beside = TRUE,
        ylab = "Speed-up ratio", xlab = "Forecast methods", args.legend = list(x = "top"))

# secs <- sapply(time1core, "[[", 3)
# effm <- secs[which.max(secs)] / secs
# names(effm) <- toupper(names(time1core))
# barplot(effm, main = "Sequential efficiency", col = RColorBrewer:::brewer.pal(6L, "Pastel1"),
#         ylab = "Speed-up ratio", xlab = "Forecast methods", beside = TRUE)

# save.image("script.RData")
