\name{plot.transiogram}
\Rdversion{1.1}
\alias{plot.transiogram}

\title{Plot One-dimensional Transiograms}

\description{
The function makes a graphical representation of transition probabilities by the use of transiogram.
}

\usage{\method{plot}{transiogram}(x, ..., main, legend = FALSE, ci = NULL)}

\arguments{
  \item{x}{an object of the class \code{transiogram}, typically with the output of the function \code{\link{transiogram}} or \code{\link{predict.tpfit}}.}
  \item{\dots}{other arguments to pass to the function \code{\link{plot}}.}
  \item{main}{the main title (on top) whose font and size are fixed.}
  \item{legend}{a logical value; if \code{TRUE}, the legend is plot on the bottom.}
  \item{ci}{a numerical value in the interval (0, 1) denoting the confidence of the interval around transition probabilities. If \code{NULL} (by default), no confidence interval is plotted.}
}

\details{
Transiogram is a diagram which is drawn for a single pair of categories in the direction \eqn{\phi}{phi}. It shows the transition probabilities in the \eqn{y}{y}-axis for some specific lags in the \eqn{x}{x}-axis.

Confidence intervals are computed on the log odds of the transition probabilities. The approximation of the confidence bounds is based on the delta method applied on the logistic transformation.
}

\value{An image is produced on the current graphics device. No values are returned.}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Li, W. (2007) Transiograms for Characterizing Spatial Variability of Soil Classes. \emph{Soil Science Society of America Journal}, \bold{71}(3), 881-893.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}
%%\note{
%%%%  ~~further notes~~
%%}

\seealso{\code{\link{tpfit}}, \code{\link{predict.tpfit}}, \code{\link{mixplot}}, \code{\link{image.multi_tpfit}}, \code{\link{plot}}}

\examples{\donttest{
data(ACM)

# Estimate empirical transition 
# probabilities by points
ETr <- transiogram(ACM$MAT3, ACM[, 1:3], c(0, 0, 1), 100, 100)

# Estimate the transition rate matrix
RTm <- tpfit(ACM$MAT3, ACM[, 1:3], c(0, 0, 1))

# Compute transition probabilities 
# from the one-dimensional MC model
TPr <- predict(RTm, lags = ETr$lags)

# Plot empirical transition probabilities
plot(ETr, type = "l", ci = 0.99)

# Plot theoretical transition probabilities
plot(TPr, type = "l")
}}

\keyword{spatial}
\keyword{distribution}
\keyword{hplot}
