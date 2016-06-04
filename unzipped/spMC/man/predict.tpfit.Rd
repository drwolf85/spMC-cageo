\name{predict.tpfit}
\Rdversion{1.1}
\alias{predict.tpfit}

\title{Compute Theoretical One-dimensional Transiograms}

\description{
The function computes theoretical transition probabilities of a 1-D continuous-lag spatial Markov chain for a specified set of lags.
}

\usage{\method{predict}{tpfit}(object, lags, ...)}

\arguments{
  \item{object}{an object of the class \code{tpfit}, typically with the output of the function \code{\link{tpfit}}.}
  \item{lags}{a vector of 1-D lags.}
  \item{\dots}{further arguments passed from other methods.}
}

\details{
A 1-D continuous-lag spatial Markov chain is probabilistic model which involves a transition rate matrix \eqn{R}{R} computed for the direction \eqn{\phi}{phi}. It defines the transition probability \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through the entry \eqn{t_{jk}}{t_(jk)} of the following matrix
\deqn{T = \mbox{expm} (h R),}{T = expm(h R),}
where \eqn{h}{h} is a positive lag value.
}

\value{
An object of the class \code{transiogram} is returned. The function \code{\link{print.transiogram}} is used to print computed probabilities. The object is a list with the following components: 
  \item{Tmat}{a 3-D array containing the probabilities.}
  \item{lags}{a vector containing one-dimensional lags.}
  \item{type}{a character string which specifies that computed probabilities are theoretical.}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{tpfit}}, \code{\link{print.tpfit}}, \code{\link{plot.transiogram}}, \code{\link{transiogram}}, \code{\link{multi_tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
RTm <- tpfit(ACM$MAT3, ACM[, 1:3], c(0, 0, 1))

# Compute transition probabilities 
# from the one-dimensional MC model
predict(RTm, lags = 0:2/2)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
