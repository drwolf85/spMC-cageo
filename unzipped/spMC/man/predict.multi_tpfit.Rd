\name{predict.multi_tpfit}
\Rdversion{1.1}
\alias{predict.multi_tpfit}

\title{Compute Theoretical Multidimensional Transiograms}

\description{
The function computes theoretical transition probabilities of a \eqn{d}{d}-D continuous-lag spatial Markov chain for a specified set of lags.
}

\usage{\method{predict}{multi_tpfit}(object, lags, byrow = TRUE, ...)}

\arguments{
  \item{object}{an object of the class \code{multi_tpfit}, typically with the output of the function \code{\link{multi_tpfit}}.}
  \item{lags}{a lag vector or matrix of \eqn{d}{d}-D lags.}
  \item{byrow}{a logical value; if \code{TRUE} (by default), each row of matrix argument \code{lags} will be considered as a lag vector.}
  \item{\dots}{further arguments passed from other methods.}
}

\details{
A \eqn{d}{d}-D continuous-lag spatial Markov chain is probabilistic model which is developed by interpolation of the transition rate matrices computed for the main directions. It defines the transition probability \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through the entry \eqn{t_{jk}}{t_(jk)} of the following matrix
\deqn{T = \mbox{expm} (\Vert h \Vert R),}{T = expm(||h|| R),}
where \eqn{h}{h} is the lag vector and the entries of \eqn{R}{R} are ellipsoidally interpolated.
}

\value{
An object of the class \code{multi_transiogram} is returned. The \code{\link{print.multi_transiogram}} function is used to print computed probabilities. The object is a list with the following components: 
  \item{Tmat}{a 3-D array containing the probabilities.}
  \item{lags}{a matrix containing the lag vectors.}
  \item{type}{a character string which specifies that computed probabilities are theoretical.}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{multi_tpfit}}, \code{\link{print.multi_tpfit}}, \code{\link{image.multi_tpfit}}, \code{\link{tpfit}}, \code{\link{transiogram}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# multidimensional MC model
RTm <- multi_tpfit(ACM$MAT3, ACM[, 1:3])

# Generate the matrix of 
# multidimensional lags
lags <- expand.grid(X=-1:1, Y=-1:1, Z=-1:1)
lags <- as.matrix(lags)

# Compute transition probabilities 
# from the multidimensional MC model
predict(RTm, lags)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
