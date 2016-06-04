\name{transiogram}
\Rdversion{1.1}
\alias{transiogram}

\title{Empirical Transition Probabilities Estimation for 1-D MC}

\description{The function estimates transition probabilities matrices for a \eqn{1}{1}-D continuous lag spatial Markov chain.}

\usage{transiogram(data, coords, direction, max.dist = Inf, 
            mpoints = 20, tolerance = pi / 8, reverse = FALSE)}

\arguments{
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{max.dist}{a numerical value which defines the maximum lag value. It's \code{Inf} by default.}
  \item{mpoints}{a numerical value which defines the number of lag intervals.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It's \code{pi/8} by default.}
  \item{reverse}{a logical value. If \code{TRUE} the transition probabilities of the reversible chain are also computed. It's \code{FALSE} by default.}
}

\details{
Empirical probabilities are estimated by counting such pairs of observations which satisfy some properties, and by normalizing the result.

A generic pair of sample points \eqn{s_i}{s_i} and \eqn{s_j}{s_j}, where \eqn{i \neq j}{i != j}, must satisfy the following properties:
\itemize{
  \item \eqn{\Vert s_i - s_j \Vert \in [a, a+\frac{m}{n}],}{a < ||s_i - s_j|| < a + m / n,} where \eqn{a}{a} is a non negative real value, while \eqn{m}{m} denotes the maximum lag value (\code{max.dist}) and \eqn{n}{n} is the number of lag intervals (\code{mpoints}).
  \item the lag vector \eqn{h = s_i - s_j}{h = s_i - s_j} must have the same direction of the vector \eqn{\phi}{phi} (\code{direction}) with a certain angular \code{tolerance}.
}
}

\value{
An object of the class \code{transiogram} is returned. The function \code{\link{print.transiogram}} is used to print computed probabilities. The object is a list with the following components: 
  \item{Tmat}{a 3-D array containing the probabilities.}
  \item{LOSE}{a 3-D array containing the standard error calculated for the log odds of the transition probabilities.}
  \item{lags}{a vector containing one-dimensional lags.}
  \item{type}{a character string which specifies that computed probabilities are empirical.}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{predict.tpfit}}, \code{\link{predict.tpfit}}, \code{\link{plot.transiogram}}}

\examples{\donttest{
data(ACM)

# Estimate empirical transition 
# probabilities by points
transiogram(ACM$MAT3, ACM[, 1:3], c(0, 0, 1), 200, 5)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
