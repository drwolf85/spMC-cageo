\name{tpfit_ils}
\Rdversion{1.1}
\alias{tpfit_ils}

\title{Iterated Least Squares Method for One-dimensional Model Parameters Estimation}

\description{
The function estimates the model parameters of a 1-D continuous lag spatial Markov chain by the use of the iterated least squares and the bound-constrained Lagrangian methods. Transition rates matrix along a user defined direction and proportions of categories are computed.
}

\usage{tpfit_ils(data, coords, direction, max.dist = Inf, mpoints = 20, 
          tolerance = pi/8, q = 10, echo = FALSE, ..., tpfit)}

\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%    direction vector (or versor) of chosen direction
%%     max.dist maximum distance for counting
%%      mpoints number of lags
%%    tolerance angle tolerance (in radians)
%%         echo logical value to print the optimization output
%%            q constant greater than one controlling the growth of rho
%%          ... further arguments to pass to nlminb function
%%        tpfit tpfit object for a further optimization
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{max.dist}{a numerical value which defines the maximum lag value. It's \code{Inf} by default.}
  \item{mpoints}{a numerical value which defines the number of lag intervals.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It's \code{pi/8} by default.}
  \item{q}{a numerical value greater than one for a constant which controls the growth of the penalization term in the loss function. It is equal to \code{10} by default.}
  \item{echo}{a logical value; if \code{TRUE}, the function prints some information about the optimization. It is \code{FALSE} by default.}
  \item{\dots}{other arguments to pass to the function \code{\link{nlminb}}.}
  \item{tpfit}{an object \code{tpfit} to optimize. If missing, the algorithm starts with a null transition rates matrix.}
}

\details{
A 1-D continuous-lag spatial Markov chain is probabilistic model which involves a transition rate matrix \eqn{R}{R} computed for the direction \eqn{\phi}{phi}. It defines the transition probability \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through the entry \eqn{t_{jk}}{t_(jk)} of the following matrix
\deqn{T = \mbox{expm} (h R),}{T = expm(h R),}
where \eqn{h}{h} is a positive lag value.

To calculate entries of the transition rate matrix, we need to minimize the discrepancies between the empirical transiogram (see \code{\link{transiogram}}) and the predicted transition probabilities.

By the use of the iterated least squares, the diagonal entries of \eqn{R}{R} are constrained to be negative,
while the off-diagonal transition rates are constrained to be positive. Further constraints are considered in order to obtain a proper transition rates matrix.
}

\value{
An object of the class \code{tpfit} is returned. The function \code{\link{print.tpfit}} is used to print the fitted model. The object is a list with the following components: 
  \item{coefficients}{the transition rates matrix computed for the user defined direction.}
  \item{prop}{a vector containing the proportions of each observed category.}
  \item{tolerance}{a numerical value which denotes the tolerance angle (in radians).}
}

\references{Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\section{Warning}{If the process is not stationary, the optimization algorithm does not converge.}

\seealso{\code{\link{predict.tpfit}}, \code{\link{print.tpfit}}, \code{\link{multi_tpfit_ils}}, \code{\link{transiogram}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
tpfit_ils(ACM$MAT3, ACM[, 1:3], c(0,0,1), 100)
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
