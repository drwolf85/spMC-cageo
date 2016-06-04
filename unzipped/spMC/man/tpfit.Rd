\name{tpfit}
\Rdversion{1.1}
\alias{tpfit}

\title{One-dimensional Model Parameters Estimation}

\description{
The function estimates the model parameters of a 1-D continuous lag spatial Markov chain. Transition rates matrix along a user defined direction and proportions of categories are computed.
}

\usage{tpfit(data, coords, direction, method = "ml",
      tolerance = pi/8, max.it = 9000, mle = "avg", ...)}

\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%    direction vector (or versor) of choosen direction
%%       method estimation method c("ml", "ils", "me")
%%    tolerance angle tolerance (in radians)
%%       max.it maximum number of iterations for the optimization (used only for the 'me' method)
%%          mle argument to pass to the function mlen (not used for the 'ils' method)
%%          ... further arguments to pass to tpfit_ils function, such as:
%%              * max.dist maximum distance for counting
%%              *  mpoints number of lags
%%              *        q constant greater than one controlling the growth of rho
%%              *     echo logical value to print the optimization output
%%              *      ... further arguments to pass to nlminb function
%%              *    tpfit tpfit object for a further optimization
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{method}{a character object specifying the method to estimate the transition rates. Possible choises are \code{"ml"} (by default) for the mean length method, \code{"ils"} for the iterated least squares and \code{"me"} for the maximum entropy method.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It's \code{pi/8} by default.}
  \item{max.it}{a numerical value which denotes the maximum number of iterations to perform during the optimization phase. It is \code{9000} by default and used only when the method is \code{"me"}.}
  \item{mle}{a character value to pass to the function \code{\link{mlen}}. It is \code{"avg"} by default and not use when the method is \code{"ils"}.}
  \item{\dots}{other arguments to pass to the functions \code{\link{tpfit_ml}}, \code{\link{tpfit_ils}} or \code{\link{tpfit_me}}.}
}

\details{
A 1-D continuous-lag spatial Markov chain is probabilistic model which involves a transition rate matrix \eqn{R}{R} computed for the direction \eqn{\phi}{phi}. It defines the transition probability \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through the entry \eqn{t_{jk}}{t_(jk)} of the following matrix
\deqn{T = \mbox{expm} (h R),}{T = expm(h R),}
where \eqn{h}{h} is a positive lag value.

Three methods are available to calculate entries of the transition rate matrix. The mean length method is performed by the use of the function \code{\link{tpfit_ml}}, the iterated least squares are applied through the function \code{\link{tpfit_ils}}, while the function \code{\link{tpfit_me}} implements the maximum entropy method.
}

\value{
An object of the class \code{tpfit} is returned. The function \code{\link{print.tpfit}} is used to print the fitted model. The object is a list with the following components: 
  \item{coefficients}{the transition rates matrix computed for the user defined direction.}
  \item{prop}{a vector containing the proportions of each observed category.}
  \item{tolerance}{a numerical value which denotes the tolerance angle (in radians).}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{predict.tpfit}}, \code{\link{print.tpfit}}, \code{\link{multi_tpfit}}, \code{\link{transiogram}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
tpfit(ACM$MAT5, ACM[, 1:3], c(0, 0, 1))
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
