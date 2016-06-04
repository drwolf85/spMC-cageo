\name{tpfit_me}
\Rdversion{1.1}
\alias{tpfit_me}

\title{Maximum Entropy Method for One-dimensional Model Parameters Estimation}

\description{
The function estimates the model parameters of a 1-D continuous lag spatial Markov chain by the use of the maximum entropy method. Transition rates matrix along a user defined direction and proportions of categories are computed.
}

\usage{tpfit_me(data, coords, direction, tolerance = pi/8,
         max.it = 9000, mle = "avg")}

\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%    direction vector (or versor) of chosen direction
%%    tolerance angle tolerance (in radians)
%%          mle logical value to pass to the function mlen
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{direction}{a \eqn{d}{d}-D numerical vector (or versor) which represents the chosen direction.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It is \code{pi/8} by default.}
  \item{max.it}{a numerical value which denotes the maximum number of iterations to perform during the optimization phase. It is \code{9000} by default.}
  \item{mle}{a character value to pass to the function \code{\link{mlen}}. It is \code{"avg"} by default.}
}

\details{
A 1-D continuous-lag spatial Markov chain is probabilistic model which involves a transition rate matrix \eqn{R}{R} computed for the direction \eqn{\phi}{phi}. It defines the transition probability \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through the entry \eqn{t_{jk}}{t_(jk)} of the following matrix
\deqn{T = \mbox{expm} (h R),}{T = expm(h R),}
where \eqn{h}{h} is a positive lag value.

To calculate entries of the transition rate matrix, we need to maximize the entropy of the transition probabilities of embedded occurrences along a given direction \eqn{\phi}{phi}. The entropy is defined as
\deqn{e = - \sum_{k}^K \sum_{j \neq k}^K \tau_{jk, \phi} \log \tau_{jk, \phi},}{e = E[log tau_(jk, phi)],}
where \eqn{\tau_{jk, \phi}}{tau_(jk, phi)} are transition probabilities of embedded occurrences. It is maximized by the use of the iterative proportion fitting method.

When some entries of the matrix \eqn{R}{R} are not identifiable, it is suggested to vary the \code{tolerance} coefficient or to set the input argument \code{mle} to \code{"mlk"}.}

\value{
An object of the class \code{tpfit} is returned. The function \code{\link{print.tpfit}} is used to print the fitted model. The object is a list with the following components: 
  \item{coefficients}{the transition rates matrix computed for the user defined direction.}
  \item{prop}{a vector containing the proportions of each observed category.}
  \item{tolerance}{a numerical value which denotes the tolerance angle (in radians).}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{predict.tpfit}}, \code{\link{print.tpfit}}, \code{\link{multi_tpfit_me}}}

\examples{\donttest{
data(ACM)

# Estimate the parameters of a 
# one-dimensional MC model
tpfit_me(ACM$MAT5, ACM[, 1:3], c(0,0,1))
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
