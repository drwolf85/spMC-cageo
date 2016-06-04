\name{multi_tpfit_me}
\Rdversion{1.1}
\alias{multi_tpfit_me}

\title{Maximum Entropy Method for Multidimensional Model Parameters Estimation}

\description{The function estimates the model parameters of a \eqn{d}{d}-D continuous lag spatial Markov chain. Transition rates matrices along axial directions and proportions of categories are computed.}

\usage{multi_tpfit_me(data, coords, tolerance = pi/8, max.it = 9000,
               rotation = NULL, mle = "avg")}

\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%    tolerance angle tolerance (in radians)
%%     rotation vector of rotation angles (in radians)
%%       mle argument to pass to the function tpfit
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It is \code{pi/8} by default.}
  \item{max.it}{a numerical value which denotes the maximum number of iterations to perform during the optimization phase. It is \code{9000} by default.}
  \item{rotation}{a numerical vector of length \eqn{d - 1}{d - 1} with rotation angles (in radians), in order to perform the main axes rotation. No rotation is performed by default.}
  \item{mle}{a character value to pass to the function \code{\link{tpfit}}. It is \code{"avg"} by default.}
}

\details{
A \eqn{d}{d}-D continuous-lag spatial Markov chain is probabilistic model which is developed by interpolation of the transition rate matrices computed for the main directions by the use of the function \code{\link{tpfit_me}}. It defines transition probabilities \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through
\deqn{\mbox{expm} (\Vert h \Vert R),}{expm(||h|| R),}
where \eqn{h}{h} is the lag vector and the entries of \eqn{R}{R} are ellipsoidally interpolated.

The ellipsoidal interpolation is given by
\deqn{\vert r_{jk} \vert = \sqrt{\sum_{i = 1}^d \left( \frac{h_i}{\Vert h \Vert} r_{jk, \mathbf{e}_i} \right)^2},}{|r_(jk, h)| = sqrt(sum_{i = 1}^d h_i / ||h|| r_(jk, e_i)),}
where \eqn{\mathbf{e}_i}{e_i} is a standard basis for a \eqn{d}{d}-D space.

If \eqn{h_i < 0}{h_i < 0} the respective entries \eqn{r_{jk, \mathbf{e}_i}}{r_(jk, e_i)} are replaced by \eqn{r_{jk, -\mathbf{e}_i}}{r_(jk, -e_i)}, which is computed as
\deqn{r_{jk, -\mathbf{e}_i} = \frac{p_k}{p_j} \, r_{kj, \mathbf{e}_i},}{r_(jk, -e_i) = p_k / p_j * r_(kj, e_i),}
where \eqn{p_k}{p_k} and \eqn{p_j}{p_j} respectively denote the proportions for the \eqn{k}{k}-th and \eqn{j}{j}-th categories. In so doing, the model may describe the anisotropy of the process.

When some entries of the rates matrices are not identifiable, it is suggested to vary the \code{tolerance} coefficient and the \code{rotation} angles. This problem may be also avoided if the input argument \code{mle} is to set to be \code{"mlk"}.
}

\value{
An object of the class \code{multi_tpfit} is returned. The function \code{\link{print.multi_tpfit}} is used to print the fitted model. The object is a list with the following components: 
  \item{coordsnames}{a character vector containing the name of each axis.}
  \item{coefficients}{a list containing the transition rates matrices computed for each axial direction.}
  \item{prop}{a vector containing the proportions of each observed category.}
  \item{tolerance}{a numerical value which denotes the tolerance angle (in radians).}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{predict.multi_tpfit}}, \code{\link{print.multi_tpfit}}, \code{\link{image.multi_tpfit}}, \code{\link{tpfit_me}}}

\examples{\donttest{
data(ACM)

# Estimate transition rates matrices and 
# proportions for the categorical variable MAT5
multi_tpfit_me(ACM$MAT5, ACM[, 1:3])

# Estimate transition rates matrices and
# proportions for the categorical variable MAT3
multi_tpfit_me(ACM$MAT3, ACM[, 1:3])

# Estimate transition rates matrices and
# proportions for the categorical variable PERM
multi_tpfit_me(ACM$PERM, ACM[, 1:3])
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
