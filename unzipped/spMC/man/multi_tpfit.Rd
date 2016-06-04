\name{multi_tpfit}
\Rdversion{1.1}
\alias{multi_tpfit}

\title{Multidimensional Model Parameters Estimation}

\description{The function estimates the model parameters of a \eqn{d}{d}-D continuous lag spatial Markov chain. Transition rates matrices along axial directions and proportions of categories are computed.}

\usage{multi_tpfit(data, coords, method = "ml", tolerance = pi/8,
            rotation = NULL, max.it = 9000, mle = "avg", ...)}
\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%       method estimation method c("ml", "ils", "me")
%%    tolerance angle tolerance (in radians)
%%     rotation vector of rotation angles (in radians)
%%       max.it maximum number of iterations for the optimization (used only for the 'me' method)
%%          mle argument to pass to the function mlen (not used for the 'ils' method)
%%          ... further arguments to pass to tpfit_ils function, such as:
%%              * max.dist maximum distance for counting
%%              *  mpoints number of lags
%%              *        q constant greater than one controlling the growth of rho
%%              *     echo logical value to print the optimization output
%%              *      ... further arguments to pass to nlminb function
%%              *    mtpfit mulit_tpfit object for a further optimization
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{method}{a character object specifying the method to estimate the transition rates. Possible choises are \code{"ml"} (by default) for the mean length method, \code{"ils"} for the iterated least squares and \code{"me"} for the maximum entropy method.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It's \code{pi/8} by default.}
  \item{rotation}{a numerical vector of length \eqn{d - 1}{d - 1} with rotation angles (in radians), in order to perform the main axes rotation. No rotation is performed by default.}
  \item{max.it}{a numerical value which denotes the maximum number of iterations to perform during the optimization phase. It is \code{9000} by default and used only when the method is \code{"me"}.}
  \item{mle}{a character value to pass to the function \code{\link{tpfit}}. It is \code{"avg"} by default and not use when the method is \code{"ils"}.}
  \item{\dots}{other arguments to pass to the functions \code{\link{multi_tpfit_ml}}, \code{\link{multi_tpfit_ils}} or \code{\link{multi_tpfit_me}}.}
}

\details{
A \eqn{d}{d}-D continuous-lag spatial Markov chain is probabilistic model which is developed by interpolation of the transition rate matrices computed for the main directions. It defines transition probabilities \eqn{\Pr(Z(s + h) = z_k | Z(s) = z_j)}{Pr(Z(s + h) = z_k | Z(s) = z_j)} through
\deqn{\mbox{expm} (\Vert h \Vert R),}{expm(||h|| R),}
where \eqn{h}{h} is the lag vector and the entries of \eqn{R}{R} are ellipsoidally interpolated.

The ellipsoidal interpolation is given by
\deqn{\vert r_{jk} \vert = \sqrt{\sum_{i = 1}^d \left( \frac{h_i}{\Vert h \Vert} r_{jk, \mathbf{e}_i} \right)^2},}{|r_(jk, h)| = sqrt(sum_{i = 1}^d h_i / ||h|| r_(jk, e_i)),}
where \eqn{\mathbf{e}_i}{e_i} is a standard basis for a \eqn{d}{d}-D space.

If \eqn{h_i < 0}{h_i < 0} the respective entries \eqn{r_{jk, \mathbf{e}_i}}{r_(jk, e_i)} are replaced by \eqn{r_{jk, -\mathbf{e}_i}}{r_(jk, -e_i)}, which is computed as
\deqn{r_{jk, -\mathbf{e}_i} = \frac{p_k}{p_j} \, r_{kj, \mathbf{e}_i},}{r_(jk, -e_i) = p_k / p_j * r_(kj, e_i),}
where \eqn{p_k}{p_k} and \eqn{p_j}{p_j} respectively denote the proportions for the \eqn{k}{k}-th and \eqn{j}{j}-th categories. In so doing, the model may describe the anisotropy of the process.
}

\value{
An object of the class \code{multi_tpfit} is returned. The function \code{\link{print.multi_tpfit}} is used to print the fitted model. The object is a list with the following components: 
  \item{coordsnames}{a character vector containing the name of each axis.}
  \item{coefficients}{a list containing the transition rates matrices computed for each axial direction.}
  \item{prop}{a vector containing the proportions of each observed category.}
  \item{tolerance}{a numerical value which denotes the tolerance angle (in radians).}
}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{predict.multi_tpfit}}, \code{\link{print.multi_tpfit}}, \code{\link{image.multi_tpfit}}, \code{\link{tpfit}}}

\examples{\donttest{
data(ACM)

# Estimate transition rates matrices and 
# proportions for the categorical variable MAT5
multi_tpfit(ACM$MAT5, ACM[, 1:3])

# Estimate transition rates matrices and
# proportions for the categorical variable MAT3
multi_tpfit(ACM$MAT3, ACM[, 1:3])

# Estimate transition rates matrices and
# proportions for the categorical variable PERM
multi_tpfit(ACM$PERM, ACM[, 1:3])
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
\keyword{models}
