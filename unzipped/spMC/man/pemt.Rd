\name{pemt}
\Rdversion{1.1}
\alias{pemt}

\title{Multi-directional Transiograms Estimation}

\description{
The function computes the multi-directional transiograms without any ellipsoidal interpolation for \eqn{2}{2}-D sections.}

\usage{
pemt(data, coords, mpoints, which.dire, max.dist,
     tolerance = pi/8, rotation = NULL, mle = "avg")
}

\arguments{
%%         data vector of data
%%       coords coordinates matrix
%%      mpoints number of points per axes
%%   which.dire two choosen 1D directions
%%     max.dist vector of maximum distances
%%    tolerance angle tolerance (in radians)
%%     rotation vector of rotation angles (in radians) to pass to multi_tpfit
%%          mle logical value to pass to the function tpfit
  \item{data}{a categorical data vector of length \eqn{n}{n}.}
  \item{coords}{an \eqn{n \times d}{n x d} matrix where each row denotes the \eqn{d}{d}-D coordinates of data locations.}
  \item{mpoints}{the number of points per axes. It controls the accuracy of images to plot.}
  \item{which.dire}{a vector with two chosen axial directions. If omitted, all \eqn{2}{2}-D sections are plotted.}
  \item{max.dist}{a scalar or a vector of maximum length for the chosen axial directions.}
  \item{tolerance}{a numerical value for the tolerance angle (in radians). It's \code{pi/8} by default.}
  \item{rotation}{a numerical vector of length \eqn{d - 1}{d - 1} with rotation angles (in radians), in order to perform the main axes rotation when multidimensional transiogram is estimated. No rotation is performed by default. See \code{\link{multi_tpfit_ml}}.}
  \item{mle}{a character value to pass to the function \code{\link{tpfit_ml}}. It is \code{"avg"} by default.}
}

\details{
A multidimensional transiogram is a diagram which shows the transition probabilities for a single pair of categories. The probability is computed for any lag vector \eqn{h}{h} through 
\deqn{\mbox{expm} (\Vert h \Vert R_h),}{expm(||h|| R),}
where entries of \eqn{R_h}{R} are not ellipsoidally interpolated, but they are estimated for the direction specified by the vector \eqn{h}{h}.

In particular cases, some entries of the estimated matrix \eqn{R_h}{R} might be not finite, so that the exponential matrix is computable and the resulting transition probabilities are set to be \code{NaN}. If \code{mle = "mlk"}, this problem may be partially solved.

The exponential matrix is evaluated by the scaling and squaring algorithm.
}

\value{An object of class \code{pemt} is returned.}

\references{Carle, S. F., Fogg, G. E. (1997) Modelling Spatial Variability with One and Multidimensional Continuous-Lag Markov Chains. \emph{Mathematical Geology}, \bold{29}(7), 891-918.

Higham, N. J. (2008) \emph{Functions of Matrices: Theory and Computation}. Society for Industrial and Applied Mathematics.

Sartore, L. (2010) Geostatistical models for 3-D data. M.Phil. thesis, Ca' Foscari University of Venice.}

\author{Luca Sartore \email{drwolf85@gmail.com}}

\seealso{\code{\link{multi_tpfit_ml}}, \code{\link{tpfit_ml}}, \code{\link{image.pemt}}, \code{\link{plot.transiogram}}}

\examples{\donttest{
data(ACM)

# Compute a 2-D section of a
# multi-directional transiogram
pemt(ACM$MAT3, ACM[, 1:3], 2,
     max.dist = c(200, 200, 20), 
     which.dire=c(1, 3), mle = "mdn")
}}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{spatial}
\keyword{distribution}
